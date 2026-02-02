import enum
import os
from typing import Optional, List, Tuple, Dict
import unicodedata
import backoff

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image

"""
Data:
4. Additional Checks.
5. Bullet Checks?


Images:
1. Image URLs
2. Downloaded Images
3. Resize images.
4. Some checks.
"""


class SevenScraper:
    def __init__(
        self,
        outputs_folder: str = "./outputs",
        assets_folder: str = "./assets",
        min_img_size: Tuple[int, int] = (550, 550),
    ) -> None:
        self.assets_folder = assets_folder
        self.outputs_folder = outputs_folder
        self.min_img_size = min_img_size

        if not os.path.exists(self.outputs_folder):
            os.makedirs(self.outputs_folder, exist_ok=True)
        if not os.path.exists(self.assets_folder):
            os.makedirs(self.assets_folder, exist_ok=True)

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_tries=5,
    )
    def _get_page_source(self, url: str) -> Optional[bytes]:
        resp = requests.get(url)
        resp.raise_for_status()

        return resp.content

    def make_soup_obj(self, url: str) -> Optional[bool]:
        try:
            content = self._get_page_source(url)
        except requests.exceptions.RequestException:
            return None

        self.soup = BeautifulSoup(content, "html.parser")

        return True

    def get_title(self) -> str:
        title = self.soup.select_one(".product-name-normal")

        return title.text if title else ""

    def get_price(self) -> str:
        raw_price = self.soup.select_one("#ProductPrice")

        if not raw_price:
            return ""

        return raw_price.text.strip().replace("$","")

    def get_bullets(self) -> List[str]:
        raw_bullets = self.soup.select_one(".panel-body")

        if not raw_bullets:
            return []

        split_bullets = raw_bullets.text.strip().strip("//").strip().split("//")

        return [unicodedata.normalize("NFKD", b.strip()) for b in split_bullets]

    def bullet_checks(self, expected_bullets: int):
        return len(self.get_bullets()) == expected_bullets

    def get_description(self) -> str:
        raw_desc = self.soup.select_one(".top-description")

        if not raw_desc:
            return ""

        raw_desc = raw_desc.text.strip().replace("\n", " ")

        # Unicode noramalize
        raw_desc = unicodedata.normalize("NFKD", raw_desc)

        # Add html tags
        raw_desc += "<BR><BR>"
        # Add bullets
        desc = raw_desc + "\n\n" + "<BR>\n".join(self.get_bullets()).rstrip("<BR>\n")

        return desc

    def _write_images_sep_folders(self, img_name: str, img: Image.Image):
        img_name_wout_extension = img_name.split(".")[0].strip()
        img_folder = f"{self.assets_folder}/{img_name_wout_extension}"
        img_path = f"{img_folder}/{img_name}"

        if not os.path.exists(img_folder):
            os.makedirs(img_folder, exist_ok=True)

        img.save(img_path)

    def _write_images_same_folder(self, img_name: str, img: Image.Image):
        img_path = f"{self.assets_folder}/{img_name}"

        img.save(img_path)

    def _image_download_and_save(self, url: str, img_name: str, folderize: bool) -> None:
        img = Image.open(requests.get(url, stream=True).raw)

        if img.mode != "RGB":
            img = img.convert("RGB")

        if any(sz <= 500 for sz in img.size):
            img = img.resize(self.min_img_size).convert("RGB")

        (
            self._write_images_sep_folders(img_name=img_name, img=img)
            if folderize
            else self._write_images_same_folder(img_name=img_name, img=img)
        )

    def _is_main_img_bg_color_white(self, url: str) -> bool:
        im = Image.open(requests.get(url, stream=True).raw)

        prominent_color = max(im.getcolors(im.size[0] * im.size[1]))[1]
        return True if prominent_color == (255, 255, 255) else False

    @staticmethod
    def _get_image_metadata(asin: str, index: int) -> dict:
        is_main_image = index == 0
        img_name = f"{asin}.main.jpg" if is_main_image else f"{asin}.pt0{index}.jpg"
        image_url_col_name = "main" if is_main_image else f"pt0{index}"

        return {"img_name": img_name, "image_url_col_name": image_url_col_name}

    def get_images(self, row: Dict[str, str], folderize: bool = False) -> Optional[Dict[str, str]]:
        image_meta = {}
        asin = row.get("ASIN")
        if not asin:
            asin = row["Seller SKU"]

        print("Fetching images for ASIN: ", asin)

        images = self.soup.select(".image-large")

        if not images:
            return None

        for index, div_tag in enumerate(images):

            # Check if the no. of images exceeds 9.
            if index == 9:
                image_meta["Exceeded 9 images"] = True

                break

            img_tag = div_tag.img

            if not img_tag:
                continue

            img_src = f"https:{img_tag['src']}"

            meta = self._get_image_metadata(asin=asin, index=index)
            img_name, image_url_col_name = meta["img_name"], meta["image_url_col_name"]

            # Add the url to the dict
            image_meta[image_url_col_name] = img_src
            if image_url_col_name == "main":
                image_meta["Is Main Image Background White"] = self._is_main_img_bg_color_white(url=img_src)

            self._image_download_and_save(url=img_src, img_name=img_name, folderize=folderize)

        return image_meta


class RunType(enum.Enum):
    fetch_data = "fetch_data"
    fetch_images = "fetch_images"


def start(df: pd.DataFrame, mode: str, progress_bar: bool = False):
    s = SevenScraper()

    data = df.to_dict("records")

    # Check if image download can even be carried out or not
    if mode == RunType.fetch_images.value and not any(col in df.columns for col in {"ASIN", "Seller SKU"}):
        raise ValueError("ASIN name required for image download to start.")

    try:
        if progress_bar:
            import streamlit as st

            status_bar = st.progress(0)
            step = 100 / len(data)

        for index, row in enumerate(data):
            if progress_bar:
                status_bar.progress(int((index + 1) * step))

            resp = s.make_soup_obj(row["URL"])

            if resp is None:
                continue

            if mode == RunType.fetch_data.value:
                row["Title"] = s.get_title()
                print(index, row["Title"])

                row["Price"] = s.get_price()
                # TODO: Max Bullets logic is remaining.
                if "Description" in row:
                    row.update({"Description": s.get_description()})
                else:
                    row["Description"] = s.get_description()

                for i, b in enumerate(s.get_bullets(), start=1):
                    row[f"Bullet{i}"] = b

            if mode == RunType.fetch_images.value:
                image_meta = s.get_images(row=row)

                if not image_meta:
                    continue

                [row.update({col_name: col_val}) for col_name, col_val in image_meta.items()]

    finally:
        out = pd.DataFrame(data)

        # Rearrange the columns to have all image name cols together
        if "Is Main Image Background White" in out.columns:
            out.insert(
                len(out.columns) - 1, "Is Main Image Background White", out.pop("Is Main Image Background White")
            )
        if "Exceeded 9 images" in out.columns:
            out.insert(len(out.columns) - 1, "Exceeded 9 images", out.pop("Exceeded 9 images"))

        fname = "Seven.csv"
        if mode == RunType.fetch_data.value:
            fname = "Seven.csv".replace(".csv", "__data.csv")
        elif mode == RunType.fetch_images.value:
            fname = "Seven.csv".replace(".csv", "__images.csv")

        out.to_csv(fname, sep=",", index=False)

        return out


if __name__ == "__main__":
    df = pd.read_csv("./seven/inputs/test.csv")

    start(df, "fetch_data")
