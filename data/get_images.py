from __future__ import annotations
import os
import time
import json
import re
import logging
from typing import Optional
from urllib.parse import quote, urlparse
import pandas as pd
import requests
from bs4 import BeautifulSoup

#!/usr/bin/env python3
"""
get_images.py

Read texas_plant_list_final.csv from the same folder as this script, build a search
query from the 'Scientific Name' and 'Common Name' columns for each row, scrape the
first image result from Bing Images, and save it to an 'images' folder using the
row's uid as filename.

Usage: run from the directory containing texas_plant_list_final.csv, or run this file directly.
"""


# basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_NAME = "texas_plant_list_final.csv"
CSV_PATH = os.path.join(BASE_DIR, CSV_NAME)
IMAGES_DIR = os.path.join(BASE_DIR, "images")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ImageScraper/1.0; +https://example.com/bot)"
}
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN = 0.3  # seconds between requests


def safe_filename(name: str) -> str:
    # keep only safe chars
    return re.sub(r"[^\w\-_\. ]", "_", str(name))


def pick_extension_from_url(url: str, content_type: Optional[str]) -> str:
    # try to get extension from URL path
    parsed = urlparse(url)
    _, ext = os.path.splitext(parsed.path)
    if ext and len(ext) <= 5:
        return ext
    # fallback from content type
    if content_type:
        if "jpeg" in content_type:
            return ".jpg"
        if "png" in content_type:
            return ".png"
        if "gif" in content_type:
            return ".gif"
        if "webp" in content_type:
            return ".webp"
    return ".jpg"


def find_first_image_url_from_bing(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    # Many Bing image anchors include a JSON blob in attribute 'm' with "murl" key.
    for a in soup.find_all("a", class_="iusc"):
        m = a.get("m")
        if not m:
            continue
        try:
            data = json.loads(m)
            murl = data.get("murl")
            if murl:
                return murl
        except Exception:
            continue
    # Fallback: first <img> tag with http src
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src and src.startswith("http"):
            return src
    return None


def download_image(url: str, target_path: str) -> bool:
    try:
        resp = requests.get(url, headers=HEADERS, stream=True, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logging.debug(f"Non-200 response for {url}: {resp.status_code}")
            return False
        content_type = resp.headers.get("Content-Type", "")
        ext = pick_extension_from_url(url, content_type)
        if not target_path.lower().endswith(ext):
            target_path = target_path + ext
        tmp_path = target_path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(1024 * 8):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, target_path)
        logging.info(f"Saved image to {target_path}")
        return True
    except Exception as e:
        logging.debug(f"Error downloading image {url}: {e}")
        return False


def main():
    if not os.path.exists(CSV_PATH):
        logging.error(f"CSV not found at {CSV_PATH}. Place texas_plant_list_final.csv next to this script.")
        return

    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    # Expect 'uid' column and names columns. Adjust if column names differ.
    if "uid" not in df.columns:
        # try common alternatives
        if "_uid" in df.columns:
            df = df.rename(columns={"_uid": "uid"})
        else:
            logging.error("CSV does not contain 'uid' column.")
            return

    # Accept several possible name column names
    sci_col = None
    com_col = None
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("scientific name", "scientific_name", "scientific"):
            sci_col = c
        if lc in ("common name", "common_name", "common"):
            com_col = c
    if sci_col is None:
        logging.error("Could not find a 'Scientific Name' column in CSV.")
        return
    if com_col is None:
        logging.warning("Could not find a 'Common Name' column; proceeding with scientific name only.")

    os.makedirs(IMAGES_DIR, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    for idx, row in df.iterrows():
        uid = row.get("uid", "")
        if not uid:
            logging.warning(f"Row {idx} missing uid — skipping")
            continue
        uid_safe = safe_filename(uid)
        # create base target path without extension
        target_base = os.path.join(IMAGES_DIR, uid_safe)

        # If file already exists with common extensions, skip
        exists = False
        for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            if os.path.exists(target_base + ext):
                logging.info(f"Image for uid {uid} already exists, skipping.")
                exists = True
                break
        if exists:
            continue

        sci = str(row.get(sci_col, "")).strip()
        com = str(row.get(com_col, "")).strip() if com_col else ""
        query = " ".join([p for p in (sci, com) if p]).strip()
        if not query:
            logging.warning(f"No name available for uid {uid}, skipping.")
            continue

        # Append some context to improve chance of a plant photo
        query_for_search = quote(query + " plant texas")

        search_url = f"https://www.bing.com/images/search?q={query_for_search}&qft=+filterui:imagesize-large&form=IRFLTR"
        logging.info(f"[{idx}] Searching image for uid={uid} -> {query}")

        try:
            resp = session.get(search_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logging.warning(f"Search failed for uid {uid}: status {resp.status_code}")
                time.sleep(SLEEP_BETWEEN)
                continue
            img_url = find_first_image_url_from_bing(resp.text)
            if not img_url:
                logging.warning(f"No image URL found for uid {uid} ({query})")
                time.sleep(SLEEP_BETWEEN)
                continue
            # download
            success = download_image(img_url, target_base)
            if not success:
                logging.warning(f"Failed to download image for uid {uid} from {img_url}")
        except Exception as e:
            logging.warning(f"Error processing uid {uid}: {e}")
        time.sleep(SLEEP_BETWEEN)

    logging.info("Done.")


if __name__ == "__main__":
    main()