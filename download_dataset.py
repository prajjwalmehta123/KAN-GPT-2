import os
from typing import List

import requests
from tqdm import tqdm

CHUNK_SIZE = 8192

def download_tinyshakespeare(
    download_path: str = "datasets/tinyshakespeare",
    base_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare",  # noqa
):
    os.makedirs(download_path, exist_ok=True)

    response = requests.get(f"{base_url}/input.txt", stream=True)
    response.raise_for_status()  # Raise HTTP errors
    total_size = int(response.headers.get("content-length", 0))

    # Open a local file for writing in binary mode
    with open(f"{download_path}/input.txt", "wb") as file, tqdm(
        desc=f"Downloading {download_path}/input.txt",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            size = file.write(chunk)
            progress_bar.update(size)


def main():
    download_tinyshakespeare()

if __name__ == "__main__":
    main()
