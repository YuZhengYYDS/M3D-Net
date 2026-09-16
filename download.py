"""Download upstream weights or the BrEaST mirror into ignored local storage."""
import argparse
from pathlib import Path

import requests

from m3d.pretrained import SOURCES, sha256_file

BREAST_URL = 'https://www.kaggle.com/api/v1/datasets/download/rabeasaleh2/breast-lesions-usg'
BREAST_SHA256 = '5ae656af4b77b2e5f2220525f513a24b206181a23d2e41970782bb1e063772fb'


def download(url, destination, expected_sha256):
    destination = Path(destination)
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f'Existing file failed SHA256 verification: {destination}')
        print(f'Verified existing file: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    headers = {'Accept': 'application/octet-stream', 'User-Agent': 'M3D-Net-reproduction'}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open('wb') as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
    if sha256_file(temporary) != expected_sha256:
        raise ValueError(f'Download failed SHA256 verification: {destination.name}; upstream may have changed')
    temporary.replace(destination)
    print(f'Downloaded and verified: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['weights', 'breast'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.kind == 'breast':
        download(BREAST_URL, args.output or Path('data/downloads/breast_lesions_usg.zip'), BREAST_SHA256)
    else:
        directory = args.output or Path('data/pretrained')
        for source in SOURCES.values():
            download(source['url'], directory / source['filename'], source['sha256'])
