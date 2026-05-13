"""Run once to download 6 LFW images into this directory. Not part of the app."""
import urllib.request
from pathlib import Path

IMAGES = [
    ("alice_1", "https://vis-www.cs.umass.edu/lfw/images/George_W_Bush/George_W_Bush_0001.jpg"),
    ("alice_2", "https://vis-www.cs.umass.edu/lfw/images/George_W_Bush/George_W_Bush_0002.jpg"),
    ("bob_1",   "https://vis-www.cs.umass.edu/lfw/images/Colin_Powell/Colin_Powell_0001.jpg"),
    ("bob_2",   "https://vis-www.cs.umass.edu/lfw/images/Colin_Powell/Colin_Powell_0002.jpg"),
    ("carol_1", "https://vis-www.cs.umass.edu/lfw/images/Tony_Blair/Tony_Blair_0001.jpg"),
    ("carol_2", "https://vis-www.cs.umass.edu/lfw/images/Tony_Blair/Tony_Blair_0002.jpg"),
]

out = Path(__file__).parent
for name, url in IMAGES:
    dest = out / f"{name}.jpg"
    if not dest.exists():
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  saved → {dest}")
    else:
        print(f"  {name}.jpg already exists, skipping.")
print("Done.")
