from pathlib import Path
from typing import Iterable
from PIL import Image, UnidentifiedImageError

# TODO deal with exceptions
# TODO connect to the APIs

# This is code for iterating through a dataset folder and getting the paths to image files.
# The image paths will be batch (or whatever) fed into the API. For example, all the image paths from a dataset will be
# collected, paired with prompts, and passed to the OpenAI API.

image_extensions = {".jpg", ".jpeg", ".png"}


def iter_image_paths(dataset_root: str | Path) -> Iterable[Path]:
    """
    Function to iterate through dataset folder and subfolders and get the paths to image files.
    Works for both strings and paths.

    :param dataset_root: Path to the top-level dataset folder. Accepts either a raw
        string or a ``pathlib.Path`` object.

    :return: Iterable[pathlib.Path]
        A **lazy** iterable (generator) that produces ``Path`` objects
        pointing to image files whose extensions are in ``image_extensions``.
        Convert to a list if you need all the paths at once.
    """
    root = Path(dataset_root)
    for p in root.rglob("*"):
        if p.suffix.lower() in image_extensions:
            yield p


def is_valid_image(path: Path) -> bool:
    """
    Function to check if a path is a valid image file.

    """
    try:
        with Image.open(path) as img:
            # cheap header check; doesn’t decode full image
            img.verify()
        # no exception, therefore looks valid
        return True
    except (UnidentifiedImageError, OSError):
        return False

# test an image to check if it's valid
bad = [p for p in iter_image_paths("Dataset") if not is_valid_image(p)]
print("Corrupt images:", bad)
