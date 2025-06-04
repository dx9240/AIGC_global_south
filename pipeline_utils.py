from pathlib import Path
from typing import Iterable
import mimetypes, base64


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

# test an image to check if it's valid
#bad = [p for p in iter_image_paths("Dataset") if not is_valid_image(p)]
#print("Corrupt images:", bad)

def encode_image_as_data_uri(path: Path) -> str:
    """
    Function to encode an image file into a data URI.
    Some of the images in my test dataset have an issue being sent to OpenAI API, and this function is made to solve
    that issue by labeling each file with its MIME type.

    :param path: Path to the image file.

    """
    path = Path(path)
    mime, _ = mimetypes.guess_type(path)
    # sensible fallback
    mime = mime or "image/jpeg"
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

