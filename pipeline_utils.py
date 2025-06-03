from pathlib import Path

# TODO deal with exceptions
# TODO connect to the APIs

# This is code for iterating through a dataset folder and getting the paths to image files.
# The image paths will be batch (or whatever) fed into the API. For example, all the image paths from a dataset will be
# collected, paired with prompts, and passed to the OpenAI API.
dataset_path = r"C:\Users\at1e18\OneDrive - University of Southampton\Documents\Lesia\2025_files\programming projects\global_south_AIGC\Dataset"

image_extensions = [".jpg", ".jpeg", ".png"]
# list of all the image paths
image_paths = []

# get the path to each image in the dataset anf go through all subfolders
dataset_folder = Path(dataset_path)
all_paths = dataset_folder.rglob("*")
for p in all_paths:
    file_extension = p.suffix.lower()
    # only keep the image files
    if file_extension in image_extensions:
        image_paths.append(p)

# print the paths and the total number of paths collected
image_counter = 0
for i in image_paths:
    image_counter += 1
    print(i)
print(image_counter)
