import openai_utils
import pipeline_utils
from openai_utils import request_openai
from pathlib import Path
from tqdm import tqdm


# prompts and image to be sent to the API
system_prompt = "You are a professional art critic specialized in art images from the Global South. When given an image, apply your formal art analysis expertise to analyze images, and then compose a write-up your analysis in 5-7 sentences.  Output this write-up only."
user_prompt = "Please analyze this image."
artwork = r"C:\Users\at1e18\OneDrive - University of Southampton\Documents\Lesia\2025_files\programming projects\global_south_AIGC\Dataset\Wai_Ming\Wai_Ming_11.jpg"
dataset_path = r"C:\Users\at1e18\OneDrive - University of Southampton\Documents\Lesia\2025_files\programming projects\global_south_AIGC\Dataset"
test_path = r"C:\Users\at1e18\OneDrive - University of Southampton\Documents\Lesia\2025_files\programming projects\global_south_AIGC\Dataset\Wai_Ming"
log_file = "20240603_openai_log.jsonl"
# call to openAI: send image and system prompt, user prompt. OpenAI generates a description for the image and sends back
# metadata. All the sent and received info is logged in the log_file as .jsonl
# openai_utils.call_and_write_to_log_process(system_prompt=system_prompt, user_prompt=user_prompt, artwork=artwork,
#                                            notes="TEST", log_file="log_for_testing.jsonl",
#                                            api_key=openai_utils.openai_api_key)


# run_batch_plain.py
#
# 1. Walk through a dataset folder
# 2. For every image, send it (plus prompts) to OpenAI
# 3. Append the response to a JSONL log file

# 0) -----  Resume logic: figure out which images are already done  -----
def images_already_done(log_path: Path) -> set[str]:
    """
    Read the existing JSONL log file (if any) and collect every
    image_path already processed. Returns a *set* of strings.
    """
    done_paths = set()

    if log_path.exists():
        with log_path.open() as f:
            for line in f:
                # string search to pull out the value of "image_path"
                # Assumes each line contains something like ..."image_path": "C:\\...\some.jpg"...
                parts = line.split('"image_path": "')
                if len(parts) >= 2:
                    path_part = parts[1].split('"')[0]
                    done_paths.add(path_part)

    return done_paths

# collect a set of the paths to image which have already been processed
done_paths_set = images_already_done(Path(log_file))
# collect a list of all image paths that still need processing
images_to_process = []

for image_path in pipeline_utils.iter_image_paths(dataset_path):
    if str(image_path) not in done_paths_set:
        images_to_process.append(image_path)

# keep track of how many images have been processed
total_images = len(images_to_process)
print(f"{total_images} images need processing.")


# loop through images, send to API, write log
progress_bar = tqdm(total=total_images, desc="Uploading images")

for image_path in images_to_process:
    # use my function to call to OpenAI API one-by-one
    log_entry = request_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        artwork=image_path,
        notes="Default OpenAI settings. Initial batch to test if uploading to API and processing works.")
    # Append the resulting dict (plus metadata) to the JSONL file
    openai_utils.print_log_data_to_file(log_entry, log_file)
    # add to the progress bar by one
    progress_bar.update(1)

progress_bar.close()
print("Batch complete.")