# 1_setup_vector_store.py

import json
import time
import io
from openai import OpenAI
from openai_utils import openai_api_key

client = OpenAI(api_key=openai_api_key)
# Once the vector store is set up, the ID is saved in this config file so that it can be referred to later without hardcoding in the code
config_file = "config.json"
# extract the synthetic captions from the json file where they are logged
synthetic_log = "20250604_openai_log.jsonl"
# This is to avoid hitting API rate limits. change to a higher number if needed.
delay_between_uploads = 1

# create the vector store
print("Creating a new vector store")
vector_store = client.vector_stores.create(
    name="synthetic captions store",
    # Set the number of days the vector store will exist for, before it is deleted. See OpenAI API docs for more options
    expires_after={"anchor": "last_active_at", "days": 30}
)
vector_store_id = vector_store.id
print(f"vector store ID: {vector_store_id}")

# record vector store ID in a config file, so that it can be accessed later
print(f"\nsaving vector store ID to '{config_file}'...")
with open(config_file, "w") as f:
    json.dump({"vector_store_id": vector_store_id}, f)
print("Done.")

# Now that the vector store has been created, upload each synthetic caption as a single 'document' so that OpenAI can
# process each document as a chunk. Uploading as single JSONL, for example, and OpenAI API will not chunk each caption
# as whole. Note that very long synthetic captions (such as approx. 600 words) might be treated as more than one chunk
# by OpenAI API. To solve this if needed, increase the max. chunk size of tokens when setting up the vector store.
print(f"\nUploading synthetic captions from '{synthetic_log}' one by one:")
captions_uploaded = 0
with open(synthetic_log, "r", encoding="utf-8") as infile:
    for line in infile:
        try:
            data = json.loads(line)
            # Check for the caption text, and response ID to be used as document name
            if "output_text" in data and "response_id" in data:
                caption_text = data["output_text"]
                unique_id = data["response_id"]

                # Create an in-memory file for the caption and upload it to Op enAI API
                file_stream = io.BytesIO(caption_text.encode("utf-8        "))
                file = client.files.create(
                    file=(f"{unique_id}.txt", file_stream),
                    purpose="assistants"
                )

                # Add the newly uploaded file to the vector store
                vector_store_file = client.vector_stores.files.create(
                    vector_store_id=vector_store_id,
                    # file_id is the name of the file, which is also the synthetic caption's response_id
                    file_id=file.id
                )

                print(f"  Uploaded file: {vector_store_file.id} (Source: {unique_id})")
                captions_uploaded += 1

                # Wait before the next upload to avoid API rate limits
                time.sleep(delay_between_uploads)

        except json.JSONDecodeError:
            print(f"  Skipping malformed line: {line.strip()}")
        except Exception as e:
            print(f"  An error occurred during upload: {e}")

print(f"\n{captions_uploaded} captions uploaded successfully.")

