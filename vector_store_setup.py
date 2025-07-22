# 1_setup_vector_store.py

import os
import json
from openai import OpenAI
from openai_utils import openai_api_key

# extract the synthetic captions from the json file where they are logged and store in a list
synthetic_log = "20250604_openai_log.jsonl"
vector_store_upload = "vector_store_upload.json"
with open(synthetic_log, "r", encoding="utf-8") as infile, open(vector_store_upload, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            data = json.loads(line)
            if "output_text" in data:
                caption = data["output_text"]
                json.dump({"text": caption}, outfile)
            outfile.write("\n")
        except json.JSONDecodeError:
            print(f"Skipping malformed line: {line.strip()}")

# the code below set up the vector store and adds the synthetic captions to it

client = OpenAI(api_key=openai_api_key)

# Once the vector store is set up, the ID is saved in this config file so that it can be referred to later without hardcoding in the code
config_file = "config.json"

# create the Vector Store
print("Creating a new vector store")
vector_store = client.vector_stores.create(
    name="synthetic captions store",
    # Set the number of days the vector store will exist for, before it is deleted. See OpenAI API docs for more options
    expires_after={"anchor": "last_active_at", "days": 30}
)
vector_store_id = vector_store.id
print(f"Vector Store created with ID: {vector_store_id}")

# Upload and Poll the file to make sure everything has been processed before performing other actions
print(f"uploading file '{vector_store_upload}' to the vector store")
with open(vector_store_upload, "rb") as f:
    file_batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store_id,
        files=[f]
    )

print("file upload complete and processed")
print(f"Status: {file_batch.status}")
print(f"File counts: {file_batch.file_counts}")

# save the ID for later use in the search task
# save vector_store_id to a JSON config file.
print(f"saving Vector Store ID to '{config_file}'...")
with open(config_file, "w") as f:
    json.dump({"vector_store_id": vector_store_id}, f)
print("done")