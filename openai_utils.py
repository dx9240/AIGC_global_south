from openai import OpenAI
import base64
import json
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# function to log LLM output
# TODO cosine similarity...an idea for later?
def log_openai_response(*, response, system_prompt, user_prompt, artwork,
                        system_prompt_version="N/A", user_prompt_version="N/A", notes=""):
    log_entry = {
        "response_id": response.id,
        "model": response.model,
        "system_prompt_version": system_prompt_version,
        "system_prompt": system_prompt,
        "user_prompt_version": user_prompt_version,
        "user_prompt": user_prompt,
        "output_text": response.output[0].content[0].text,
        "temperature": response.temperature,
        "top_p": response.top_p,
        "image_path": artwork,
        "notes": notes,
        "timestamp": response.created_at,
        "max_tokens": response.max_output_tokens,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return log_entry


def request_openai(*, api_key=openai_api_key, system_prompt, user_prompt, artwork,
                   system_prompt_version="N/A", user_prompt_version="N/A", llm_model="gpt-4o", notes=""):
    # open image
    with open(artwork, "rb") as img_file:
        image_bytes = img_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # send info and get response
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=llm_model,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"},
                    {"type": "input_text", "text": user_prompt}
                ]
            }
        ],
    )
    # API response and other info needed to be logged
    log_data = log_openai_response(
        response=response,
        system_prompt=system_prompt,
        system_prompt_version=system_prompt_version,
        user_prompt=user_prompt,
        user_prompt_version=user_prompt_version,
        artwork=artwork,
        notes=notes)
    return log_data


# get the openAI response API log data and print it to file
def print_log_data_to_file(log_data, log_file="openai_log_file.jsonl"):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")

def call_and_write_to_log_process(*, system_prompt, user_prompt, artwork,
                   system_prompt_version="N/A", user_prompt_version="N/A", llm_model="gpt-4o", notes="", log_file="openai_log_file.jsonl", api_key=openai_api_key):
    log_data = request_openai(api_key=api_key, system_prompt=system_prompt, user_prompt=user_prompt, artwork=artwork,
                   system_prompt_version=system_prompt_version, user_prompt_version=user_prompt_version, llm_model=llm_model, notes=notes)
    print_log_data_to_file(log_data, log_file)