from openai import OpenAI
import base64
import json
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


def request_openai(*, api_key=openai_api_key, system_prompt, user_prompt, artwork,
                   system_prompt_version="N/A", user_prompt_version="N/A", llm_model="gpt-4o", notes=""):
    """
    Function to request an OpenAI API response by inputting a system prompt, user prompt, and path to an image.
    The purpose is to send an image and prompts to the API, and to get back a text description of the image.

    :arg: artwork must be a link to an image
    :return: input sent to the API and selected info from the API response.
    """
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


def log_openai_response(*, response, system_prompt, user_prompt, artwork,
                        system_prompt_version="N/A", user_prompt_version="N/A", notes=""):
    """Function to log the input and info sent to the openAI API, and log selected info from the OpenAI API response.
        The purpose is to collect and structure the info so that it can easily be written to file by another function.

        :return: A dict with all the info sent, and the received response output text and selected metadata.
    """
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


def print_log_data_to_file(log_data, log_file="openai_log_file.jsonl"):
    """Function to get the API input and response info from another function and to print it to a log file as a .jsonl
        If the named file doesn't already exist it will create it, otherwise it will append to the existing file. Inside
        the log file you can find info such as the text that was generated based on the input prompt and image.
    """
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")


def call_and_write_to_log_process(*, system_prompt, user_prompt, artwork,
                                  system_prompt_version="N/A", user_prompt_version="N/A", llm_model="gpt-4o", notes="",
                                  log_file="openai_log_file.jsonl", api_key=openai_api_key):
    """Function to connect other functions into one larger action: input prompting configuration info and path to image
        file into the function. This then gets sent to the OpenAI API. You get a line in a .jsonl log file back. In this
        file you will find the info such as the text that was generated based on the input prompt, and the file path to
        the image."""
    log_data = request_openai(api_key=api_key, system_prompt=system_prompt, user_prompt=user_prompt, artwork=artwork,
                              system_prompt_version=system_prompt_version, user_prompt_version=user_prompt_version,
                              llm_model=llm_model, notes=notes)
    print_log_data_to_file(log_data, log_file)

# TODO cosine similarity...an idea for later?
