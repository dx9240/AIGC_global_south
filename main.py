from dotenv import load_dotenv
import os
from google import genai
from openai import OpenAI
import base64

load_dotenv()
openAI_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

initial_system_prompt = "You are a professional art critic specialized in art images from the Global South. When given an image, apply your formal art analysis expertise to analyze images, and then compose a write-up your analysis in 5-7 sentences.  Output this write-up only."
artwork = r"C:\Users\at1e18\OneDrive - University of Southampton\Documents\Lesia\2025_files\programming projects\global_south_AIGC\Dataset\Wai_Ming\Wai_Ming_04.jpg"

#Use Gemini
# client = genai.Client(api_key=gemini_api_key)
#
# my_file = client.files.upload(file=artwork)
#
# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=[initial_system_prompt,
#               my_file,
#               "add an emoji at the end of your write-up."],
# )
#
# print(response.text)

#Use GPT
#open image
with open(artwork, "rb") as img_file:
    image_bytes = img_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

#send info and get response
client = OpenAI(api_key=openAI_api_key)
response = client.responses.create(
    model="gpt-4o",
    input=[
        {"role": "system", "content": initial_system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"},
                {"type": "input_text", "text": "Please analyze this image."}
            ]
        }
    ],
)

# Output result
print(response.output_text)
