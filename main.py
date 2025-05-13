from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
openAI_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

initial_system_prompt = "You are a professional art critic specialized in art images from the Global South. When given an image, apply your formal art analysis expertise to analyze images, and then compose a write-up your analysis in 5-7 sentences.  Output this write-up only."
artwork = r"C:\Users\at1e18\OneDrive - University of Southampton\Documents\Lesia\2025_files\programming projects\global_south_AIGC\Dataset\Wai_Ming\Wai_Ming_04.jpg"

client = genai.Client(api_key=gemini_api_key)

my_file = client.files.upload(file=artwork)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[initial_system_prompt,
              my_file,
              "add an emoji at the end of your write-up."],
)

print(response.text)
