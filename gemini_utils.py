# from google import genai

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Use Gemini
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