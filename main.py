import openai_utils

system_prompt = "You are a professional art critic specialized in art images from the Global South. When given an image, apply your formal art analysis expertise to analyze images, and then compose a write-up your analysis in 5-7 sentences.  Output this write-up only."
user_prompt = "Please analyze this image."
artwork = r"C:\Users\at1e18\OneDrive - University of Southampton\Documents\Lesia\2025_files\programming projects\global_south_AIGC\Dataset\Wai_Ming\Wai_Ming_11.jpg"

openai_utils.call_and_write_to_log_process(system_prompt=system_prompt, user_prompt=user_prompt, artwork=artwork,
                   notes="TEST", log_file="log_for_testing.jsonl", api_key=openai_utils.openai_api_key)