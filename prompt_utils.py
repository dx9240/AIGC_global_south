import spacy
from sympy.codegen import Print
import faiss # this library might run into problems running on windows - try linux or mac instead if you have issues
from openai import OpenAI
from openai_utils import openai_api_key
import json
import requests

text_1 = "A mature woman wearing a blue silk sari standing against a background of moonlit Jaipur."
text_2 = "An ancient, blue maple tree clinging to a cliff, overlooking a purple lake. There is blue moss growing on the cliff, two crescent moons and jupiter in the sky, and a spaceship flying across the sky in the distance. The painting is in a traditional Chinese style, with futuristic caligraphy."

synthetic_1 = "This painting embodies the traditional Chinese landscape style, known as \"shan shui,\" which translates to \"mountain-water.\" The use of vibrant blues and greens creates a serene and harmonious atmosphere, emphasizing the natural beauty of the scene. The waterfall acts as a focal point, drawing the viewer's eye through the composition and symbolizing the flow of life. The subtle inclusion of human figures at the bottom of the painting adds a sense of scale and highlights humanity's smallness in the face of nature's grandeur. The misty mountains in the background suggest depth and evoke a sense of mystery and tranquility. Overall, the artwork beautifully captures the essence of nature's majesty and the traditional aesthetic values of Chinese landscape painting."
synthetic_2 = "This artwork presents a striking interplay between the realistic depiction of the human form and abstract elements. The central figure, a woman, is portrayed with a dramatic sense of movement and emotion, her body draped in a white garment accentuated by vivid red fabric. The red not only serves as a visual anchor but also adds a sense of urgency and intensity to the composition. The reaching hand, seemingly emerging from the background, introduces a narrative tension, suggesting themes of escape or entanglement."

# nlp = spacy.load("en_core_web_trf")
# doc = nlp(text_1)
#
# # Noun chunks - lemmatized head nouns
# noun_chunks = set(chunk.lemma_.lower() for chunk in doc.noun_chunks)
#
# def extract_meaningful_nouns(text):
#     doc = nlp(text)
#
#     # Use POS tagging
#     nouns = {token.lemma_.lower() for token in doc
#              if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop}
#     return nouns
#
# # Named entities - surface form, lowercase
# # list of entity lables can be found here: https://github.com/explosion/spaCy/discussions/9147
# entity_labels = {"GPE", "PERSON", "ORG", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE"}
# named_entities = set(
#     ent.text.lower() for ent in doc.ents if ent.label_ in entity_labels
# )
#
# print("NOUN CHUNKS")
# print(noun_chunks)
#
# print("SPECIFIC NOUNS")
# print(extract_meaningful_nouns(text_1))
#
# print("ENTITIES")
# print(named_entities)

# Implement OpenAI API vector store to search and return top k matches of synthetic captions to an input user prompt

# 1. get the synthetic captions out of the jsonl log file
# 2. generate vectors for the synthetic captions (or whatever it is you do the get them into the vector store)
# 3. input the user prompt and search for top k matches in the vectorstore

#Run the to lines below to test if the FAISS library is working on Windows
#index = faiss.IndexFlatL2(1536)
#print(index.is_trained)

# 1. Generate vector embeddings
# 2. Store them in OpenAI vector store
# 3. Take a user prompt → embed → search
# 4. Return top-k most similar captions

# extract the synthetic captions from the json file where they are logged and store in a list
synthetic_log = "20250604_openai_log.jsonl"
vector_store_upload = "vector_store_upload.json"
with open(synthetic_log, "r", encoding="utf-8") as infile, open(vector_store_upload, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        if "output_text" in data:
            caption = data["output_text"]
        json.dump({
            "text": caption,
            "metadata": {"text": caption}  # metadata mirrors the text
        }, outfile)
        outfile.write("\n")

# set up vector store
client = OpenAI(api_key=openai_api_key)
# variable 'response' contains the vector store's ID. Access the ID with vector_store_id
synthetic_store_response = client.vector_stores.create(
    name="synthetic_captions_vector_store",
    #optional line of code - vector store is deleted from openAI after 30 days
    expires_after={
    "anchor": "last_active_at",
        "days": 1}
    )
synthetic_store_id = synthetic_store_response.id

# consider expanding metadata tags or scope for a formal art and global south art domain-specific ontology to access more specific terms, in order to generate better images
with open(vector_store_upload, "rb") as f:
    file_upload_response = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=synthetic_store_id,
        files=[f]
    )

# # check out the vector store
# file_list = client.vector_stores.files.list(vector_store_id=synthetic_store_id)
# print(file_list)

# search vector store
query = "a woman sitting"

headers = {
    "Authorization": f"Bearer {openai_api_key}",
    "Content-Type": "application/json"
}

response = requests.post(
    f"https://api.openai.com/v1/vector_stores/{synthetic_store_id}/search",
    headers=headers,
    json={"query": query}
)


results = response.json()
print(results)