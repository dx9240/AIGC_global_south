import spacy
from sympy.codegen import Print
import faiss # this library might run into problems running on windows - try linux or mac instead if you have issues
from openai import OpenAI
import json
import requests
from openai_utils import openai_api_key

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


# 3. Take a user prompt → embed → search
# 4. Return top-k most similar captions

# # check out the vector store
# file_list = client.vector_stores.files.list(vector_store_id=synthetic_store_id)
# print(file_list)

# search vector store
client = OpenAI(api_key=openai_api_key)
config_file = "config.json"

# function to get the vector store ID from the config file
def get_vector_store_id(config_path):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("vector_store_id")
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found. It's possible that the vector store has been automatically deleted")
        print("run the 'setup_vector_store.py' file to set up a vector store and store the vector store ID in the config file")
        return None

# search for the most similar matches to the user prompt in the vector store. the API returns the top 10 most similar matches.
user_prompt = "In a regal and harmonious mood, make a detailed gongbi painting of the Forbidden City at sunrise, with golden rooftops glowing against vermilion walls. There should be famous animals in the Ming Dynasty hanfu strolling through courtyards adorned with peony blossoms. The artwork should include intricate lattice windows and cranes symbolising longevity, and the palette should be of vermilion, jade green, and gold."
# Load the ID from our config file.
vector_store_id = get_vector_store_id(config_file)
#search the vector store to match the user prompt
search_results = client.vector_stores.search(
  vector_store_id=vector_store_id,
  query=user_prompt
)

# iterate through and print search results
for result in search_results:
    print("--- Result ---")
    print(f"Similarity Score: {result.score:.4f}")

    # The .content attribute is a LIST of chunks. We need to loop through it.
    print("Content:")
    for chunk in result.content:
        # Now 'chunk' is the object with the .text attribute
        print(chunk.text)

    print()  # Add a newline for cleaner separation between results