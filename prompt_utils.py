import spacy
from openai import OpenAI
import json
from openai_utils import openai_api_key

# some testing prompts
shepuya_prompt = "In a regal and harmonious mood, make a detailed gongbi painting of the Forbidden City at sunrise, with golden rooftops glowing against vermilion walls. There should be famous animals in the Ming Dynasty hanfu strolling through courtyards adorned with peony blossoms. The artwork should include intricate lattice windows and cranes symbolising longevity, and the palette should be of vermilion, jade green, and gold."
text_1 = "A mature woman wearing a blue silk sari standing against a background of moonlit Jaipur."
text_2 = "An ancient, blue maple tree clinging to a cliff, overlooking a purple lake. There is blue moss growing on the cliff, two crescent moons and Jupiter in the sky, and a spaceship flying across the sky in the distance. The painting is in a traditional Chinese style, with futuristic caligraphy."

synthetic_1 = "This painting embodies the traditional Chinese landscape style, known as \"shan shui,\" which translates to \"mountain-water.\" The use of vibrant blues and greens creates a serene and harmonious atmosphere, emphasizing the natural beauty of the scene. The waterfall acts as a focal point, drawing the viewer's eye through the composition and symbolizing the flow of life. The subtle inclusion of human figures at the bottom of the painting adds a sense of scale and highlights humanity's smallness in the face of nature's grandeur. The misty mountains in the background suggest depth and evoke a sense of mystery and tranquility. Overall, the artwork beautifully captures the essence of nature's majesty and the traditional aesthetic values of Chinese landscape painting."
synthetic_2 = "This artwork presents a striking interplay between the realistic depiction of the human form and abstract elements. The central figure, a woman, is portrayed with a dramatic sense of movement and emotion, her body draped in a white garment accentuated by vivid red fabric. The red not only serves as a visual anchor but also adds a sense of urgency and intensity to the composition. The reaching hand, seemingly emerging from the background, introduces a narrative tension, suggesting themes of escape or entanglement."

nlp = spacy.load("en_core_web_trf")

def nlp_processing(text):
    """
    Processes a single text string to extract head nouns and named entities.
    Takes a string to process as input.
    Returns dict of extracted results.
    """
    # Create the spaCy doc object once
    doc = nlp(text)

    # extract noun chunks (lemmatized head nouns)
    noun_chunks = {chunk.lemma_.lower() for chunk in doc.noun_chunks}

    # extract meaningful nouns (non-stopwords)
    nouns = {token.lemma_.lower() for token in doc
             if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop}

    # extract named entities
    # list of entity labels can be found here: https://github.com/explosion/spaCy/discussions/9147
    entity_labels = {"GPE", "PERSON", "ORG", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE"}
    named_entities = {ent.text.lower() for ent in doc.ents if ent.label_ in entity_labels}

    return {
        "noun_chunks": noun_chunks,
        "nouns": nouns,
        "entities": named_entities
    }


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


def vector_search(store_id, query):
    """
    Function to search a vector store with a given query.
    Takes OpenAI API vector store ID, and a string query as arguments.
    Returns the OpenAI API response which contains the top 10 vector search results based on normalized cosine similarity.

    """
    #search the vector store to match the user prompt
    search_results = client.vector_stores.search(
      vector_store_id=store_id,
      query=query
    )
    return search_results


def vector_search_results_printer(query, search_response):
    """
    Function to print the results of a vector search.
    Takes the search query as input so that it can be shown alongside the search results.
    Also takes OpenAI API vector search response as input and prints the top 10 vector search results based on normalized cosine similarity.

    """
    # iterate through the returned response and print search results
    print(f"User Prompt: {query} \n")
    for result in search_response:
        print("--- Vector Search Results ---")
        print(f"Similarity Score: {result.score:.4f}")
        # The .content attribute is a LIST of chunks. We need to loop through it.
        print("Content:")
        for chunk in result.content:
            # Now 'chunk' is the object with the .text attribute
            print(chunk.text)
        print()  # Add a newline for cleaner separation between results


def calculate_jaccard_similarity(set_a, set_b):
    """
    Calculates the Jaccard Similarity between two sets. Use this to compare keywords between two sets.
    """
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(intersection) / len(union)


def get_comparison_scores(user_prompt, search_response):
    """
    Compares a user prompt to each search result to get both cosine
    and Jaccard similarity scores for nouns, noun chunks, and entities.
    """
    # nlp process the user's prompt
    prompt_nlp_data = nlp_processing(user_prompt)
    prompt_nouns = prompt_nlp_data['nouns']
    prompt_chunks = prompt_nlp_data['noun_chunks']
    prompt_entities = prompt_nlp_data['entities']
    final_results = []

    # Loop through each search result
    for result in search_response:
        caption_text = " ".join(chunk.text for chunk in result.content)

        # nlp process the caption returned in the search
        caption_nlp_data = nlp_processing(caption_text)
        caption_nouns = caption_nlp_data['nouns']
        caption_chunks = caption_nlp_data['noun_chunks']
        caption_entities = caption_nlp_data['entities']

        # Compare overlap in keywords and phrases by calculating Jaccard score for each category
        jaccard_score_nouns = calculate_jaccard_similarity(prompt_nouns, caption_nouns)
        jaccard_score_chunks = calculate_jaccard_similarity(prompt_chunks, caption_chunks)
        jaccard_score_entities = calculate_jaccard_similarity(prompt_entities, caption_entities)

        # Store all scores together in a structured dictionary
        final_results.append({
            "text": caption_text,
            "cosine_score": result.score,
            "jaccard_scores": {
                "nouns": jaccard_score_nouns,
                "noun_chunks": jaccard_score_chunks,
                "entities": jaccard_score_entities
            }
        })

    return final_results


# search vector store
client = OpenAI(api_key=openai_api_key)
# Load the ID from config file.
config_file = "config.json"
vector_store_id = get_vector_store_id(config_file)
# search for the most similar matches to the user prompt in the vector store.
user_prompt = text_2
search_results = vector_search(vector_store_id, user_prompt)
final_scores = get_comparison_scores(user_prompt, search_results)


# Print the final results with cosine similarity and jaccard simialrity scores
print(f"--- Comparison for Prompt: '{user_prompt[:50]}...' ---\n")
for item in final_scores:
    # Access the nested dictionary for Jaccard scores
    j_scores = item['jaccard_scores']

    print(f"Cosine Score: {item['cosine_score']:.4f}")
    print(f"Jaccard (Nouns): {j_scores['nouns']:.4f}")
    print(f"Jaccard (Chunks): {j_scores['noun_chunks']:.4f}")
    print(f"Jaccard (Entities): {j_scores['entities']:.4f}")
    print(f"Text: \"{item['text'][:100]}...\"\n")