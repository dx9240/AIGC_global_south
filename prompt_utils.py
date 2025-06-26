import spacy
from sympy.codegen import Print

nlp = spacy.load("en_core_web_trf")

text_1 = "A mature woman wearing a blue silk sari standing against a background of moonlit Jaipur."
text_2 = "An ancient, blue maple tree clinging to a cliff, overlooking a purple lake. There is blue moss growing on the cliff, two crescent moons and jupiter in the sky, and a spaceship flying across the sky in the distance. The painting is in a traditional Chinese style, with futuristic caligraphy."

synthetic_1 = "This painting embodies the traditional Chinese landscape style, known as \"shan shui,\" which translates to \"mountain-water.\" The use of vibrant blues and greens creates a serene and harmonious atmosphere, emphasizing the natural beauty of the scene. The waterfall acts as a focal point, drawing the viewer's eye through the composition and symbolizing the flow of life. The subtle inclusion of human figures at the bottom of the painting adds a sense of scale and highlights humanity's smallness in the face of nature's grandeur. The misty mountains in the background suggest depth and evoke a sense of mystery and tranquility. Overall, the artwork beautifully captures the essence of nature's majesty and the traditional aesthetic values of Chinese landscape painting."
synthetic_2 = "This artwork presents a striking interplay between the realistic depiction of the human form and abstract elements. The central figure, a woman, is portrayed with a dramatic sense of movement and emotion, her body draped in a white garment accentuated by vivid red fabric. The red not only serves as a visual anchor but also adds a sense of urgency and intensity to the composition. The reaching hand, seemingly emerging from the background, introduces a narrative tension, suggesting themes of escape or entanglement."
synthetic_3 = "The artist skillfully blends realism with abstraction, as seen in the loose brushstrokes and blurred lines on the left side, which contrast with the detailed rendering of the woman's form. This juxtaposition creates a dynamic visual tension, inviting viewers to contemplate the relationship between the tangible and the ethereal."

doc = nlp(synthetic_3)

# Noun chunks - lemmatized head nouns
noun_chunks = set(chunk.lemma_.lower() for chunk in doc.noun_chunks)

def extract_meaningful_nouns(text):
    doc = nlp(text)

    # Use POS tagging
    nouns = {token.lemma_.lower() for token in doc
             if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop}
    return nouns

# Named entities - surface form, lowercase
# list of entity lables can be found here: https://github.com/explosion/spaCy/discussions/9147
entity_labels = {"GPE", "PERSON", "ORG", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE"}
named_entities = set(
    ent.text.lower() for ent in doc.ents if ent.label_ in entity_labels
)

print("NOUNs")
print(noun_chunks)

print("SPECIFIC NOUNS")
print(extract_meaningful_nouns(synthetic_3))

print("ENTITIES")
print(named_entities)