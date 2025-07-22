# AIGC_global_south

After installing dependencies, you'll need to install en_core_web_trf transformer (for spacy>=3.0.0) from SpaCy for identifying Named Entities.
https://spacy.io/models/en

Run the file vector_store_setup.py to set up the vector store and add the synthetic captions. This file creates a config.json file which contains the vector store ID. This ID is needed to use the vector store in the cosine similarity search task. By default, the file sets the vector store to be deleted after 30 days, but this can be changed.

If you need to manage the vector store, it can be done through the OpenAI Playground GUI. While the config file overwrites previous vector store IDs with the newest one, the code doesn't yet manage deleting vector stores. Current work-around: vector store can be deleted through the OPenAI Playground GUI.
