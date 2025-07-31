#!/usr/bin/env python3

import sys
import os

# Add the virtual environment site-packages to the path
venv_site_packages = "/home/parantapsinha/Parantap/Sentiment-Analysis-of-App-Reviews/.analysis/lib/python3.11/site-packages"
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

try:
    import spacy

    print(f"spaCy location: {spacy.__file__}")

    # Try to load the model
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    print("✅ spaCy model loaded successfully!")

    # Test basic functionality
    doc = nlp("This is a test sentence.")
    print(f"✅ Basic spaCy functionality works: {len(doc)} tokens")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
