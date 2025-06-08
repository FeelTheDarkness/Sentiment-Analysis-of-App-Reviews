import nltk

# Download all required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')  # Specifically for the punkt tables
nltk.download('omw-1.4')    # Open Multilingual WordNet data

print("All NLTK data downloaded successfully!")