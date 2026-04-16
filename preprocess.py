import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """
    Class to handle text cleaning and preprocessing for NLP tasks.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Cleans text by removing special characters, numbers, and converting to lowercase.
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess(self, text):
        """
        Full preprocessing pipeline: cleaning, tokenization, stopword removal, and lemmatization.
        """
        cleaned = self.clean_text(text)
        tokens = word_tokenize(cleaned)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words
        ]
        
        return " ".join(processed_tokens)

if __name__ == "__main__":
    # Test
    preprocessor = TextPreprocessor()
    sample = "The stock market is rallying! Tech companies like Apple are up by 5%."
    print(f"Original: {sample}")
    print(f"Processed: {preprocessor.preprocess(sample)}")
