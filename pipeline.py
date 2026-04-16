from classifier import NewsClassifier
from summarizer import ExtractiveSummarizer
from preprocess import TextPreprocessor

class NewsPipeline:
    """
    Unified pipeline for news classification and summarization.
    """
    def __init__(self, model_path='news_classifier.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
        # Initialize components
        self.classifier = NewsClassifier(model_path, vectorizer_path)
        self.summarizer = ExtractiveSummarizer()
        self.preprocessor = TextPreprocessor()
        
        # Load the model if it exists
        try:
            self.classifier.load_model()
        except:
            pass

    def process_article(self, text, summary_sentences=2):
        """
        Classifies and summarizes an article.
        """
        if not text:
            return {"error": "Empty text provided"}

        # Classification
        try:
            category = self.classifier.predict(text)
        except Exception as e:
            category = f"Error classifying: {str(e)} (Model might not be trained)"

        # Summarization
        summary = self.summarizer.summarize(text, num_sentences=summary_sentences)

        return {
            "category": category,
            "summary": summary
        }

    def get_probabilities(self, text):
        """
        Returns classification probabilities for the given text.
        """
        return self.classifier.predict_probabilities(text)

if __name__ == "__main__":
    # Test the pipeline
    pipeline = NewsPipeline()
    sample_news = """
    Tech giant Apple has just announced its latest lineup of iPhones at the Steve Jobs Theater. 
    The new models feature advanced camera systems, faster processors, and longer battery life. 
    Investors responded positively to the news, with Apple's stock price seeing a 2% jump in late-day trading. 
    The event also highlighted Apple's commitment to sustainability, using recycled materials in the new devices.
    """
    
    result = pipeline.process_article(sample_news)
    print("Article Analysis:")
    print("-" * 20)
    print(f"Category: {result['category']}")
    print(f"Summary: {result['summary']}")
