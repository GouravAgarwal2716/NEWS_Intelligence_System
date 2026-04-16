import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preprocess import TextPreprocessor

class NewsClassifier:
    """
    Class to train and use a news classification model.
    """
    def __init__(self, model_path='news_classifier.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.preprocessor = TextPreprocessor()
        
        # AG News labels mapping
        self.id_to_label = {
            1: "Politics/World",
            2: "Sports",
            3: "Business",
            4: "Tech"
        }

    def train(self, train_csv_path):
        """
        Trains the model using the provided CSV file.
        """
        print(f"Loading data from {train_csv_path}...")
        df = pd.read_csv(train_csv_path)
        
        # AG News has columns: Class Index, Title, Description
        # We will combine Title and Description for better features
        print("Preprocessing text...")
        df['text'] = df['Title'] + " " + df['Description']
        df['cleaned_text'] = df['text'].apply(self.preprocessor.preprocess)
        
        print("Vectorizing text...")
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df['Class Index']
        
        print("Training Logistic Regression model...")
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
        
        # Save model and vectorizer
        print("Saving model and vectorizer...")
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            
        print("Training completed successfully.")

    def load_model(self):
        """
        Loads the trained model and vectorizer from disk.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            return True
        return False

    def predict(self, text):
        """
        Predicts the category of a given news text.
        """
        if self.model is None or self.vectorizer is None:
            if not self.load_model():
                raise Exception("Model not trained or loaded.")
        
        cleaned = self.preprocessor.preprocess(text)
        X = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(X)[0]
        
        return self.id_to_label.get(prediction, "Unknown")

    def predict_probabilities(self, text):
        """
        Returns the probability distribution over classes for a given text.
        """
        if self.model is None or self.vectorizer is None:
            if not self.load_model():
                raise Exception("Model not trained or loaded.")
        
        cleaned = self.preprocessor.preprocess(text)
        X = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(X)[0]
        
        return {self.id_to_label[i+1]: float(p) for i, p in enumerate(probs)}

    def evaluate(self, test_csv_path):
        """
        Evaluates the model on a test dataset.
        """
        if self.model is None or self.vectorizer is None:
            self.load_model()
            
        df = pd.read_csv(test_csv_path)
        df['text'] = df['Title'] + " " + df['Description']
        df['cleaned_text'] = df['text'].apply(self.preprocessor.preprocess)
        
        X = self.vectorizer.transform(df['cleaned_text'])
        y_true = df['Class Index']
        y_pred = self.model.predict(X)
        
        print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=[self.id_to_label[i] for i in range(1, 5)]))

if __name__ == "__main__":
    classifier = NewsClassifier()
    # To train: classifier.train('data/train.csv')
    # To evaluate: classifier.evaluate('data/test.csv')
    
    # Quick test
    if classifier.load_model():
        test_text = "The star athlete won the gold medal after a stunning performance."
        print(f"Test Classify: {test_text}")
        print(f"Result: {classifier.predict(test_text)}")
    else:
        print("Model not found. Please run training first.")
