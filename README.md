# News Intelligence System 📰

A smart NLP pipeline for news article classification and extractive summarization.

## Features
- **Preprocessing**: Robust text cleaning, tokenization, stopword removal, and lemmatization.
- **Classification**: TF-IDF + Logistic Regression model trained on 120,000 news articles.
- **Summarization**: Extractive summarizer using word frequency scoring.
- **Pipeline**: Unified interface to process raw text and get structured insights.

## Project Structure
- `preprocess.py`: Text cleaning and normalization.
- `classifier.py`: Training and prediction logic for classification.
- `summarizer.py`: Extractive summarization logic.
- `pipeline.py`: Unified pipeline class.
- `main.py`: Command-line interface.
- `streamlit_app.py`: Interactive web application.
- `News_NLP_Pipeline.ipynb`: Comprehensive notebook for submission and demo.
- `news_classifier.pkl`: Trained model (generated after training).
- `tfidf_vectorizer.pkl`: Fitted vectorizer (generated after training).

## Installation
Ensure you have the required libraries:
```bash
pip install pandas scikit-learn nltk matplotlib seaborn
```

## Usage

### Web Application
To launch the interactive dashboard:
```bash
streamlit run streamlit_app.py
```

### Command Line
To analyze an article:
```bash
python main.py --text "Your news article text here"
```

To analyze from a file:
```bash
python main.py --file path/to/article.txt
```

### Python API
```python
from pipeline import NewsPipeline

pipeline = NewsPipeline()
result = pipeline.process_article("Article text...")
print(f"Category: {result['category']}")
print(f"Summary: {result['summary']}")
```

## Dataset & Model Performance
The model is trained on the AG News dataset with the following mapping:
1. **World** -> Politics/World
2. **Sports** -> Sports
3. **Business** -> Business
4. **Sci/Tech** -> Tech

### Evaluation Results (on 7,600 test samples):
- **Accuracy**: 90.66%
- **F1-Score**: ~0.91

| Category | Precision | Recall | F1-Score |
|---|---|---|---|
| Politics/World | 92% | 90% | 91% |
| Sports | 95% | 98% | 96% |
| Business | 88% | 87% | 87% |
| Tech | 88% | 88% | 88% |

## Summarization Method
Extractive summarization is implemented by scoring sentences based on the frequency of important words (excluding stopwords). This provides a quick preview of the article's most significant points without changing the original wording.
