import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re

class ExtractiveSummarizer:
    """
    Extractive summarization using word frequency scoring.
    """
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            
        self.stop_words = set(stopwords.words('english'))

    def summarize(self, text, num_sentences=2):
        """
        Summarizes the input text into a specified number of sentences.
        """
        if not text or len(text.strip()) == 0:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text

        # Tokenize into words and calculate frequency
        words = word_tokenize(text.lower())
        word_frequencies = Counter()
        for word in words:
            if word.isalnum() and word not in self.stop_words:
                word_frequencies[word] += 1

        if not word_frequencies:
            return " ".join(sentences[:num_sentences])

        # Normalize frequencies
        max_freq = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_freq

        # Score sentences
        sentence_scores = {}
        for sent in sentences:
            sentence_words = word_tokenize(sent.lower())
            score = 0
            for word in sentence_words:
                if word in word_frequencies:
                    score += word_frequencies[word]
            sentence_scores[sent] = score

        # Pick top sentences
        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_sentences = sorted_sentences[:num_sentences]
        
        # Sort top sentences to maintain original order
        top_sentences.sort(key=lambda x: sentences.index(x[0]))
        
        summary = " ".join([sent[0] for sent in top_sentences])
        return summary

if __name__ == "__main__":
    summarizer = ExtractiveSummarizer()
    sample_text = """
    Artificial intelligence is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving".
    This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
    """
    print("Original Text:\n", sample_text)
    print("\nSummary:\n", summarizer.summarize(sample_text, num_sentences=2))
