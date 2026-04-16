import argparse
from pipeline import NewsPipeline
import sys

def main():
    parser = argparse.ArgumentParser(description="News Classifier and Summarizer Pipeline")
    parser.add_argument("--text", type=str, help="The news article text to process")
    parser.add_argument("--file", type=str, help="Path to a text file containing the news article")
    parser.add_argument("--sentences", type=int, default=2, help="Number of sentences for the summary")
    
    args = parser.parse_args()

    # Initialize the pipeline
    try:
        pipeline = NewsPipeline()
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    article_text = ""
    if args.text:
        article_text = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                article_text = f.read()
        except FileNotFoundError:
            print(f"Error: File not found at {args.file}")
            return
    else:
        # Interactive mode if no arguments provided
        print("--- News Intelligence System ---")
        print("Enter article text (press Ctrl+Z on Windows or Ctrl+D on Linux then Enter to finish):")
        article_text = sys.stdin.read()

    if not article_text.strip():
        print("No text provided. Exiting.")
        return

    print("\nProcessing article...\n")
    result = pipeline.process_article(article_text, summary_sentences=args.sentences)

    print("=" * 30)
    print(f"CLASS: {result['category']}")
    print("-" * 30)
    print(f"SUMMARY:\n{result['summary']}")
    print("=" * 30)

if __name__ == "__main__":
    main()
