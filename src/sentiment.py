import nltk
import pandas as pd
import os

from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

word_tokenizer = nltk.RegexpTokenizer(r'\w+')
sia = SentimentIntensityAnalyzer()

def analyze_directory_sentiment(input_directory, output_directory):
    for file_name in os.listdir(input_directory):
        input_file_full_path = '%s/%s' % (input_directory, file_name)
        output_file_name = os.path.splitext(file_name)[0] + '.csv'
        output_file_full_path = "%s/%s" % (output_directory, output_file_name)
        file = open(input_file_full_path, 'r', encoding='utf-8')
        raw_text = file.read()
        frame = analyze_sentiment(raw_text)
        frame.to_csv(output_file_full_path)

def analyze_sentiment(raw_text):
    sentences = nltk.sent_tokenize(raw_text)
    sentiment_of_sentence = []
    for sentence in sentences:
        count = len(word_tokenizer.tokenize(sentence))
        sentence_scores = sia.polarity_scores(sentence)
        sentence_scores['word_count'] = count
        sentiment_of_sentence.append(sentence_scores)
    return pd.DataFrame(sentiment_of_sentence)

if __name__ == '__main__':
    analyze_directory_sentiment("../data/short_sample_raw_text", "../data/sentiment")
