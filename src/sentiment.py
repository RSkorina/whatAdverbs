import nltk
import pandas as pd
import os

from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def analyze():
    print(os.listdir("../data/raw_text"))

def single_analyze(file_path):
    file = open(file_path)
    raw_text = file.read()
    setences = nltk.sent_tokenize(raw_text)
    arr = []
    for setence in setences:
        arr.append(sia.polarity_scores(setence))
    frame = pd.DataFrame(arr)
    #frame.to_csv('')
    print(frame)

if __name__ == '__main__':
    single_analyze('../data/raw_text/test_sentiment.txt')
