import nltk
import pandas as pd
import os
import re

from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def analyze(input_directory, output_directory):
    listdir = os.listdir(input_directory)
    for dir in ["test_sentiment.txt"]:
        print(dir)
        output_file_name = os.path.splitext(dir)[0] + '.csv'
        input_full_path = '%s/%s' % (input_directory, dir)
        output_full_path = "%s/%s" %(output_directory, output_file_name)
        single_analyze(input_full_path,output_full_path)

#TODO add rows in sentence
def single_analyze(input_file_path, output_file_path):
    file = open(input_file_path)
    raw_text = file.read()
    setences = nltk.sent_tokenize(raw_text)
    arr = []
    for setence in setences:
        arr.append(sia.polarity_scores(setence))
    frame = pd.DataFrame(arr)
    frame.to_csv(output_file_path)
    print(frame)

if __name__ == '__main__':
    analyze("../data/raw_text", "../data/sentiment")
