import nltk
import pandas as pd
import os
import re

from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

word_tokenizer = nltk.RegexpTokenizer(r'\w+')
sia = SentimentIntensityAnalyzer()

def analyze(input_directory, output_directory):
    listdir = os.listdir(input_directory)
    for file_name in listdir:
        input_file_full_path = '%s/%s' % (input_directory, file_name)
        output_file_name = os.path.splitext(file_name)[0] + '.csv'
        output_file_full_path = "%s/%s" %(output_directory, output_file_name)
        single_analyze(input_file_full_path,output_file_full_path)

#TODO add rows in sentence
def single_analyze(input_file_path, output_file_path):
    file = open(input_file_path, 'r', encoding='utf-8')
    raw_text = file.read()
    setences = nltk.sent_tokenize(raw_text)
    arr = []
    for setence in setences:
        count = len(word_tokenizer.tokenize(setence))
        scores = sia.polarity_scores(setence)
        scores['word_count'] = count
        arr.append(scores)
    frame = pd.DataFrame(arr)
    frame.to_csv(output_file_path)
    print(frame)

def count_words(text):
    tokens = word_tokenizer.tokenize(text)
    print(tokens)

if __name__ == '__main__':
    analyze("../data/short_sample_raw_text", "../data/sentiment")
