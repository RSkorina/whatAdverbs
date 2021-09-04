import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



if __name__ == '__main__':
    f = open('../data/test_clean.txt')
    raw = f.read()
    tokens = nltk.word_tokenize(raw)
    pos = nltk.pos_tag(tokens)


    print(pos)