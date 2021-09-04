import nltk
from collections import defaultdict

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



if __name__ == '__main__':
    f = open('../data/test_clean.txt')
    raw = f.read()
    tokens = nltk.word_tokenize(raw)
    pos_tags = nltk.pos_tag(tokens)
    dictionary = defaultdict(int)
    for pos in pos_tags:
        #RB is abbrevaition for adverb
       if(pos[1] == 'RB'):
           word = pos[0]
           dictionary[word] +=1
    for w in sorted(dictionary, key=dictionary.get, reverse=True):
        print(w, dictionary[w])