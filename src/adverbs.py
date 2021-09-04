import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



if __name__ == '__main__':
    f = open('../data/test_clean.txt')
    raw = f.read()
    tokens = nltk.word_tokenize(raw)
    pos_tags = nltk.pos_tag(tokens)
    dictionary = {}
    for pos in pos_tags:
        #RB is abbrevaition for adverb
       if(pos[1] == 'RB'):
           word = pos[0]
           dictionary[word] = dictionary.get(word, 0) + 1

    print(dictionary)