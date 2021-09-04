import nltk
from collections import defaultdict

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def tokenize_file(file_path):
    file = open(file_path)
    raw_text = file.read()
    words = nltk.word_tokenize(raw_text)
    return nltk.pos_tag(words)

def count_adverbs(words_with_pos_tags):
    adverb_instances = defaultdict(int)
    adverbCounter = 0
    for pos in words_with_pos_tags:
        #RB is abbrevaition for adverb
        if(pos[1] == 'RB'):
            word = pos[0]
            adverb_instances[word] +=1
            adverbCounter+=1
    return adverb_instances, adverbCounter


def analyze(file_path):
    words_with_pos_tags = tokenize_file(file_path)
    adverb_instances = defaultdict(int)
    adverbCounter = 0
    for pos in words_with_pos_tags:
        #RB is abbrevaition for adverb
        if(pos[1] == 'RB'):
            word = pos[0]
            adverb_instances[word] +=1
            adverbCounter+=1
    for w in sorted(adverb_instances, key=adverb_instances.get, reverse=True):
        print(w, adverb_instances[w])
    adverb_percentage = adverbCounter/len(words_with_pos_tags) * 100
    print("adverb percentage is %f"%adverb_percentage)

if __name__ == '__main__':
    analyze('../data/test_clean.txt')