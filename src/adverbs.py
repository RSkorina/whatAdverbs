import nltk
from collections import defaultdict

# requirements for nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def tokenize_file(file_path):
    file = open(file_path)
    raw_text = file.read()
    words = nltk.word_tokenize(raw_text)
    return nltk.pos_tag(words)


def count_adverbs(words_with_pos_tags, only_ly):
    adverb_instances = defaultdict(int)
    adverb_counter = 0
    for word_with_pos_tag in words_with_pos_tags:
        word = word_with_pos_tag[0]
        # RB is abbrevaition for adverb
        if (word_with_pos_tag[1] == 'RB' and (not only_ly or word[-2:] == 'ly')):
            adverb_instances[word] += 1
            adverb_counter += 1
    return adverb_instances, adverb_counter


def print_results(adverb_instances, adverb_counter, words_with_pos_tags):
    for adverb in sorted(adverb_instances, key=adverb_instances.get, reverse=True):
        print(adverb, adverb_instances[adverb])
    adverb_percentage = adverb_counter / len(words_with_pos_tags) * 100
    print("Total percentage of text that are adverbs is %.2f%%" % adverb_percentage)


def analyze(file_path, only_ly=False):
    words_with_pos_tags = tokenize_file(file_path)
    results = count_adverbs(words_with_pos_tags, only_ly)
    print_results(*results, words_with_pos_tags)


if __name__ == '__main__':
    analyze('../data/acrossTheRiverAndIntoTheTrees.txt', True)
