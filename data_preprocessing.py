import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
from collections import Counter
from nltk.stem import WordNetLemmatizer

# hm_lines = 100

lemmatizer = WordNetLemmatizer()
def create_lexicon(true_contents, fake_contents):
    lexicon = []

    for line in true_contents:
        words = word_tokenize(line)
        lexicon += list(words)
#    print("completed reading true file")

    for line in fake_contents:
        words = word_tokenize(line)
        lexicon += list(words)
#    print("completed reading fake file")
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        if 30 > w_counts[w] > 5:
            l2.append(w)
    # print("lemmatizing")
    # lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
#    print("lexicon created")
    print("length of lexicon  : ", len(l2))
    return l2


def sample_handling(sample, lexicon, classification):
    featureset = []
#    print("started reading sample file")
    with open(sample, "r") as f:
        contents = f.readlines()
#        print("sample file read starting vectorizing")
        for line in contents:
            all_words = word_tokenize(line.lower())
            # all_words = [lemmatizer.lemmatize(i) for i in all_words]
            features = np.zeros(len(lexicon))
            for word in all_words:
                if word in lexicon:
                    index = lexicon.index(word)
                    features[index] +=1 
            features = list(features)
            featureset.append([features, classification])
#    print("sample file vectorizing completed")
    return featureset

def create_train_test_data(true, false, lexicon, test_size=0.1):
    features = []

    features += sample_handling(true, lexicon, [1,0])
    features += sample_handling(false, lexicon, [0,1])
    random.shuffle(features)
    features = np.array(features)
#    print(features.shape)
#    print(features.ndim)

    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    return train_x, train_y, test_x, test_y