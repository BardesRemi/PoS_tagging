import json
from collections import Counter, defaultdict
from itertools import product
import math

#Loading data
fr_foot_test = json.load(open("corpus/fr/fr.foot.test.json"))

fr_ftb_dev = json.load(open("corpus/fr/fr.ftb.dev.json"))
fr_ftb_test = json.load(open("corpus/fr/fr.ftb.test.json"))
fr_ftb_train = json.load(open("corpus/fr/fr.ftb.train.json"))

fr_gsd_dev = json.load(open("corpus/fr/fr.gsd.dev.json"))
fr_gsd_test = json.load(open("corpus/fr/fr.gsd.test.json"))
fr_gsd_train = json.load(open("corpus/fr/fr.gsd.train.json"))

fr_natdis_test = json.load(open("corpus/fr/fr.natdis.test.json"))

fr_partut_dev = json.load(open("corpus/fr/fr.partut.dev.json"))
fr_partut_test = json.load(open("corpus/fr/fr.partut.test.json"))
fr_partut_train = json.load(open("corpus/fr/fr.partut.train.json"))

fr_pud_dev = []#json.load(open("corpus/fr/fr.pud.dev.json")) was empty
fr_pud_test = json.load(open("corpus/fr/fr.pud.test.json"))
fr_pud_train = json.load(open("corpus/fr/fr.pud.train.json"))

fr_sequoia_dev = json.load(open("corpus/fr/fr.sequoia.dev.json"))
fr_sequoia_test = json.load(open("corpus/fr/fr.sequoia.test.json"))
fr_sequoia_train = json.load(open("corpus/fr/fr.sequoia.train.json"))

fr_spoken_dev = json.load(open("corpus/fr/fr.spoken.dev.json"))
fr_spoken_test = json.load(open("corpus/fr/fr.spoken.test.json"))
fr_spoken_train = json.load(open("corpus/fr/fr.spoken.train.json"))

#creating list for each different group of dataset
foot = [fr_foot_test]
natdis = [fr_natdis_test]
ftb = [fr_ftb_train, fr_ftb_test]
gsd = [fr_gsd_train, fr_gsd_test]
partut = [fr_partut_train, fr_partut_test]
pud = [fr_pud_train, fr_pud_test]
sequoia = [fr_sequoia_train, fr_sequoia_test]
spoken =  [fr_spoken_train, fr_spoken_test]

"""ftb = [fr_ftb_dev, fr_ftb_test, fr_ftb_train]
gsd = [fr_gsd_dev, fr_gsd_test, fr_gsd_train]
partut = [fr_partut_dev, fr_partut_test, fr_partut_train]
pud = [fr_pud_dev,fr_pud_test, fr_pud_train]
sequoia = [fr_sequoia_dev, fr_sequoia_test, fr_sequoia_train]
spoken =  [fr_spoken_dev, fr_spoken_test, fr_spoken_train]"""

train_datasets = [("train_ftb", fr_ftb_train),
                  ("train_gsd", fr_gsd_train),
                  ("train_partut", fr_partut_train),
                  ("train_pud", fr_pud_train),
                  ("train_sequoia", fr_sequoia_train),
                  ("train_spoken", fr_spoken_train)]
test_datasets = [("test_foot", fr_foot_test),
                 ("test_natdis", fr_natdis_test),
                 ("test_ftb", fr_ftb_test),
                 ("test_gsd", fr_gsd_test),
                 ("test_partut", fr_partut_test),
                 ("test_pud", fr_pud_test),
                 ("test_sequoia", fr_sequoia_test),
                 ("test_spoken", fr_spoken_test)]

datasets = [ftb, gsd, partut, pud, sequoia, spoken]
# full_datasets = [foot, ftb, gsd, natdis, partut, pud, sequoia, spoken]
full_datasets = [fr_foot_test, fr_natdis_test, fr_ftb_train, fr_ftb_test, fr_gsd_train, fr_gsd_test, fr_partut_train, fr_partut_test, fr_pud_train, fr_pud_test, fr_sequoia_train, fr_sequoia_test, fr_spoken_train, fr_spoken_test]

#make a dictionnary of all the words inside a dataset
def make_dict(dataset):
    words_dict= {}
    cpt = 0
    for words, labels in dataset:
        for word in words:
            if not word in words_dict:
                words_dict[word] = cpt
                cpt += 1
    return words_dict

#make a dictionnary of all the words inside a list of dataset
def make_dicts(datasets):
    words_dict = {}
    cpt = 0
    for dataset in datasets:
        for words, labels in dataset:
            for word in words:
                if not word in words_dict:
                    words_dict[word] = cpt
                    cpt += 1
    return words_dict

def make_txt(dataset):
    title = dataset[0] + ".txt"
    filename = "text/" + title
    f = open(filename, "w")
    for sentence, labels in dataset[1] :
        line = " ".join(sentence)
        f.write(line + "\n")
    f.close()


#display number of sentences and words of each dataset group
"""for d in full_datawords
    size = 0
    for data in d:
        size += len(dwords
    words, chars = mawords
    print(len(data))
    print(len(words))words
    print("------------")"""

# Compute the OOV rate in a corpus
def OOV_calculation (train, test):
    train_words = make_dict(train)
    test_words = make_dict(test)
    cpt = 0
    for w in test_words:
        if not w in train_words:
            cpt += 1
    return cpt/len(test_words) * 100

def OOV_words (train, test):
    train_words = make_dict(train)
    test_words = make_dict(test)
    res = []
    for w in test_words:
        if not w in train_words:
            res.append(w)
    return res

def ambiguous_words(train):
    words_dict= {}
    res = []
    for words, labels in train:
        for word in words:
            if not word in words_dict:
                words_dict[word] = [labels]
            else:
                if not labels in words_dict[word]:
                    words_dict[word].append(labels)
    
    for w, labels in words_dict:
        if len(labels > 1):
            res.append(w)
    return res


# Display the OOV rate
# for d in datasets:
#     print(OOV_calculation(d[0],d[1]))
#     print("------------")

# Function that generates all the ngram from a sentence in a list form
def ngram(sentence, n):
    return [sentence[i:i+n] for i in range(len(sentence) - n + 1)]

# Function that generates a long string consisting of all the sentences in a dataset concatened
def generate_corp(dataset):
    res = ""
    for sentence, labels in dataset:
        res += " ".join(sentence)
    return res

# Function that generates the alphabet , e.g. the set of all the present characters, from a corpus
def generate_alphabet(corpus):
    corp = ""
    for dataset in corpus:
        corp += "".join(generate_corp(dataset[1]))
    return set(corp)

def generate_alphabet_V2(corpus):
    corp = ""
    for dataset in corpus:
        corp = generate_corp(dataset)
    return set(corp)

#print(len(generate_alphabet(sequoia)))
#print(generate_alphabet(sequoia))
# print(generate_corp(ftb[1]))

""" 
 Function that generates a dict associating all n-grams existing in a corpus to their # of appearances
 Return 2 dictionaries, one for each dataset (train, test) of the corpus
 Note : in reality we return the # of appearances + 1 of each n-grams to not give an infinite weight to non-present n-grams
 later in the probability formula
"""
def generate_N(corpus):
    #generate N
    sentence = ""
    N_train = defaultdict(lambda: 1)
    N_test = defaultdict(lambda: 1)
    for words, labels in corpus[0]:
        sentence = " ".join(words)
        counts = Counter(ngram(sentence, 3))
        for k,v in counts.items():
            if k in N_train:
                N_train[k] += v
            else :
                N_train.update({k: v + 1})
    for words, labels in corpus[1]:
        sentence = " ".join(words)
        counts = Counter(ngram(sentence, 3))
        for k,v in counts.items():
            if k in N_test:
                N_test[k] += v
            else :
                N_test.update({k: v + 1})
    return N_train, N_test

"""
 Returns the probability of a n-gram in a dataset given :
 - the n-gram
 - the alphabet of the corpus
 - the length in characters of the corpus/dataset
 - the dict n-grams/occurences for each dataset of the corpus
 - the parameter n of the n-grams
 The probability of the n_grams is the # of occurences of this n-grams (+1) in this dataset divided by the total number of 
 existing  unique n-grams given the alphabet + the total number of n-grams in the dataset
"""
def prob(tri_gram, alphabet, count, N, n):
    return N[tri_gram] / (len(alphabet) ** n + count)

# Function that computes the KL divergence from a corpus
def KL(corpus, alphabet, all_ngrams):
    N_train, N_test = generate_N(corpus)
    somme = 0
    count_train = sum(N_train.values())
    count_test = sum(N_test.values())
    for k in all_ngrams:
        prob_train = prob(k, alphabet, count_train, N_train, 3)
        prob_test = prob(k, alphabet, count_test, N_test, 3)
        somme +=  prob_test * math.log(prob_test/prob_train)
    return somme

def KL_display(trains, tests):
    alphabet = generate_alphabet(tests)
    all_ngrams = list("".join(k) for k in product(alphabet, repeat=3))
    filename = "Metrics_computations.csv"
    f = open(filename, "w")
    f.write("train_Set 1 ; test_Set 2 ; OOV ; KL_Divergence;\n")
    for train_set in trains:
        for test_set in tests:
            f.write(train_set[0] + ";" + test_set[0] + ";" + str(OOV_calculation(train_set[1], test_set[1])) + " % ;" + str(KL( [train_set[1], test_set[1]], alphabet, all_ngrams)) +";\n" )
            #print("Comparaison de " + train_set[0] + " et " + test_set[0])
            #print(KL( [train_set[1], test_set[1]], alphabet, all_ngrams))
    f.close()


#KL_display(train_datasets, test_datasets)
corpus = [fr_pud_train, fr_partut_train]
alphabet = generate_alphabet_V2(corpus)
all_ngrams = list("".join(k) for k in product(alphabet, repeat=3))
#print(KL(corpus, alphabet, all_ngrams))

# for dataset in train_datasets :
#     make_txt(dataset)
# for dataset in test_datasets :
#     make_txt(dataset)

#KL_display([("ftb_test" ,fr_ftb_test)], [("ftb_test", fr_ftb_test)])

#print(train_datasets)

"""
Perplexity : 
sur KenLM : https://kheafield.com/code/kenlm/
Suivre les instructions pour compiler le programme
Suivre la commande pour transformer un fichier texte en .arpa utilisé par le modèle
( Il faut transformer le JSON en .txt, on ne garde que les phrases pas les labels, 
une phrase par ligne)
Puis on peut calculer la perplexité (avec query ?) faut chercher un peu apparemment
"""