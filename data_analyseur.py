import json
from collections import Counter, defaultdict

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


datasets = [ftb, gsd, partut, pud, sequoia, spoken]
full_datasets = [foot, ftb, gsd, natdis, partut, pud, sequoia, spoken]

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

#display number of sentences and words of each dataset group
"""for d in full_datawords
    size = 0
    for data in d:
        size += len(dwords
    words, chars = mawords
    print(len(data))
    print(len(words))words
    print("------------")"""


def OOV_calculation (train, test):
    train_words = make_dict(train)
    test_words = make_dict(test)
    cpt = 0
    for w in test_words:
        if not w in train_words:
            cpt += 1
    return cpt/len(test_words) * 100

for d in datasets:
    print(OOV_calculation(d[0],d[1]))
    print("------------")

def ngram(sentence, n):
    return [sentence[i:i+n] for i in range(len(sentence) - n + 1)]

def generate_corp(dataset):
    res = ""
    for sentence, labels in dataset:
        res += " ".join(sentence)
    return res

def generate_alphabet(corpus):
    crop = ""
    for dataset in corpus:
        corp = "".join(generate_corp(dataset))
    return set(corp)

print(len(generate_alphabet(ftb)))
print(generate_alphabet(ftb))
# print(generate_corp(ftb[1]))

def generate_N(corpus):
    #generate N
    N_train = defaultdict(lambda:1)
    N_test = defaultdict(lambda:1)
    for words, labels in corpus[0]:
        sentence = " ".join(words)
        counts = Counter(ngram(sentence, 3))
        N_train.update({k: v + 1 for k,v in counts.items()})
    for words, labels in corpus[1]:
        sentence = " ".join(words)
        counts = Counter(ngram(sentence, 3))
        N_test.update({k: v + 1 for k,v in counts.items()})
    return N_train, N_test

def prob(tri_gram, alphabet, length, N, n):
    return N[tri_gram] / (len(alphabet) ** n + length - n + 1)

def KL():
    #itérer sur tous tri_gram possibles
    #pour chaque tri gram appliquer formule
    #sommer les résultats
    #victoire
    pass