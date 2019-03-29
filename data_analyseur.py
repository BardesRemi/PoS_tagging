import json

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

# fr_pud_dev = json.load(open("corpus/fr/fr.pud.dev.json"))
fr_pud_test = json.load(open("corpus/fr/fr.pud.test.json"))
fr_pud_train = json.load(open("corpus/fr/fr.pud.train.json"))

fr_sequoia_dev = json.load(open("corpus/fr/fr.sequoia.dev.json"))
fr_sequoia_test = json.load(open("corpus/fr/fr.sequoia.test.json"))
fr_sequoia_train = json.load(open("corpus/fr/fr.sequoia.train.json"))

fr_spoken_dev = json.load(open("corpus/fr/fr.spoken.dev.json"))
fr_spoken_test = json.load(open("corpus/fr/fr.spoken.test.json"))
fr_spoken_train = json.load(open("corpus/fr/fr.spoken.train.json"))

# def make_dict(dataset):
#     words_dict, chars_dict = {},{}
#     cpt = 0
#     for words, labels in dataset:
#         for word in words:
#             if not word in words_dict:
#                 words_dict[word] = cpt
#                 cpt += 1
#             for c in word:
#                 if not c in chars_dict:
#                     chars_dict[c] = cpt
#                     cpt += 1
#     return words_dict, chars_dict

def make_dicts(datasets):
    words_dict, chars_dict = {},{}
    cpt = 0
    for dataset in datasets:
        for words, labels in dataset:
            for word in words:
                if not word in words_dict:
                    words_dict[word] = cpt
                    cpt += 1
                for c in word:
                    if not c in chars_dict:
                        chars_dict[c] = cpt
                        cpt += 1
    return words_dict, chars_dict

foot = [fr_foot_test]
ftb = [fr_ftb_dev, fr_ftb_test, fr_ftb_train]
gsd = [fr_gsd_dev, fr_gsd_test, fr_gsd_train]
natdis = [fr_natdis_test]
partut = [fr_partut_dev, fr_partut_test, fr_partut_train]
pud = [fr_pud_test, fr_pud_train]
sequoia = [fr_sequoia_dev, fr_sequoia_test, fr_sequoia_train]
spoken =  [fr_spoken_dev, fr_spoken_test, fr_spoken_train]

datasets = [foot, ftb, gsd, natdis, partut, pud, sequoia, spoken]
for d in datasets:
    size = 0
    for data in d:
        size += len(data)
    words, chars = make_dicts(d)
    print(len(data))
    print(len(words))
    print("------------")
    # print("##########################################################")
    # print(sum(len(i[0]) for i in d))   
    # for j in range(10):
    #     print("-----------------")
    #     print(d[j])