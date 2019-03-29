import json

#Loading data
fr_foot_test = json.load(open("corpus/fr/fr.foot.test.json"))
fr_ftb_train = json.load(open("corpus/fr/fr.ftb.train.json"))
fr_gsd_train = json.load(open("corpus/fr/fr.gsd.train.json"))
fr_natdis_test = json.load(open("corpus/fr/fr.natdis.test.json"))
fr_partut_train = json.load(open("corpus/fr/fr.partut.train.json"))
fr_pud_train = json.load(open("corpus/fr/fr.pud.train.json"))
fr_sequoia_train = json.load(open("corpus/fr/fr.sequoia.train.json"))
fr_spoken_train = json.load(open("corpus/fr/fr.spoken.train.json"))

datasets = [fr_foot_test, fr_ftb_train, fr_gsd_train, fr_natdis_test, fr_partut_train, fr_pud_train, fr_sequoia_train, fr_spoken_train]
for d in datasets:
    print("##########################################################")
    print(len(d))
    print(sum(len(i[0]) for i in d))   
    for j in range(10):
        print("-----------------")
        print(d[j])