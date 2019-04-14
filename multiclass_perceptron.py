import re
import json
from data_analyseur import *
from collections import defaultdict
from operator import itemgetter


class Perceptron:

    def __init__(self, labels):
        self.labels = labels
        # Each feature gets its own weight vector, with one weight for
        # each possible label
        self.weights = defaultdict(lambda: defaultdict(float))
        # The accumulated values of the weight vector at the t-th
        # iteration: sum_{i=1}^{n - 1} w_i
        #
        # The current value (w_t) is not yet added. The key of this
        # dictionary is a pair (feature, label)
        self._accum = defaultdict(int)
        # The last time the feature was changed, for the averaging.
        self._last_update = defaultdict(int)
        # Number of examples seen
        self.n_updates = 0

    def predict(self, features):
        '''Dot-product the features and current weights and return
        the best class.'''
        from operator import itemgetter
        scores = self.score(features)
        maxi_ind = max(enumerate(scores), key=itemgetter(1))[0]
        return self.labels[maxi_ind]
    
    def score(self, features, labels=None):
        """
        Parameters
        ----------

        - features, an iterable
             a sequence of binary features. Each feature must be
             hashable. WARNING: the `value' of the feature is always
             assumed to be 1.
        - labels, a subset of self.labels
             if not None, the score is computed only for these labels
        """
        labs = labels
        if labels == None:
            labs = self.labels
        res = []
        # print(features)
        for l in labs :
            dot_product = sum(self.weights[key][l] for key in features )
            res += [dot_product]
        return res


    def update(self, truth, guess, features):
        def upd_feat(label, feature, v):
            param = (label, feature)
            self._accum[param] += (self.n_updates -
                                   self._last_update[param]) * self.weights[feature][label]
            self._last_update[param] = self.n_updates
            self.weights[feature][label] += v
            
        self.n_updates += 1

        if truth == guess:
            return

        for f in features:
            upd_feat(truth, f, 1.0)
            upd_feat(guess, f, -1.0)

    def average_weights(self):
        """
        Average weights of the perceptron

        Training can no longer be resumed.
        """
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for label, w in weights.items():
                param = (label, feat)
                # Be careful not to add 1 to take into account the
                # last weight vector (without increasing the number of
                # iterations in the averaging)
                total = self._accum[param] + \
                    (self.n_updates + 1 - self._last_update[param]) * w
                averaged = round(total / self.n_updates, 3)
                if averaged:
                    new_feat_weights[label] = averaged
            self.weights[feat] = new_feat_weights

    def __getstate__(self):
        """
        Serialization of a perceptron

        We are only serializing the weight vector as a dictionnary
        because defaultdict with lambda can not be serialized.
        """
        # should we also serialize the other attributes to allow
        # learning to continue?
        return {"weights": {k: v for k, v in self.weights.items()}}

    def __setstate__(self, data):
        """
        De-serialization of a perceptron
        """

        self.weights = defaultdict(lambda: defaultdict(float), data["weights"])
        # ensure we are no longer able to continue training
        self._accum = None
        self._last_update = None

#creating all labels for the loaded data
def all_labels(datasets):
    # Il existe un module "chain" qui contient la méthode "from_iterable"
    # qui transforme une liste de liste de liste etc en une simple liste c'est plus opti
    return set([label for dataset in datasets for words, labels in dataset for label in labels ])


#Generate a feature sparse vector (as a dictionnary) from a sentence, and a word position. 

#Only the word as feature
def features1(s,i):
    res = {}
    res['Word: {}'.format(s[i])] = 1
    return res

"""-------------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------------"""

#The word, with the 2 before and after him in the sentence
def features2(s,i):
    res = features1(s,i)
    
    if i>=2 and i< len(s)-2 :
        res['wi-1: {}'.format(s[i-1])] = 1
        res['wi-2: {}'.format(s[i-2])] = 1
        res['wi+1: {}'.format(s[i+1])] = 1
        res['wi+2: {}'.format(s[i+2])] = 1
    else :
        if i<1 :
            res['wi-1: NULL'] = 1
            res['wi-2: NULL'] = 1
        elif i<2 :
            res['wi-1: {}'.format(s[i-1])] = 1
            res['wi-2: NULL'] = 1
        else :
            res['wi-1: {}'.format(s[i-1])] = 1
            res['wi-2: {}'.format(s[i-2])] = 1
        if i>=len(s)-1 :
            res['wi+1: NULL'] = 1
            res['wi+2: NULL'] = 1
        elif i>=len(s)-2 :
            res['wi+1: {}'.format(s[i+1])] = 1
            res['wi+2: NULL'] = 1
        else :
            res['wi+1: {}'.format(s[i+1])] = 1
            res['wi+2: {}'.format(s[i+2])] = 1
    return res

"""-------------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------------"""

# function that returns the n most frequent words in a train set
def words_frequency_from_corpus (train, n):
    word_frequency = defaultdict(int)
    for s, label in train:
        for word in s :
            word_frequency.update({word : word_frequency[word]+1 })
    return [i[0] for i in sorted(word_frequency.items(), key=itemgetter(1), reverse=True)[:n]]

"""
Returns a dictionnary associating, for each word w in a train set, a list of (words ci, freq f) sorted on f
the couple ci, f is added in the list only if the word ci is in the n most frequent words calculated with the previous function
Returns also the dictionnary for the word, ci combinations
"""
def distrib_features_dict(train):
    # First we create the dictionnary ci, freq if ci belongs to the n most frequent words
    freq = words_frequency_from_corpus(train, 100)
    d_left = defaultdict(lambda:defaultdict(int))
    d_right = defaultdict(lambda:defaultdict(int))
    for s, label in train :
        for i in range(len(s)):
            if i == 0 and i < len(s)-1:
                if s[i+1] in freq:
                    d_right[s[i]][s[i+1]] += 1
            elif i == len(s)-1 :
                if s[i-1] in freq:
                    d_left[s[i]][s[i-1]] += 1
            else:
                if s[i-1] in freq:
                    d_left[s[i]][s[i-1]] += 1
                if s[i+1] in freq:
                    d_right[s[i]][s[i+1]] += 1

    # Then for each word we sort the lists associated and returns them
    d_left_list = defaultdict(lambda: list())
    d_right_list = defaultdict(lambda: list())
    for w, d in d_left.items():
        tmp = sorted(d.items(), key=itemgetter(1), reverse=True)
        d_left_list[w] = tmp
    for w, d in d_right.items():
        tmp = sorted(d.items(), key=itemgetter(1), reverse=True)
        d_right_list[w] = tmp
    return d_left_list, d_right_list

#The features described in "FLORS : Fast and Simple Domain Adaptation for PoS Tagging"
def features3(s,i, l_feat_list, r_feat_list):
    res = features2(s, i)

    #Left features
    for k in range(len(l_feat_list)):
        res["l_" + str(k) + "_" + l_feat_list[k][0]] = 1
    #Right features
    for k in range(len(r_feat_list)):
        res["r_" + str(k) + "_" + r_feat_list[k][0]] = 1
    
    #Shape features
    res['firstUpper: {}'.format(s[i][0].isupper())] = 1
    res['has_digit: {}'.format(bool(re.search(r'\d', s[i])))] = 1
    res['has_hyphen: {}'.format("-" in s[i])] = 1
    res['has_upper: {}'.format(any (c.isupper() for c in s[i]))] = 1
    res['allUpper: {}'.format(s[i].isupper())] = 1


    #Suffix features
    for c in range(len(s[i])):
        res['last'+ str(c+1) +'c: {}'.format(s[i][-(c+1):])] = 1

    return res

max_epoch = 10
all_labels = list(all_labels(full_datasets))

filename = "Results_features3.csv"
f = open(filename, 'w')

for train in train_datasets:
    #Training
    name_train = train[0]
    perceptron = Perceptron(all_labels)
    count = 0
    l_feat_list, r_feat_list = distrib_features_dict(train[1])
    for epoch in range(max_epoch):
        for words, labels in train[1]:
            for i in range(len(words)):
                #features = features1(words, i)
                #features = features2(words,i)
                features = features3(words,i, l_feat_list[words[i]], r_feat_list[words[i]])
                prediction = perceptron.predict(features)
                perceptron.update(labels[i],prediction,features)
                # Affichage pour vérifier que le perceptron tourne bien
                #count += 1
                #if count%10000==0:
                #    print(count)

    #Testing
    global_error = 0.0
    global_OOV_error = 0.0
    global_ambiguous_error = 0.0
    ambiguous = ambiguous_words(train[1])
    for entry in test_datasets:
        error = 0
        OOV_error = 0
        ambiguous_error = 0
        count_error = 0
        count_OOV = 0
        count_ambiguous = 0
        OOV = OOV_words(train[1], entry[1])
        name_test = entry[0]
        for words, labels in entry[1]:
            for i in range(len(words)):
                #prediction = perceptron.predict(features1(words,i))
                #prediction = perceptron.predict(features2(words,i))
                prediction = perceptron.predict(features3(words,i, l_feat_list[words[i]], r_feat_list[words[i]]))
                
                # computing error rates
                count_error += 1
                if labels[i] != prediction:
                    error += 1
                if words[i] in OOV:
                    count_OOV += 1
                    if labels[i] != prediction:
                        OOV_error += 1
                if words[i] in ambiguous:
                    count_ambiguous += 1
                    if labels[i] != prediction:
                        ambiguous_error += 1
                             
        global_error += error * 100 / count_error
        global_OOV_error += OOV_error * 100 / count_OOV
        global_ambiguous_error += ambiguous_error * 100 / count_ambiguous
        f.write(name_train + ";" + name_test + ";" + str(error * 100 / count_error) + ";" + str(OOV_error * 100 / count_OOV) + ";" + str(ambiguous_error * 100 / count_ambiguous) + ";\n")
        # Affichage intermédiaire pour vérifier que les résultats ne sont pas aberrants avant d'avoir tout calculé
        #print(name_test + " " + str(error * 100 / count_error)) 

    print("Global error : " + str(global_error / len(test_datasets)))
    f.write("global;;"+ str(global_error / len(test_datasets)) + ";" + str(global_OOV_error / len(test_datasets)) + ";" + str(global_ambiguous_error / len(test_datasets)))
    f.write("\n")
f.close()