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

#Loading data

#creating all labels for the loaded data
def all_labels(datasets):
    # Il existe un module "chain" qui contient la méthode "from_iterable"
    # qui transforme une liste de liste de liste etc en une simple liste c'est plus opti
    return set([label for dataset in datasets for words, labels in dataset for label in labels ])

#label_set = all_labels([train_set, test_set, dev, foot_set, minecraft_set])

#Generate a feature sparse vector (as a dictionnary) from a sentence, and a word position. 
def feature_from_word(s,i):

    # print("---------------------------")
    # print(i)
    # print(s)
    # print(len(s))
    # print("---------------------------")
    res = {}
    res['last3c: {}'.format(s[i][-3:])] = 1
    res['last: {}'.format(s[i][-1])] = 1
    res['first: {}'.format(s[i][0])] = 1
    res['Word: {}'.format(s[i])] = 1
    res['firstUpper: {}'.format(s[i][0].isupper())] = 1
    res['allUpper: {}'.format(s[i].isupper())] = 1
    if i>=2 and i< len(s)-2 :
        res['wi-1: {}'.format(s[i-1])] = 1
        res['wi-2: {}'.format(s[i-2])] = 1
        res['wi+1: {}'.format(s[i+1])] = 1
        res['wi+2: {}'.format(s[i+2])] = 1
    else :
        if i<1 :
            res['wi-1: '] = 1
            res['wi-2: '] = 1
        elif i<2 :
            res['wi-1: {}'.format(s[i-1])] = 1
            res['wi-2: '] = 1
        else :
            res['wi-1: {}'.format(s[i-1])] = 1
            res['wi-2: {}'.format(s[i-2])] = 1
        if i>=len(s)-1 :
            res['wi+1: '] = 1
            res['wi+2: '] = 1
        elif i>=len(s)-2 :
            res['wi+1: {}'.format(s[i+1])] = 1
            res['wi+2: '] = 1
        else :
            res['wi+1: {}'.format(s[i+1])] = 1
            res['wi+2: {}'.format(s[i+2])] = 1
    return res

#perceptron = Perceptron(list(label_set))


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
    res['Word: {}'.format(s[i])] = 1
    if i>=2 and i< len(s)-2 :
        res['wi-1: {}'.format(s[i-1])] = 1
        res['wi-2: {}'.format(s[i-2])] = 1
        res['wi+1: {}'.format(s[i+1])] = 1
        res['wi+2: {}'.format(s[i+2])] = 1
    else :
        if i<1 :
            res['wi-1: '] = 1
            res['wi-2: '] = 1
        elif i<2 :
            res['wi-1: {}'.format(s[i-1])] = 1
            res['wi-2: '] = 1
        else :
            res['wi-1: {}'.format(s[i-1])] = 1
            res['wi-2: {}'.format(s[i-2])] = 1
        if i>=len(s)-1 :
            res['wi+1: '] = 1
            res['wi+2: '] = 1
        elif i>=len(s)-2 :
            res['wi+1: {}'.format(s[i+1])] = 1
            res['wi+2: '] = 1
        else :
            res['wi+1: {}'.format(s[i+1])] = 1
            res['wi+2: {}'.format(s[i+2])] = 1
    return res

"""-------------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------------"""

def words_frequency_from_corpus (train, test, n):
    word_frequency = defaultdict(int)
    for s, label in train:
        for word in s :
            word_frequency.update({word : word_frequency[word]+1 })
    for s, label in test:
        for word in s :
            word_frequency.update({word : word_frequency[word]+1 })
    return [i[0] for i in sorted(word_frequency.items(), key=itemgetter(1), reverse=True)[:n]]

def distrib_features_dict(train,test):
    freq = words_frequency_from_corpus(train,test, 10)
    d_left = defaultdict(lambda:defaultdict(int))
    d_right = defaultdict(lambda:defaultdict(int))
    for s, label in train :
        for i in range(len(s)):
            #word = s[i]
            #right_word = s[i+1]
            #left_word = s[i-1]
            if i == 0 and i < len(s)-1:
                if s[i+1] in freq:
                    d_right[s[i]][s[i+1]] += 1
            elif i == len(s)-1 :
                if s[i-1] in freq:
                    d_left[s[i]][s[i-1]] += 1
                else :
                    d_left[s[i]]["a trash"] += 1
            else:
                if s[i-1] in freq:
                    d_left[s[i]][s[i-1]] += 1
                else :
                    d_left[s[i]]["a trash"] += 1
                if s[i+1] in freq:
                    d_right[s[i]][s[i+1]] += 1
    for s, label in test :
        for i in range(len(s)):
            #word = s[i]
            #right_word = s[i+1]
            #left_word = s[i-1]
            if i == 0 and i < len(s)-1:
                if s[i+1] in freq:
                    d_right[s[i]][s[i+1]] += 1
            elif i == len(s)-1 :
                if s[i-1] in freq:
                    d_left[s[i]][s[i-1]] += 1
                else :
                    d_left[s[i]]["a trash"] += 1
            else:
                if s[i-1] in freq:
                    d_left[s[i]][s[i-1]] += 1
                else :
                    d_left[s[i]]["a trash"] += 1
                if s[i+1] in freq:
                    d_right[s[i]][s[i+1]] += 1

    return (d_left,d_right)


#The features described in "FLORS : Fast and Simple Domain Adaptation for PoS Tagging"
def features3(s,i, l_features, r_features):
    res = {}

    #Left features
    for ci, count in l_features[s[i]].items():
        res["l_" + ci + "_" + s[i]] = 1 + math.log(count)
    #Right features
    for ci, count in r_features[s[i]].items():
        res["r_" + s[i] + "_" + ci] = 1 + math.log(count)
    
    #Shape features

    #Suffix features
    for c in range(len(s[i])):
        res['last'+ str(c+1) +'c: {}'.format(s[i][-(c+1):])] = 1

    #shape features
    return res

train = train_datasets[0][1][:1000]
test = test_datasets[0][1][:1000]
l_feat, r_feat = distrib_features_dict(train,test)



for ci, count in features3(train[1][0], 5, l_feat, r_feat).items():
    print(ci + " = " + str(count))
print("-------------")

"""
#Make a dictionnary that associate every word & character to a unique identifier
def make_dicts(datasets):
    words_dict, chars_dict = {},{}
    for dataset in dataset :
        for words, labels in dataset :
            for w in words :
                if not w in words_dict :
                    words_dict[w] = len(words_dict)
                for c in w :
                    if not c in chars_dict :
                        chars_dict[c] = len(chars_dict)
    return words_dict, chars_dict
"""

#Training
"""count = 0
for epoch in range(10):
    for words, labels in train_set:
        for i in range(len(words)):
            features = feature_from_word(words,i)
            prediction = perceptron.predict(features)
            #print(prediction)
            perceptron.update(labels[i],prediction,features)
            count += 1
            if count%1000==0:
                print(count)"""

#testing
"""error = 0
total_test = 0
for entry in foot_set:
    for i in range (len(entry[0])):
        error += (perceptron.predict(feature_from_word(entry[0],i)) != entry[1][i])
        total_test += 1

print(error*100/total_test)"""

"""
Idée pour  évaluer le perceptron :
Train normal puis executer sur tous le test set
Renvoyer le texte de test avec les labels originaux + les labels prédit
Cela permet d'évaluer directement sa précision global
On peut ensuite garder uniquement les mots OOV ou ambigus pour tester sa précision
sur ces mots spécifiquement
"""