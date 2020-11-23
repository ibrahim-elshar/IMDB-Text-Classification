# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 12:17:19 2019

"""

import os
import nltk
import pickle
from nltk.corpus import stopwords
import timeit
import matplotlib
import matplotlib.pyplot as plt

stopWords = set(stopwords.words('english'))
words = []

files = os.listdir('aclImdb/train/pos')
for file in files:
    with open('aclImdb/train/pos/{}'.format(file), 'r', encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        for token in review:
            if token not in stopWords:
                words.append(token)

files = os.listdir('aclImdb/train/neg')
for file in files:
    with open('aclImdb/train/neg/{}'.format(file), 'r', encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        for token in review:
            if token not in stopWords:
                words.append(token)

words = nltk.FreqDist(words)
words_sorted_by_freq = sorted(words.items(), key=lambda kv: kv[1])
words_sorted_by_freq.reverse()
words_ls_sorted = [i[0] for i in words_sorted_by_freq]
def find_features(doc):
    ''' Takes a tokenized review and returns a dictionary of the feature vector
        with each has a value of True if that feature_word in inside the review. '''
    words = set(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

results_list = []
start_time = timeit.default_timer()
for num_keywords in range(100,10001,100):
    word_features = words_ls_sorted[:num_keywords]#list(words.keys())[:100] # maybe we should shuffle the keys

    feature_sets = []
    n = 12500 
    files = os.listdir('aclImdb/train/pos')[:n]
    for file in files:
        with open('aclImdb/train/pos/{}'.format(file), 'r', encoding = 'utf-8') as f:
            review = nltk.word_tokenize(f.read())
            feature_sets.append((find_features(review), 'pos'))
    print('Finished aclImdb/train/pos ')
    
    files = os.listdir('aclImdb/train/neg')[:n]
    for file in files:
        with open('aclImdb/train/neg/{}'.format(file), 'r', encoding = 'utf-8') as f:
            review = nltk.word_tokenize(f.read())
            feature_sets.append((find_features(review), 'neg'))
    print('Finished aclImdb/train/neg ')
    
    files = os.listdir('aclImdb/test/pos')[:n]
    for file in files:
        with open('aclImdb/test/pos/{}'.format(file), 'r', encoding = 'utf-8') as f:
            review = nltk.word_tokenize(f.read())
            feature_sets.append((find_features(review), 'pos'))
    print('Finished aclImdb/test/pos ')
    
    files = os.listdir('aclImdb/test/neg')[:n]
    for file in files:
        with open('aclImdb/test/neg/{}'.format(file), 'r', encoding = 'utf-8') as f:
            review = nltk.word_tokenize(f.read())
            feature_sets.append((find_features(review), 'neg'))
    print('Finished aclImdb/test/neg ')
    
    training_set = feature_sets[:2*n]
    test_set = feature_sets[2*n:]
    
    clf = nltk.NaiveBayesClassifier.train(training_set)
    result = nltk.classify.accuracy(clf, test_set)*100
    print('Accuracy of the Naive Bayes classification model for number of keywords ' + str(num_keywords)+': '+ str(result))
    results_list.append(result)

elapsed_time = timeit.default_timer() - start_time
print('elapsed_time:',elapsed_time)
with open("NB_results_100_10000.pkl", 'wb') as f:  
     pickle.dump([results_list, elapsed_time], f,protocol=2)
     
results_list, elapsed_time = pickle.load( open( "NB_results_100_10000.pkl", "rb" ) )

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 50}
x_axis = list(range(100,10001,100))
matplotlib.rc('font', **font)
plt.plot(x_axis,results_list, label='Naive Bayes') 
plt.xlabel('Number of keywords') 
plt.ylabel('Accuracy') 
plt.title('Classification Accuracy') 
plt.legend(loc='best')     
