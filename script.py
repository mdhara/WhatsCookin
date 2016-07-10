import pandas
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk

import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


traindf = pandas.read_json('train.json')
traindf.to_csv("train.csv")
#lists comma separated ingredients without any other character
#[u'romaine lettuce', u'black olives', u'grape tomatoes', u'ga...] ->
#romaine lettuce,black olives,grape tomatoes,ga...
traindf['ingredients_clean_string'] = [','.join(z).strip() for z in traindf['ingredients']]
#romaine lettuce black olives grape tomatoes ga...
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]   
print(traindf)

testdf = pandas.read_json('test.json')
testdf.to_csv("test.csv")
testdf['ingredients_clean_string'] = [','.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']] 


corpustr = traindf['ingredients_string']
#vectorizertr = TfidfVectorizer(stop_words='english',
#                             ngram_range = ( 1 , 1 ),analyzer="word", 
#                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)

vectorizertr = CountVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+')
#size of resulting matrix: (39774, 2963) (2963 unique words >= 2 chars)
tfidftr=vectorizertr.fit_transform(corpustr).todense()
#np.set_printoptions(threshold=np.nan)
#print(tfidftr[0])
#print(tfidftr.shape)

#shlitting about 78/22% (total:39774)
X = tfidftr[0:31000,:]

y = traindf['cuisine'][0:31000]

Xt = tfidftr[31000:,:]

yt = traindf['cuisine'][31000:]



corpusts = testdf['ingredients_string']
#vectorizertr = TfidfVectorizer(stop_words='english',
#                             ngram_range = ( 1 , 1 ),analyzer="word", 
#                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)

#vectorizerts = CountVectorizer(stop_words='english',
#                             ngram_range = ( 1 , 1 ),analyzer="word", 
#                             max_df = .57 , binary=False , token_pattern=r'\w+')

#size of resulting matrix: (9944, 2963)
tfidfts=vectorizertr.transform(corpusts).todense()
Xval = tfidfts



#Implementing Linear SVM
#parameters = {'C':[0.1, 0.3, 0.5, 0.8]}
clf = svm.SVC(kernel='linear', decision_function_shape='ovo', C=0.3)

#clf = GridSearchCV(svm.SVC(kernel='linear', decision_function_shape='ovo'), parameters)
                       
model = clf.fit(X,y)
#print(clf.best_params_)
y_pred = clf.predict(X)
print(accuracy_score(y,y_pred))
yt_pred = clf.predict(Xt)
print(accuracy_score(yt,yt_pred))

yval = clf.predict(Xval)
print(Xval.shape)
print(yval.shape)
testdf['cuisine'] = yval
testdf = testdf.sort('id' , ascending=True)

testdf[['id' , 'ingredients_clean_string' , 'cuisine' ]].to_csv("submission.csv")


