import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import tornado.ioloop
from tornado.iostream import IOStream
from zmq.eventloop.ioloop import IOLoop
from bs4 import BeautifulSoup
import plotly.graph_objs as go
import plotly.plotly as py
import cufflinks
from IPython.core.interactiveshell import InteractiveShell
import plotly.figure_factory as ff
InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import xgboost, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import warnings
import sys
import datetime
import os
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from pandas import Series
import csv

warnings.filterwarnings(action = 'ignore', category = DeprecationWarning)
PYTHONIOENCODING="UTF-8"  

final_stop_words = set(stopwords.words('english') + stopwords.words('french') + stopwords.words('spanish') + stopwords.words('german'))

def classification_accuracy(classifier, X_train, Y_train, X_test, Y_test, is_neural_net=False):

    classifier.fit(X_train, Y_train) # fit training set on classifier
    
    predictions = classifier.predict(X_test) # predict category on test set

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, Y_test) # accuracy measure


def run_model(classifier, X_train, Y_train, X_test, Y_test, is_neural_net=False):

    classifier.fit(X_train, Y_train) # fit training set on classifier
    
    predictions = classifier.predict(X_test) # predict category on test set

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return predictions

def get_confusion_matrix(classifier, X_train, Y_train, X_test, Y_test, classes, cmap=plt.cm.Blues, is_neural_net=False):

    classifier.fit(X_train, Y_train) # fit training set on classifier
    
    predictions = classifier.predict(X_test) # predict category on test set
    
    print('Confusion Matrix')
    
    cm = confusion_matrix(Y_test, predictions)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    print(cm) # print confusion matrix

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# preprocessing text data
def preprocess_text(text):
    for i in text:
        i = i.lower() # lowercase text
        i = ' '.join(word for word in i.split() if word not in final_stop_words) # remove stopwords from text
    return text

# applying preprocessing
def preprocess_apply(df_in, nameOfFeature):
    return df_in[nameOfFeature].apply(preprocess_text)

##########################################################################    

# Manage dataset

df = pd.read_csv('Final sample Gettech.csv', sep=';', index_col=0)

df_test = df.groupby(['accommodation_id','basename','at','description','value_type_id', 'category'])['amenities_id'].apply(list)
df_test = df_test.to_frame().reset_index()
df_test2 = df.groupby(['accommodation_id','basename','at','description','value_type_id', 'category'])['amenities_cont'].apply(list)
df_test2 = df_test2.to_frame().reset_index()
final_df = pd.merge(df_test, df_test2[['accommodation_id', 'amenities_cont']], on="accommodation_id", how="left")
final_df.to_csv('processed_csv_file.csv', sep='\t', encoding='utf-8')

features = ['basename','description']
class_names = ['Apartment','Hotel']
df_t = final_df.copy()

feature_1 = ['basename']
X1 = preprocess_apply(df_t,feature_1)
feature_2 = ['description']
X2 = preprocess_apply(df_t,feature_2)
feature_3 = ['amenities_id']
X3 = df_t[feature_3]

Y1 = df_t['category']

data_df = pd.concat([X1, X2, Y1], axis = 1)

# change features in X here to provide as input to the model
X = data_df['description']

Y = data_df['category']

############################################################################

# Manage data representations

# split test train set
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.30, random_state=0)
encoder = preprocessing.LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_test = encoder.fit_transform(Y_test)

# create count vector
count_vector = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vector.fit(X)
X_train_count =  count_vector.transform(X_train)
X_test_count =  count_vector.transform(X_test)

# create word vector
word_tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100000)
word_tfidf.fit(X)
X_train_tfidf =  word_tfidf.transform(X_train)
X_test_tfidf =  word_tfidf.transform(X_test)

# create character vector
tfidf_ngram_char = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=100000)
tfidf_ngram_char.fit(X)
X_train_ngram_char =  tfidf_ngram_char.transform(X_train) 
X_test_ngram_char =  tfidf_ngram_char.transform(X_test) 

##########################################################################

print("------------------------------------------------------------------------------")

# Implementation of ML models on the data representations

# Xtreme Gradient Boosting

# XGB -> count vector
xgb_count = classification_accuracy(xgboost.XGBClassifier(), X_train_count.tocsc(), Y_train, X_test_count.tocsc(), Y_test)
print("1. XGB -> count vec -> {:0.4f}".format(xgb_count*100))

'''
code for running the model on test dataset

xgb_count_test_run = run_model(xgboost.XGBClassifier(), X_train_count.tocsc(), Y_train, X_test_count.tocsc(), Y_test)
xgb_count_test_run = encoder.inverse_transform(xgb_count_test_run)
for i in xgb_count_test_run:
    print (i)
'''

get_confusion_matrix(xgboost.XGBClassifier(), X_train_count.tocsc(), Y_train, X_test_count.tocsc(), Y_test, class_names)
plt.savefig(os.path.join('CM_xgb_count.png'), dpi=300, format='png', bbox_inches='tight')   
#plt.show()

print("------------------------------------------------------------------------------")

# XGB -> word tfidf
xgb_word = classification_accuracy(xgboost.XGBClassifier(), X_train_tfidf.tocsc(), Y_train, X_test_tfidf.tocsc(), Y_test)
print("2. XGB -> word tfidf -> {:0.4f}".format(xgb_word*100))

'''
code for running the model on test dataset

xgb_word_test_run = run_model(xgboost.XGBClassifier(), X_train_tfidf.tocsc(), Y_train, X_test_tfidf.tocsc(), Y_test)
xgb_word_test_run = encoder.inverse_transform(xgb_word_test_run)
for i in xgb_word_test_run:
    print (i)
'''

get_confusion_matrix(xgboost.XGBClassifier(), X_train_tfidf.tocsc(), Y_train, X_test_tfidf.tocsc(), Y_test, class_names)
plt.savefig(os.path.join('CM_xgb_word.png'), dpi=300, format='png', bbox_inches='tight')   
#plt.show()

print("------------------------------------------------------------------------------")

# XGB -> char tfidf
xgb_char = classification_accuracy(xgboost.XGBClassifier(), X_train_ngram_char.tocsc(), Y_train, X_test_ngram_char.tocsc(), Y_test)
print("3. XGB -> char tfidf -> {:0.4f}".format(xgb_char*100))

'''
code for running the model on test dataset

xgb_char_test_run = run_model(xgboost.XGBClassifier(), X_train_ngram_char.tocsc(), Y_train, X_test_ngram_char.tocsc(), Y_test)
xgb_char_test_run = encoder.inverse_transform(xgb_char_test_run)
for i in xgb_char_test_run:
    print (i)
'''

get_confusion_matrix(xgboost.XGBClassifier(), X_train_ngram_char.tocsc(), Y_train, X_test_ngram_char.tocsc(), Y_test, class_names)
plt.savefig(os.path.join('CM_xgb_char.png'), dpi=300, format='png', bbox_inches='tight')   
#plt.show()

print("------------------------------------------------------------------------------")

###########################################################################

# visualize and compare prediction results

# save accuracy metrics 
xgb_count = round(xgb_count*100, 4)
xgb_word = round(xgb_word*100, 4)
xgb_char = round(xgb_char*100, 4)


# plot accuracy with labels
X_data = ['XGB_char_vector', 'XGB_word_vector', 'XGB_count_vector']
Y_data = [xgb_char, xgb_word, xgb_count]

fig, ax = plt.subplots()    
width = 0.3 
index = np.arange(len(Y_data))
ax.barh(index, Y_data, width, align='center', color='blue')
plt.title('Accuracy Measures for DESCRIPTION', fontsize=20)
plt.ylabel('Model', fontsize=15)
plt.xlabel('Accuracy', fontsize=15)
plt.xlim(80, 100)
ax.set_yticks(index + width/2)
ax.set_yticklabels(X_data, minor=False)
for i, v in enumerate(Y_data):
    ax.text(v + 0.15, i, str(v), color='black', fontsize=10)
plt.savefig(os.path.join('accuracy_comparison.png'), dpi=300, format='png', bbox_inches='tight')   
#plt.show()

###########################################################################

