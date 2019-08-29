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
import os

warnings.filterwarnings(action = 'ignore', category = DeprecationWarning)
PYTHONIOENCODING="UTF-8"  

MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
epochs = 2
batch_size = 50
final_stop_words = set(stopwords.words('english') + stopwords.words('french') + stopwords.words('spanish') + stopwords.words('german'))

# pre processing text data
def preprocess_text(text):
    for i in text:
        i = i.lower() # lowercase text
        i = ' '.join(word for word in i.split() if word not in final_stop_words) # remove stopwords from text
    return text

# applying preprocessing
def preprocess_apply(df_in, nameOfFeature):
    return df_in[nameOfFeature].apply(preprocess_text)


###########################################################################    

# Manage dataset

df = pd.read_csv('Final sample Gettech.csv', sep=';', index_col=0)
df = df.reset_index(drop=True)
df_test = df.groupby(['accommodation_id','basename','at','description','value_type_id', 'category'])['amenities_id'].apply(list)
df_test = df_test.to_frame().reset_index()
df_test2 = df.groupby(['accommodation_id','basename','at','description','value_type_id', 'category'])['amenities_cont'].apply(list)
df_test2 = df_test2.to_frame().reset_index()
final_df = pd.merge(df_test, df_test2[['accommodation_id', 'amenities_cont']], on="accommodation_id", how="left")
final_df.to_csv('processed_csv_file.csv', sep='\t', encoding='utf-8')

features = ['basename','description']
class_names = ['Hotel','Apartment']
df_t = final_df.copy()

feature_1 = ['basename']
X1 = preprocess_apply(df_t,feature_1)

feature_2 = ['description']
X2 = preprocess_apply(df_t,feature_2)

feature_3 = ['amenities_id']
X3 = df_t[feature_3]

Y1 = df_t['category']

data_df = pd.concat([X1, X2, X3, Y1], axis = 1)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(data_df['basename'].values) # tokenize text
#tokenizer.fit_on_texts(data_df['description'].values) # tokenize text
word_index = tokenizer.word_index
print('UNIQUE tokens-> %s' % len(word_index))

X = tokenizer.texts_to_sequences(data_df['basename'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(data_df['category']).values # create category tensor

###########################################################################

# Manage data representations

# split test train set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 50) # split text into train/test ratio

# LSTM model creation

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model_hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.3,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

acc = model.evaluate(X_test, Y_test)
print('Test set\n  Loss-> {:0.4f}\n  Accuracy-> {:0.4f}'.format(acc[0]*100,acc[1]*100))


###########################################################################


# Plot the training and testing accuracy
plt.title('LSTM Training/Testing Accuracy Measure')
plt.plot(model_hist.history['acc'], label='train')
plt.plot(model_hist.history['val_acc'], label='test')
plt.legend()
plt.savefig(os.path.join('lstm_accuracy.png'), dpi=300, format='png', bbox_inches='tight')   
#plt.show()

###########################################################################


# Test the trained model on novel input
new_hotel = ['Hello']
seq = tokenizer.texts_to_sequences(new_hotel)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = ['Hotel','Apartment']
print(labels[np.argmax(pred)])