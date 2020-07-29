import os
from flask import Flask, render_template, request, jsonify
import json
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm


import pandas as pd
#import csv
from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.corpus import stopwords
import gensim

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import pickle

UPLOAD_FOLDER = '/static/uploads/'
ALLOWED_EXTENSIONS = set(['xlsx','xls','xlt'])

def prepare_train():
  df = pd.read_excel('news.xlsx')
  docs = df['Description']
  tokenizer = RegexpTokenizer(r'\w+')
  en_stop = set(stopwords.words('english'))
  texts = []
  for i in docs:
    raw = i
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if (not i in en_stop and not str(i).isdigit() and len(str(i)) > 2 )]
    texts.append(stopped_tokens)
  df['cleaned'] = [" ".join(review) for review in texts]
  return df

df_t = prepare_train()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_t.cleaned)

kmeanModel = KMeans(n_clusters=3).fit(X)
kmeanModel.fit(X)

def process_exl(filename):
  df = pd.read_excel(filename)
  docs = df['Description']
  tokenizer = RegexpTokenizer(r'\w+')
  en_stop = set(stopwords.words('english'))
  texts = []
  for i in docs:
    raw = i
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if (not i in en_stop and not str(i).isdigit() and len(str(i)) > 2 )]
    texts.append(stopped_tokens)
  df['cleaned'] = [" ".join(review) for review in texts]
  X = vectorizer.transform(df.cleaned)
  Y = kmeanModel.predict(X)

  return Y






app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        
        if 'cin_front' not in request.files:
         
          return render_template('upload.html', msg='No file selected')
        news = request.files['cin_front']
        if news and allowed_file(news.filename):
        
          exl_path =os.path.join(os.getcwd() + UPLOAD_FOLDER, news.filename) 
          news.save(exl_path)

          msg=process_exl(exl_path)
    
          return render_template('upload.html',
                                   msg=msg,passed=True,len=len(msg))
                              
    elif request.method == 'GET':
        return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
