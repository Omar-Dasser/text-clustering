import numpy as np
import pickle
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.corpus import stopwords
import gensim
import pandas as pd 

# df = pd.read_excel('news.xlsx')
# filename = 'finalized_model.sav'


df = pd.read_excel('news.xlsx')


docs = df['Description']

tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))


texts = []
for i in docs:
	raw = i
	tokens = tokenizer.tokenize(raw)

	# remove stop words from tokens
	stopped_tokens = [i for i in tokens if (not i in en_stop and not str(i).isdigit() and len(str(i)) > 2 )]

    # stem tokens
    #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
	texts.append(stopped_tokens)
    # add tokens to list

df['cleaned'] = [" ".join(review) for review in texts]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.cleaned)

kmeanModel = KMeans(n_clusters=3).fit(X)
kmeanModel.fit(X)




with open('Text_to_pred.txt','r') as f:
	text = f.read()
	# df['description'] = [text]
	# docs = list(df['Description'])
	print(text)
	# exit(0)

	X = vectorizer.transform([text])
	Y = kmeanModel.predict(X)
	print("Article is in cluster n# {}".format(Y))


