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
# print(X)
# exit(0)
# distortions = []
K = range(2,6)
for k in K:
	kmeanModel = KMeans(n_clusters=k).fit(X)
	kmeanModel.fit(X)
	label = kmeanModel.labels_
	sil_coeff = silhouette_score(X, label, metric='euclidean')
	print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, sil_coeff))

#################BEST K= 3##################################


kmeanModel = KMeans(n_clusters=3).fit(X)
kmeanModel.fit(X)

print("Top terms per cluster:")
order_centroids = kmeanModel.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(3):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
     


# Y = vectorizer.transform(["While the lockdown has been almost entirely lifted in most of France, the epidemic remains present. It is therefore very important that you remain alert and that you respect both the barrier measures and the remaining restrictions."])
# prediction = kmeanModel.predict(Y)
# print(prediction)

for i,art in enumerate(docs):
	Y = vectorizer.transform([art])
	prediction = kmeanModel.predict(Y)
	print("Article n# {} is in Cluster n# {}".format(i+1,prediction))

filename = 'finalized_model.sav'
pickle.dump(kmeanModel, open(filename, 'wb'))
	# distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# print(distortions)
# print(texts)
# exit(0)

# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(docs)


# print(X.shape)
# exit(0)
# K = range(1,10)
# for k in K:
# 	model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
# 	model.fit(X)
	# label = model.labels_
	# sil_coeff = silhouette_score(X, label, metric='euclidean')
	# print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()
# print("Top terms per cluster:")
# order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names()
# for i in range(true_k):
#     print("Cluster %d:" % i),
#     for ind in order_centroids[i, :10]:
#         print(' %s' % terms[ind]),
#     print

# print("\n")
# print("Prediction")

# string_test = Y = vectorizer.transform(["Two national surveys by the non-profit Angus Reid Institute show that Liberal voters are still committed to Prime Minister  Justin Trudeau in spite of the current scandal over the WE Charity contract and generally no improvement for Canada on the world stage in spite of strong claims by Trudeau of bringing Canada ‘back’ onto the world stage."])
# prediction = model.predict(Y)
# print(prediction)
