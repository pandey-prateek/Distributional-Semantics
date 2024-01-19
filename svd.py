

import pandas as pd
from tqdm import tqdm
import json
from collections import defaultdict
import re
import numpy as np
import nltk
from nltk import word_tokenize
nltk.download('punkt')
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy import spatial
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

data=pd.read_json('./reviews_Movies_and_TV.json',lines=True,chunksize=50000)
def clean(i):
    i=i.lower()
    i=re.sub("[']","", i)
    i=re.sub("[0-9!.,$]"," ", i)
    if len(i)>0 and i[-1]==" ":
      i=i[:-1]
    return i
for chunk in tqdm(data):
    data=[word_tokenize(clean(i)) for i in chunk['reviewText']]
    break

vocab = defaultdict(int)
for line in tqdm(data):
    for word in line:
        vocab[word]+=1

indx={}
w_indx={}
for idx,key in enumerate(vocab.keys()):
    indx[key]=idx
    w_indx[idx]=key

len(vocab)

cooc_matrix = np.zeros(shape=(len(vocab),len(vocab)))

window_size=2
for line in tqdm(data):
  for i, word in enumerate(line):
      if word!="":
        for j in range(max(0, i - window_size), min(len(line), i + window_size + 1)):
            if i != j:
                cooc_matrix[indx[word]][indx[line[j]]] += 1

# svd = TruncatedSVD(n_components=256)
# word_embeddings = svd.fit_transform(cooc_matrix)
word_embeddings, S, Vt = svds(cooc_matrix, k=256)
word_embeddings = word_embeddings @ np.diag(np.sqrt(S))

word_embeddings.shape

def get_embedding_top_n(word,n,w_indx,word_embeddings):
  word_index=indx[word]
  embedding=word_embeddings[word_index]

  res = {}
  for i, embed in tqdm(enumerate(word_embeddings)):
      if i!=word_index:
          res[i] = 1 - spatial.distance.cosine(word_embeddings[i], embedding)
  dist=sorted(res.items(), key=lambda x:x[1],reverse=True)
  b=[w_indx[v[0]] for v in dist[:n]]
  a=[word_embeddings[v[0]] for v in dist[:n]]
  return b,a

word,embedding=get_embedding_top_n('the',10,w_indx,word_embeddings)





from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne_plot(words,embeds):
        tsne_model = TSNE(perplexity=1)
        res_embeds = tsne_model.fit_transform(embeds)

        x_axis_val = []
        y_axis_val = []
        for val in res_embeds:
            x_axis_val.append(val[0])
            y_axis_val.append(val[1])
            
        plt.figure(figsize=(10, 10)) 
        for i in range(len(x_axis_val)):
            plt.scatter(x_axis_val[i],y_axis_val[i])
            plt.annotate(words[i],
                        xy=(x_axis_val[i],y_axis_val[i]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
        plt.show()
