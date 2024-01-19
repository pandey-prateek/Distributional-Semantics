import pandas as pd
from tqdm import tqdm
import json
from collections import defaultdict
import re
import numpy as np
from torchvision import transforms
import nltk
from nltk import word_tokenize
nltk.download('punkt')
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy import spatial
import random
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import accuracy_score
# Commented out IPython magic to ensure Python compatibility.
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# %matplotlib inline


data=pd.read_json('./reviews_Movies_and_TV.json',lines=True,chunksize=20000)
def clean(i):
    i=i.lower()
    i=re.sub("[']","", i)
    i=re.sub("[0-9!.,$()]"," ", i)
    if len(i)>0 and i[-1]==" ":
      i=i[:-1]
    return i
for chunk in tqdm(data):
    data=[word_tokenize(clean(i)) for i in chunk['reviewText']]
    break



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embedded = torch.mean(self.embedding(inputs), axis=1)
        output = self.linear1(embedded)
        return output


class Word2VecDataset(Dataset):
    def __init__(self, corpus, window_size, neg_samples):
        self.window_size = window_size
        self.neg_samples = neg_samples

        vocab = list(set([word for line in corpus for word in line]))
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}

        data = []
        for line in corpus:
            for i, word in enumerate(line):
                context_words = []
                for j in range(max(0, i - window_size), min(len(line), i + window_size + 1)):
                    if i != j:
                        context_words.append(line[j])
                if len(context_words) == 2 * window_size:
                    data.append((self.word2idx[word], [self.word2idx[w] for w in context_words]))
        self.data = data

        self.neg_distr = np.array([self.word2idx[word] for word in vocab])
        self.neg_distr = np.power(self.neg_distr, 0.75)
        self.neg_distr /= np.sum(self.neg_distr)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_word, context_words = self.data[idx]
        negative_samples = np.random.choice(len(self.neg_distr), size=self.neg_samples, replace=True, p=self.neg_distr)
        return input_word, context_words, negative_samples


class Word2Vec:
    def __init__(self, corpus, embedding_dim=300, window_size=4, neg_samples=8, batch_size=1024,saved=True):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.batch_size = batch_size

        self.dataset = Word2VecDataset(corpus, window_size, neg_samples)
        self.vocab_size = len(self.dataset.word2idx)
        
        self.cbow = CBOW(self.vocab_size, embedding_dim)
        if saved:
            self.cbow=self.cbow.load_state_dict(torch.load('model.pt'))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.cbow.parameters(), lr=1e-4)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, num_epochs=5):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        self.cbow.train()
        print()
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0
            y_true = []
            y_pred = []
            for batch_idx, (input_word, context_words, negative_samples) in enumerate(dataloader):
                self.optimizer.zero_grad()
                context_words = torch.stack(context_words)
                output = self.cbow(context_words.T)
                dot_products = [torch.dot(input_word.squeeze(), negative_samples.T[i].squeeze()) for i in range(len(negative_samples.T))]
                neg_loss = -sum([F.logsigmoid(-score.float()) for score in dot_products])/len(dot_products)
                
                # neg_output=self.cbow(negative_samples)
                # target_var = torch.zeros(neg_output.shape[0], dtype=torch.long)
                loss  = self.criterion(output, input_word) + neg_loss#self.criterion(-neg_output, target_var)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                y_true += input_word.cpu().tolist()
                y_pred += torch.argmax(output, axis=1).cpu().tolist()
            
            print(f"Epoch {epoch + 1}, loss={total_loss/len(dataloader):.2f}, accuracy={accuracy_score(y_true, y_pred):.2f}")


    def get_embedding(self, word):
        word_var = torch.LongTensor([self.dataset.word2idx[word]])
        embedding = self.cbow.embedding(word_var).detach().numpy()[0]
        return embedding





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
def get_embedding_top_n(word,n,indx,w_indx,word_embeddings):
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

w2v=Word2Vec(data,batch_size=1024)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
w2v.train()

embedding=[[1]]*(len(w2v.vocab))

for i in w2v.word2idx.keys():
    embedding[w2v.word2idx[i]]=w2v.get_embedding(i)

