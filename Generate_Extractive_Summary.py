# -*- coding: utf-8 -*-
# importing required libraries
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os, sys, string
import numpy as np
import pandas as pd
import nltk
import re

nltk.download('punkt')
## Read required libraries & utilities
from pickle import dump,load
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize

## Set up working directory
import os
workdir = os.getcwd()
print("The working directory is :", workdir)

NO_SAMPLES_FEED = 500 # number of samples from the original data.

def load_data_drive(fptr):
    with open(fptr, encoding = 'utf-8') as f:
        text = f.read()
        f.close()
    return text

##  function to extract news story and highlights
def dump_story_feed(document_index):
    idx = document_index.find('@highlight')
    story_part, highlight_part = document_index[:idx], document_index[idx:].split('@highlight')
    highlight_part = [h.strip() for h in highlight_part if len(h)>0]
    return story_part, highlight_part

## function to load all stories from data directory
def load_data(dir):
    stories = list()
    doc_iter = 0
    for files in os.listdir(dir):
        doc_iter += 1
        print("processing document_index number {}".format(doc_iter))
        fptr = os.path.join(dir, files)
        document_index = load_data_drive(fptr)
        story_part, highlight_part = dump_story_feed(document_index)
        if story_part is not None:
            stories.append({'story': story_part, 'highlight' : highlight_part})
            
        if doc_iter == NO_SAMPLES_FEED:
            break
    return stories

from google.colab import drive
drive.mount('/content/drive')

path_to_zip_file = '/content/drive/My Drive/dataset.zip'
directory_to_extract_to = '/content/sample_data'

import zipfile
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

#dir = os.path.join(workdir, 'Data', 'stories_text_summarization_dataset_train') 
dir = '/content/sample_data/dataset/stories_text_summarization_dataset_train'
stories = load_data(dir)
print('Loaded Stories %d' % len(stories))

# Reading in the training data
import os

X_text = []
counter = 0
file_list = os.listdir('/content/sample_data/dataset/stories_text_summarization_dataset_train')
for fptr in file_list:
  counter+=1
  with open(os.path.join('/content/sample_data/dataset/stories_text_summarization_dataset_train',fptr), 'r', encoding='utf-8') as f:
    content=f.read()
    X_text.append(content)
    f.close()
  if counter % 10000 == 0:
    print(counter)

"""
- Graph based algorithm

Basic steps
- Cleaning Text (remove punctuation, Stopwords, stemming)
- Vector representation of sentences: **This part can be customized by using different pre-trained vectorization models or train your own model**
- Use cosine similarity find similarity of sentences
- Apply PageRank algorithm: use networkx(networkx.score) to rank sentences
- Extract top N sentences as summary

Skip implementation, there are >3 existing packages using graph

### 4.1 Gensim summarizer
https://github.com/RaRe-Technologies/gensim/tree/develop/gensim/summarization

### 4.2 Pytextrank package
https://github.com/DerwenAI/pytextrank/blob/master/explain_summ.ipynb
"""
def remove_punctuations(sent):
  """ To remove punctuations and special characters """
  cleaned=re.sub(r'[?|!|\'|"|#]', r'', sent)
  cleaned=re.sub(r'[.|,|)|(|\|/]', r' ', sent)
  cleaned=cleaned.strip()
  cleaned=cleaned.replace('\n',' ')
  return cleaned
 
def remove_non_alphanumeric(sent):
  """ To remove non-alphabetic characters """
  alpha_sent=''
  for word in sent.split():
    alpha_word=re.sub('[^a-z A-Z]+', ' ', word)
    alpha_sent+=alpha_word
    alpha_sent+=' '
  alpha_sent=alpha_sent.strip()
  return alpha_sent

X_data = [remove_punctuations(sent) for sent in X_text]
X_data = [remove_non_alphanumeric(sent) for sent in X_data]
X_data = [sent.lower() for sent in X_data]

nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(sent):
  sent_new = ' '.join([i for i in sent if i not in stop_words])
  return sent_new

X_data = [remove_stopwords(r.split()) for r in X_data]

print('No. of sentences',len(X_data))
print('First 5 sentences',X_data[:5])

## Splitting sentences into words for building word embeddings
X_sent_words = []
for sent in X_data:
  sent_word_list = [word for word in sent.split()]
  X_sent_words.append(sent_word_list)
print(X_sent_words[:5])

## training word2vec model to create word embeddings
model = Word2Vec(X_sent_words, min_count=1)

## saving the model
model.wv.save_word2vec_format('word_embeddings.bin')

# Generating Summary for a test files
X_test = []
counter = 0
file_list = os.listdir('/content/sample_data/dataset/stories_text_summarization_dataset_test')
for fptr in file_list:
  counter+=1
  with open(os.path.join('/content/sample_data/dataset/stories_text_summarization_dataset_test',fptr), 'r', encoding='utf-8') as f:
    content=f.read()
    X_test.append(content)
    f.close()
  print(counter)

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def generate_summary(content, num_sentences):
  # splitting into sentences
  sentences = sent_tokenize(content)
  
  # cleaning sentences as above
  sentence_token = [remove_punctuations(sent) for sent in sentences]
  sentence_token = [remove_non_alphanumeric(sent) for sent in sentence_token]
  sentence_token = [sent.lower() for sent in sentence_token]
  
  # removing stopwords
  sentence_token = [remove_stopwords(r.split()) for r in sentence_token]
    
  # making sentence vectors
  sentence_vectors = []
  for i in sentence_token:
    if len(i)!=0:
      v = sum([model[w] for w in i.split()])/(len(i.split())+0.001)
    else:
      v = np.zeros((100,))
    sentence_vectors.append(v)
  
  # similarity matrix
  sim_mat = np.zeros([len(sentences), len(sentences)])
  
  for i in range(len(sentences)):
    for j in range(len(sentences)):
      if i!=j:
        sim_mat[i][j]=cosine_similarity(sentence_vectors[i].reshape(1,100),
                                       sentence_vectors[j].reshape(1,100))[0,0]
  
  # building graph for ranking
  nx_graph = nx.from_numpy_array(sim_mat)
  scores = nx.pagerank(nx_graph)
  
  ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
  
  # Generate summary
  summary=[]
  for i in range(num_sentences):
    summary.append(ranked_sentences[i][1])
  
  return summary

print(X_test[0])
print('')
print('Summary')

summary = generate_summary(X_test[0],5)
for sent in summary:
  print(sent)
