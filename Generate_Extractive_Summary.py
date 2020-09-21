# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

## Setting up working directory
import os
work_dir = os.getcwd()
print("The working directory is :", work_dir)

## Importing the necessary modules in current namespace
import os, sys, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf



# Commented out IPython magic to ensure Python compatibility.
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout, Concatenate, TimeDistributed, Add, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints # Importng Layer Weight Regularizers
from pickle import dump,load
from tensorflow.keras import layers
from tensorflow.keras import regularizers

## Layer weight regularizers
## Regularizers allow you to apply penalties on layer parameters or layer activity during optimization. These penalties are summed into the loss function that the network optimizes.
"""
kernel_regularizer: Regularizer to apply a penalty on the layer's kernel
bias_regularizer: Regularizer to apply a penalty on the layer's bias
activity_regularizer: Regularizer to apply a penalty on the layer's output
"""
layer = layers.Dense(
    units=64,
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)
)

# %load_ext tensorboard
import datetime, os

# Mouting Google Drive to load data
from google.colab import drive
drive.mount('/content/drive')

## Variables initialization
NO_SAMP       = 10000 # number of samples from the original data.
SIZE_VOCAB    = 50000 # max vocabulary size
DIM_EMBEDDED  = 300   # embedding dimension from the embedding matrix glove
DIM_LATENT    = 128   # latent dimentionality of the encoding space

SIZE_BATCH    = 64    # batch size for each training pass
NO_OF_EPOCS   = 50    # number of epochs to be trained
VALIDATION_SPLIT_RATIO = 0.2 # propertion of the validation sample

text_container         = []  # placeholder to save in input text (Story)
target_container       = []  # placeholder to save in target text (Summary)
tt_container = []  # placeholder to save in target text offset by 1 (Summary)
path_to_zip_file = '/content/drive/My Drive/dataset.zip'
directory_to_extract_to = '/content/sample_data'

## The zipfile module does not support ZIP files with appended comments, or multi-disk ZIP files. It does support ZIP files larger than 4 GB that use the ZIP64 extensions.
import zipfile
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

## Function to load a single document given a filename. The data has some unicode characters, so we will load the dataset by forcing the encoding to be UTF-8.
## The function below named read_documents() will load a single document as text given a filename.

def read_documents(filename):
    '''
    Function to read the document using a loop
    input: file name
    output: text
    '''
    with open(filename, encoding = 'utf-8') as f:
        text = f.read()
        f.close()
    return text

## The split for these two points is the first occurrence of the ‘@highlight‘ token. Once split, we can organize the highlights into a list.
## The function below named extract_news_story() implements this behavior and splits a given loaded document text into a story and list of highlights.

def extract_news_story(doc_pointer):
    '''
    Function to extract the news story and highlights from the input file
    input : document 
    output: story , highlights 
    '''
    idx = doc_pointer.find('@highlight')
    stry_p, h_p = doc_pointer[:idx], doc_pointer[idx:].split('@highlight')
    h_p = [h.strip() for h in h_p if len(h)>0]
    return stry_p, h_p

## We can now update the loadStory() function to call the extract_news_story() function for each loaded document and then store the results in a list.

def loadStory(o_dir):
    '''
    function to load all num_stories from data directory
    '''
    num_stories = list()
    iter_doc = 0
    for files in os.listdir(o_dir):
        iter_doc += 1
        print("processing doc_pointer number {}".format(iter_doc))
        filename = os.path.join(o_dir, files)
        doc_pointer = read_documents(filename)
        stry_p, h_p = extract_news_story(doc_pointer)
        if stry_p is not None:
            num_stories.append({'story': stry_p, 'highlight' : h_p})
            
        if iter_doc == NO_SAMP:
            break
    return num_stories

# load num_stories
# /content/sample_data/dataset (Have created a placeholder to keep the data feeds)
o_dir = os.path.join(work_dir, 'sample_data/dataset', 'stories_text_summarization_dataset_train') 
num_stories = loadStory(o_dir)
print('Loaded Stories %d' % len(num_stories))
print(num_stories[0])
print(num_stories[1])

print(num_stories[0])

## Data Cleaning
## function named data_preprocessing() that takes a list of lines of text and returns a list of clean lines of text.
def data_preprocessing(lines):
    cln_txt = list()
    punctuation_table = str.maketrans('', '', string.punctuation)
    for line in lines:
        

        idx = line.find('(CNN) -- ')
        if idx > -1:
            line = line[idx+len('(CNN)'):]
        line = line.split()
        line = [word.lower() for word in line]
        line = [w.translate(punctuation_table) for w in line]
        line = [word for word in line if word.isalpha()]
        cln_txt.append(' '.join(line))
        
        ## remove empty strings
        cln_txt = [c for c in cln_txt if len(c) > 0]
    return cln_txt

# clean num_stories
stories_cleaned = list()
for exp in num_stories:
    temp_cleaned_s = str()
    temp_cleaned_h = str()
    
    exp['story'] = data_preprocessing(exp['story'].split('\n'))
    exp['highlight'] = data_preprocessing(exp['highlight'])
   
    temp_cleaned_s = ' '.join([str(n) for n in exp['story']]) 
    temp_cleaned_h = ' '.join([str(n) for n in exp['highlight']]) 
    
    stories_cleaned.append({'story': temp_cleaned_s , 'highlight' : temp_cleaned_h})

print(stories_cleaned[0])

# Dump the cleaned data 
dump(stories_cleaned, open(os.path.join(work_dir, "/content/sample_data", 'dataset_convolution_nn.pkl') , 'wb'))

load_story = load(open(os.path.join(work_dir, "/content/sample_data", 'dataset_convolution_nn.pkl'), 'rb'))
print('Loaded Stories %d' % len(load_story))

load_story[0]['story']

# Create seperate placeholder for stories and Summary
for i, val in enumerate(load_story):
    if i == NO_SAMP:
        break
    text_container.append(str(val["story"]))
    target_container.append(str(val["highlight"]) + " <eos>")
    tt_container.append("<sos> " + str(val["highlight"]) )

print(text_container[0])
print(" ")
print(target_container[0])
print("")
print(tt_container[0])

print(text_container[0])

"""## 1. Opinosis Dataset
 - Contain **51 paragraphs** of user reviews on a given topic, obtained from Tripdvisor, Edmunds and Amazon
 - Each paragraph contains 100 sentence in average
 - Data file also contains **gold standard summaries** of each paragraph for test and validation
 
 https://kavita-ganesan.com/opinosis-opinion-dataset/#.Xe6Ya-hKhPY
 
 https://github.com/kavgan/opinosis-summarization
"""

df = pd.read_csv('opinosis.csv')

df.columns

## Clean the paragraph
def rm_rrn(string):
    if isinstance(string, str):
        return string.replace('\r\r\n',' ')

df = df.applymap(rm_rrn)

"""#### An example paragraph: text1"""

text1 = df['text'][0]
text1

"""#### Human written summary of text1"""

df['summary_number_1'][0]

!pip install rouge

"""## 2. Rouge Score for text summarization

#### Metric: Rouge score (Recall-Oriented Understudy for Gisting Evaluation)
     - Rouge N: measure N-gram overlap between model output summary and reference summary
     - Rouge L: measures longest matching sequence of words using LCS(longest Common Subsequence)
Rouge score is composed of: 
    - Precision = # of overlapping words / total words in the reference summary
    - Recall = # of overlapping words / total workds in the model generated summary
    - F1-score
Interpretation:
    - ROUGE-n recall=40% : 40% of the n-grams in the reference summary are also present in the generated summary.
    - ROUGE-n precision=40% : 40% of the n-grams in the generated summary are also present in the reference summary.
    - ROUGE-n F1-score=40% is like any F1-score.
"""

from rouge import Rouge
rouge = Rouge()

"""**lets see the two gold summaries Rouge score**

1-gram F1-score: 0.378
"""

rouge.get_scores(df['summary_number_2'][0], df['summary_number_1'][0])

"""## 3. Word Frequency Algorithm
Bag of words based algorithm
 - compute word frequency
 - score each sentence according to word frequency (can be weighted)
 - generate threshold of sentence selection (average score, etc.)
 - Selected sentence (score > threshold) as summary
"""

import word_frequency_model as wf
import nltk
nltk.download('stopwords')
nltk.download('punkt')

"""#### summary of word frequency model on text1
1-gram F1-score: 0.156
"""

wf_summary1 = wf.summarize_text_wf(text1) ## summary output from 
wf_summary1

rouge.get_scores(wf_summary1, df['summary_number_1'][0])

"""## 4. TextRank Algorithms
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

import textrank_graph_model as tr

tr_summary1 = tr.gensim_summarize(text1) # summary output fro gensim package
tr_summary1

"""**summary of text rank model on text1**

1-gram F1-score: 0.156
"""

rouge.get_scores(tr_summary1, df['summary_number_1'][0])

!pip install tensorflow-gpu==1.15.0
!pip install bert-tensorflow
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

! pip install pytorch_pretrained_bert

"""## 5. Kmean clustering of sentence embedding using Bert

### 5. 1 Impelemtation leveraging Bert Pretrained pytorch Model

https://pypi.org/project/pytorch-pretrained-bert/

#### Implementation

- Step 1: Tokenize paragraph into sentences
- Step 2: Format each sentence as Bert input format, and Use Bert tokenizer to tokenize each sentence into words
- Step 3: Call Bert pretrained model, conduct word embedding, obtain embeded word vector for each sentence.(The Bert output is a 12-layer latent vector) 
- Step 4: Decide how to use the 12-layer latent vector: 
    - 1) Use only the last layer; 
    - 2) Average all or last 4 layers, and more...
- Step 5: Apply pooling strategy to obtain sentence embedding from word embedding, eg. mean, max of all word vector
- Step 6: Obtain sentence vector for each sentence in the paragraph, apply Kmeans, Gaussian Mixture, etc to cluster similar sentence
- Step 7: Return the closest sentence to each centroid (euclidean distance) as the summary, ordered by appearance
"""

import bert_clustering_summary as bs

! pip install pytorch_pretrained_bert

bs_summary1 = bs.bertSummarize(text1)
bs_summary1

"""**Summary of BERT clustering model on text1**

1-gram F1-score: 0.155
"""

rouge.get_scores(bs_summary1, df['summary_number_1'][0])

! pip uninstall summarizer

! pip install Bert-extractive-summarizer

! pip install transformers

!pip install transformers
from transformers import BertModel, TFBertModel # no attribute 'TFBertModel'
! pip install tensorflow-gpu
from transformers import BertModel, TFBertModel # good to go

"""### 5.2 Another package using similar implementation
bert-extractive-summarizer

https://github.com/dmmiller612/bert-extractive-summarizer
"""

from summarizer import Summarizer

import logging
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s- %(message)s')
logging.debug('Start of program')

model = Summarizer()

bert_sum = model(text1)

from google.colab import drive
drive.mount('/content/drive')

"""**summary of bert-extractive-summarizer on text1**

1-gram F1-score: 0.149
"""

rouge.get_scores(bert_sum, df['summary_number_1'][0])

"""## 6. Test and Compare using Opinosis Dataset

### 6.1 Produce summary and compute ROUGE score from each model
"""

df.columns

df['wf_summary'] = df['text'].apply(lambda x: wf.summarize_text_wf(x))
df['wf_rouge1_f1'] = df[['wf_summary','summary_number_1']]. \
                apply(lambda x: rouge.get_scores(x[0],x[1])[0]['rouge-1']['f'],axis=1)

df['tr_summary'] = df['text'].apply(lambda x: tr.gensim_summarize(x))
df['tr_rouge1_f1'] = df[['tr_summary','summary_number_1']]. \
                apply(lambda x: rouge.get_scores(x[0],x[1])[0]['rouge-1']['f'],axis=1)

df['bs_summary'] = df['text'].apply(lambda x: bs.bertSummarize(x))
df['bs_rouge1_f1'] = df[['bs_summary','summary_number_1']]. \
                apply(lambda x: rouge.get_scores(x[0],x[1])[0]['rouge-1']['f'],axis=1)

np.mean(df['tr_rouge1_f1']),np.mean(df['wf_rouge1_f1']),np.mean(df['bs_rouge1_f1'])

df.to_csv('opniosis with summary.csv')

"""### 6.2 Plot and Compare"""

from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,4))

x = ['Word frequency model','Text rank model','BERT clustering model']
ROUGE1_mean = [0.10086302166961408, 0.09081258684152636, 0.11577391319962924]

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, ROUGE1_mean, color=['#E69F00','#56B4E9','#009E73'])


plt.xlabel('Rouge_1-gram_F1 score')
plt.ylabel('Mean')
plt.title("ROUGE-1 Score Mean")

plt.xticks(x_pos, x)

plt.figure(figsize=(10,4))

sns.distplot(df['wf_rouge1_f1'], hist=False, kde=True, color = '#E69F00', 
             kde_kws={'linewidth': 4},label='Word frequency model')

sns.distplot(df['tr_rouge1_f1'], hist=False, kde=True,color = '#56B4E9',
             kde_kws={'linewidth': 4},label='Text rank model')

sns.distplot(df['bs_rouge1_f1'], hist=False, kde=True,color = '#009E73',
             kde_kws={'linewidth': 4},label='BERT clustering model')

plt.legend(prop={'size': 10}, title = 'Models')

plt.xlabel('Rouge_1-gram_F1 score')
plt.ylabel('Density')
plt.title("ROUGE-1 Score Density")

"""### It seems the clustering summary using BERT embedding is slightly better than word frequency and text rank model summary!

#### Future work could be considered to 
- 1) Try out different BERT layers to produce the latent vectors (word embedding)
- 2) Try different pooling strategy from word vector to sentence vectors
- 3) Some other clustering method

#### Use supervise learning to fine tune BERT model for summarization purpose could be another topic to develop
https://arxiv.org/pdf/1903.10318.pdf
"""

## Set up working directory
import os
workdir = os.getcwd()
print("The working directory is :", workdir)

## Read required libraries & utilities
import os, sys, string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pickle import dump,load

import networkx as nx
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

NUM_SAMPLES      = 100 # number of samples from the original data.

##  function to read each document
def load_single_doc(filename):
    with open(filename, encoding = 'utf-8') as f:
        text = f.read()
        f.close()
    return text

##  function to extract news story and highlights
def extract_story(doc):
    idx = doc.find('@highlight')
    story_part, highlight_part = doc[:idx], doc[idx:].split('@highlight')
    highlight_part = [h.strip() for h in highlight_part if len(h)>0]
    return story_part, highlight_part

## function to load all stories from data directory
def load_all_stories(dir):
    stories = list()
    doc_iter = 0
    for files in os.listdir(dir):
        doc_iter += 1
        print("processing doc number {}".format(doc_iter))
        filename = os.path.join(dir, files)
        doc = load_single_doc(filename)
        story_part, highlight_part = extract_story(doc)
        if story_part is not None:
            stories.append({'story': story_part, 'highlight' : highlight_part})
            
        if doc_iter == NUM_SAMPLES:
            break
    return stories

path_to_zip_file = '/content/drive/My Drive/dataset.zip'
directory_to_extract_to = '/content/sample_data'

import zipfile
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

from google.colab import drive
drive.mount('/content/drive')

# load stories
#dir = os.path.join(workdir, 'Data', 'stories_text_summarization_dataset_train') 
dir = "/content/sample_data/dataset/stories_text_summarization_dataset_train"
stories = load_all_stories(dir)
print('Loaded Stories %d' % len(stories))

## 1.4.1 : Remove CNN office if exists
## 1.4.2 : tokenize on whitespace
## 1.4.3 : convert to lowercase
## 1.4.4 : remove punctuation chars from each tokens
## 1.4.5 : remove words that have non-alphabatic chars

def data_cleansing(lines):
    cleaned = list()
    punct_table = str.maketrans('', '', string.punctuation)
    for line in lines:
        
        ## 1.4.1
        idx = line.find('(CNN) -- ')
        if idx > -1:
            line = line[idx+len('(CNN)'):]
            
        ## 1.4.2
        line = line.split()
        
        ## 1.4.3
        line = [word.lower() for word in line]
        
        ## 1.4.4
        line = [w.translate(punct_table) for w in line]
        
        ## 1.4.5
        line = [word for word in line if word.isalpha()]
        
        ## store as string
        cleaned.append(' '.join(line))
        
        ## remove empty strings
        cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned

# clean stories
stories_cleaned = list()
for example in stories:
    cleaned_temp_s = str()
    cleaned_temp_h = str()
    
    example['story'] = data_cleansing(example['story'].split('\n'))
    example['highlight'] = data_cleansing(example['highlight'])
   
    ##cleaned_temp_s = ' '.join([str(n) for n in example['story']]) 
    ##cleaned_temp_h = ' '.join([str(n) for n in example['highlight']]) 
    
    ##stories_cleaned.append({'story': cleaned_temp_s , 'highlight' : cleaned_temp_h})
    stories_cleaned.append({'story': example['story'] , 'highlight' : example['highlight']})

### Separate story and summary into different list 
input_texts         = []  # placeholder to save in input text (Story)
target_texts        = []  # placeholder to save in target text (Summary)

for i, val in enumerate(stories_cleaned):
    if i == NUM_SAMPLES:
        break
    input_texts.append(val["story"])
    target_texts.append(val["highlight"])

print(input_texts[0])
print(" ")
print(target_texts[0])

##Plot the sequence distribution of stories
text_word_count = []
summary_word_count = []

# populate the lists with sentence lengths
for i in input_texts:
      text_word_count.append(len(str(i).split()))

for i in target_texts:
      summary_word_count.append(len(str(i).split()))

length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})
length_df.hist(bins = 30)
plt.show()

for i,v in enumerate(input_texts):
    print(len(str(v).split()))

##Select top N samples with specific sequence length threshold (longests)
def sub_sample(seq_len_threshold):
    idx = [i for i,v in enumerate(input_texts) if len(str(v).split()) >= seq_len_threshold] ## used greater than, I want few long sentences 
    input_texts_subsample = [input_texts[i] for i in idx]
    target_texts_subsample = [target_texts[i] for i in idx]
    return input_texts_subsample,target_texts_subsample
    
input_texts,target_texts = sub_sample(seq_len_threshold=1200)

print(len(input_texts))
print(len(target_texts))

## Create sentence_list of oly one story (for further demo)
sentences_list = input_texts[0]

print(sentences_list)
type(sentences_list)

cv = CountVectorizer()
cv_matrix = cv.fit_transform(sentences_list)

print('The data type of bow matrix {}'.format(type(cv_matrix)))
print('Shape of the matrix {}'.format(cv_matrix.get_shape))
print('Size of the matrix is: {}'.format(sys.getsizeof(cv_matrix)))
print(cv.get_feature_names())
print(cv_matrix.toarray())

## Get TF-IDF
normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
print(normal_matrix.toarray())

## Fraph represntation

res_graph = normal_matrix * normal_matrix.T

plt.spy(res_graph)

nx_graph = nx.from_scipy_sparse_matrix(res_graph)
nx.draw_circular(nx_graph)
print('Number of edges {}'.format(nx_graph.number_of_edges()))
print('Number of vertices {}'.format(nx_graph.number_of_nodes()))
plt.title("Graph of nodes and ages \n Note that the graph above is dense and therefor it resembles a circle. if a shorter document is taken, a beautiful circular graph can be seen")
# if a shorter document is taken, a beautiful circular graph can be seen")
plt.show()
print('The memory used by the graph in Bytes is: {}'.format(sys.getsizeof(nx_graph)))

## ranks is a dictionary with key=node(sentences) and value=textrank (the rank of each of the sentences)
ranks = nx.pagerank(nx_graph)

## analyse the data type of ranks
print(type(ranks))
print('The size used by the dictionary in Bytes is: {}'.format(sys.getsizeof(ranks)))

## print the dictionary
for i in ranks:
    print(i, ranks[i])

sentence_array = sorted(((ranks[i], s) for i, s in enumerate(sentences_list)), reverse=True)
sentence_array = np.asarray(sentence_array)

## as sentence_array is in descending order wrt score value the first value is with the largest score 
## and the last vaue is with the smallest score
rank_max = float(sentence_array[0][0])
rank_min = float(sentence_array[len(sentence_array) - 1][0])
print(rank_max)
print(rank_min)

## Normalize the score
temp_array = []


flag = 0
if rank_max - rank_min == 0:
    temp_array.append(0)
    flag = 1

## If the sentence has different ranks
if flag != 1:
    for i in range(0, len(sentence_array)):
        temp_array.append((float(sentence_array[i][0]) - rank_min) / (rank_max - rank_min))

print(len(temp_array))

## We take the mean value of normalized scores
## any sentence with the normalized score 0.2 more than the mean value is considered to be imporartance.
## this approach can be twicked, if needed
threshold = (sum(temp_array) / len(temp_array)) + 0.2

important_sentence_list = []
if len(temp_array) > 1:
    for i in range(0, len(temp_array)):
        if temp_array[i] > threshold:
                important_sentence_list.append(sentence_array[i][1])
else:
    important_sentence_list.append(sentence_array[0][1])

def get_summary(topn):
    summary = " ".join(str(x) for i,x in enumerate(important_sentence_list) if i <= topn)
    return summary

summary = get_summary(5)

print ("Original story :", sentences_list)
print("")
print("Generated Summary :", summary)

# Approach - 2
import numpy as np
import pandas as pd
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pickle

path = '/content/drive/My Drive/Embeddings/glove.840B.300d.pkl'

with open(path,'rb') as f:
  embeddings = pickle.load(f)

from nltk.stem import WordNetLemmatizer
import re
lem = WordNetLemmatizer()

def clean(sentence):
  sentence = sentence.lower()
  sentence = re.sub(r'http\S+',' ',sentence)
  sentence = re.sub(r'[^a-zA-Z]',' ',sentence)
  sentence = sentence.split()
  sentence = [lem.lemmatize(word) for word in sentence if word not in stopwords.words('english')]
  sentence = ' '.join(sentence)
  return sentence

def average_vector(sentence):
  words = sentence.split()
  size = len(words)
  average_vector = np.zeros((size,300))
  unknown_words=[]

  for index,word in enumerate(words):
    try:  
        average_vector[index] = embeddings[word].reshape(1,-1)
    except Exception as e:
      unknown_words.append(word)
      average_vector[index] = 0

  if size!=0:
    average_vector = sum(average_vector)/size
  return average_vector,unknown_words

def cosine_similarity(vector_1,vector_2):
  cosine_similarity = 0
  try:
    cosine_similarity = (np.dot(vector_1,vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2)))
  except Exception as e :
    # print("Exception occured",str(e))
    pass
  return cosine_similarity

def find_similarity(string1,string2):
  # string1,string2 = clean(string1),clean(string2)
  vector1,unknown_words1 = average_vector(string1)
  vector2,unknown_words2 = average_vector(string2)
  similarity = cosine_similarity(vector1,vector2)
  return similarity

from bs4 import BeautifulSoup
import requests

subject = input("Enter the wikipedia topic to be summarised")
base_url = "https://en.wikipedia.org/wiki/"+subject
page = requests.get(base_url)

soup = BeautifulSoup(page.content,'html.parser')
paragraphs = soup.find_all('p')

content=""
for paragraph in paragraphs:
    content+=paragraph.text

print(content)

content = sentences_list

content

sentences = sent_tokenize(content)

cleaned_sentences=[]
for sentence in sentences:
  cleaned_sentences.append(clean(sentence))

similarity_matrix = np.zeros((len(cleaned_sentences),len(cleaned_sentences)))

for i in range(0,len(cleaned_sentences)):
  for j in range(0,len(cleaned_sentences)):
    if type(find_similarity(cleaned_sentences[i],cleaned_sentences[j])) == np.float64 :
      similarity_matrix[i,j] = find_similarity(cleaned_sentences[i],cleaned_sentences[j])

similarity_matrix

class Graph:
  
  def __init__(self,graph_dictionary):
    if not graph_dictionary:
      graph_dictionary={}
    self.graph_dictionary = graph_dictionary
  
  def vertices(self):
    return self.graph_dictionary.keys()
  
  def edges(self):
    return self.generate_edges()

  def add_vertex(self,vertex):
    if vertex not in graph_dictionary.keys():
      graph_dictionary[vertex] = []
  
  def add_edge(self,edge):
    vertex1,vertex2 = tuple(set(edge))
    if vertex1 in graph_dictionary.keys():
      graph_dictionary[vertex1].append(vertex2)
    else:
      graph_dictionary[vertex1] = [vertex2]

  def generate_edges(self):
    edges = set()
    for vertex in graph_dictionary.keys():
      for edges in graph_dictionary[vertex]:
        edges.add([vertex,edge])
    return list(edges)

similarity_threshold = 0.70
network_dictionary = {}

for i in range(len(cleaned_sentences)):
    network_dictionary[i] = []  

for i in range(len(cleaned_sentences)):
  for j in range(len(cleaned_sentences)):
    if similarity_matrix[i][j] > similarity_threshold:
      if j not in network_dictionary[i]:
        network_dictionary[i].append(j)
      if i not in network_dictionary[j]:
        network_dictionary[j].append(i)

similarity_matrix

graph = Graph(network_dictionary)

def page_rank(graph,iterations = 50,sentences=20):
  ranks = []
  # ranks = {}
  network = graph.graph_dictionary
  current_ranks = np.squeeze(np.zeros((1,len(cleaned_sentences))))
  prev_ranks = np.array([1/len(cleaned_sentences)]*len(cleaned_sentences))
  for iteration in range(0,iterations):
    for i in range(0,len(list(network.keys()))):
      current_score = 0
      adjacent_vertices = network[list(network.keys())[i]]
      for vertex in adjacent_vertices:
          current_score += prev_ranks[vertex]/len(network[vertex])
      current_ranks[i] = current_score
    prev_ranks = current_ranks
  
  for index in range(len(cleaned_sentences)):
      # ranks[index] = prev_ranks[index]
      if prev_ranks[index]: 
        ranks.append((index,prev_ranks[index]))
  # ranks = {index:rank for index,rank in sorted(ranks.items(),key=ranks.get,reverse=True)}[:sentences]
  ranks = sorted(ranks,key = lambda x:x[1],reverse=True)[:sentences]

  return ranks

ranks = page_rank(graph,iterations=1000)

summary = ""
for index,rank in ranks:
  summary+=sentences[index]+" "

summary

# Approach 3
# importing required libraries
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import re
from gensim.models import Word2Vec

# Reading in the training data
import os

X_text = []
counter = 0
file_list = os.listdir('/content/sample_data/dataset/stories_text_summarization_dataset_train/')
for filename in file_list:
  counter+=1
  with open(os.path.join('/content/sample_data/dataset/stories_text_summarization_dataset_train/',filename), 'r', encoding='utf-8') as f:
    content=f.read()
    X_text.append(content)
    f.close()
  if counter % 10000 == 0:
    print(counter)

def remove_punc(sent):
  """ To remove punctuations and special characters """
  cleaned=re.sub(r'[?|!|\'|"|#]', r'', sent)
  cleaned=re.sub(r'[.|,|)|(|\|/]', r' ', sent)
  cleaned=cleaned.strip()
  cleaned=cleaned.replace('\n',' ')
  return cleaned
 
def keep_alpha(sent):
  """ To remove non-alphabetic characters """
  alpha_sent=''
  for word in sent.split():
    alpha_word=re.sub('[^a-z A-Z]+', ' ', word)
    alpha_sent+=alpha_word
    alpha_sent+=' '
  alpha_sent=alpha_sent.strip()
  return alpha_sent

X_text_clean = [remove_punc(sent) for sent in X_text]
X_text_clean = [keep_alpha(sent) for sent in X_text_clean]
X_text_clean = [sent.lower() for sent in X_text_clean]

nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(sent):
  sent_new = ' '.join([i for i in sent if i not in stop_words])
  return sent_new

X_text_clean = [remove_stopwords(r.split()) for r in X_text_clean]

print('No. of sentences',len(X_text_clean))
print('First 5 sentences',X_text_clean[:5])

## Splitting sentences into words for building word embeddings
X_sent_words = []
for sent in X_text_clean:
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
file_list = os.listdir('/content/sample_data/dataset/stories_text_summarization_dataset_test/')
for filename in file_list:
  counter+=1
  with open(os.path.join('/content/sample_data/dataset/stories_text_summarization_dataset_test/',filename), 'r', encoding='utf-8') as f:
    content=f.read()
    X_test.append(content)
    f.close()
  print(counter)

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def summarize(content, num_sentences):
  # splitting into sentences
  sentences = sent_tokenize(content)
  
  # cleaning sentences as above
  sentences_clean = [remove_punc(sent) for sent in sentences]
  sentences_clean = [keep_alpha(sent) for sent in sentences_clean]
  sentences_clean = [sent.lower() for sent in sentences_clean]
  
  # removing stopwords
  sentences_clean = [remove_stopwords(r.split()) for r in sentences_clean]
    
  # making sentence vectors
  sentence_vectors = []
  for i in sentences_clean:
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

summary = summarize(X_test[0],5)
for sent in summary:
  print(sent)

# #############################
# importing required libraries
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import re
from gensim.models import Word2Vec

## Read required libraries & utilities
import os, sys, string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pickle import dump,load

import networkx as nx
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

## Set up working directory
import os
workdir = os.getcwd()
print("The working directory is :", workdir)

NUM_SAMPLES = 500 # number of samples from the original data.

def load_single_doc(filename):
    with open(filename, encoding = 'utf-8') as f:
        text = f.read()
        f.close()
    return text

##  function to extract news story and highlights
def extract_story(doc):
    idx = doc.find('@highlight')
    story_part, highlight_part = doc[:idx], doc[idx:].split('@highlight')
    highlight_part = [h.strip() for h in highlight_part if len(h)>0]
    return story_part, highlight_part

## function to load all stories from data directory
def load_all_stories(dir):
    stories = list()
    doc_iter = 0
    for files in os.listdir(dir):
        doc_iter += 1
        print("processing doc number {}".format(doc_iter))
        filename = os.path.join(dir, files)
        doc = load_single_doc(filename)
        story_part, highlight_part = extract_story(doc)
        if story_part is not None:
            stories.append({'story': story_part, 'highlight' : highlight_part})
            
        if doc_iter == NUM_SAMPLES:
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
stories = load_all_stories(dir)
print('Loaded Stories %d' % len(stories))

# Reading in the training data
import os

X_text = []
counter = 0
file_list = os.listdir('/content/sample_data/dataset/stories_text_summarization_dataset_train')
for filename in file_list:
  counter+=1
  with open(os.path.join('/content/sample_data/dataset/stories_text_summarization_dataset_train',filename), 'r', encoding='utf-8') as f:
    content=f.read()
    X_text.append(content)
    f.close()
  if counter % 10000 == 0:
    print(counter)

def remove_punc(sent):
  """ To remove punctuations and special characters """
  cleaned=re.sub(r'[?|!|\'|"|#]', r'', sent)
  cleaned=re.sub(r'[.|,|)|(|\|/]', r' ', sent)
  cleaned=cleaned.strip()
  cleaned=cleaned.replace('\n',' ')
  return cleaned
 
def keep_alpha(sent):
  """ To remove non-alphabetic characters """
  alpha_sent=''
  for word in sent.split():
    alpha_word=re.sub('[^a-z A-Z]+', ' ', word)
    alpha_sent+=alpha_word
    alpha_sent+=' '
  alpha_sent=alpha_sent.strip()
  return alpha_sent

X_text_clean = [remove_punc(sent) for sent in X_text]
X_text_clean = [keep_alpha(sent) for sent in X_text_clean]
X_text_clean = [sent.lower() for sent in X_text_clean]

nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(sent):
  sent_new = ' '.join([i for i in sent if i not in stop_words])
  return sent_new

X_text_clean = [remove_stopwords(r.split()) for r in X_text_clean]

print('No. of sentences',len(X_text_clean))
print('First 5 sentences',X_text_clean[:5])

## Splitting sentences into words for building word embeddings
X_sent_words = []
for sent in X_text_clean:
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
for filename in file_list:
  counter+=1
  with open(os.path.join('/content/sample_data/dataset/stories_text_summarization_dataset_test',filename), 'r', encoding='utf-8') as f:
    content=f.read()
    X_test.append(content)
    f.close()
  print(counter)

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def summarize(content, num_sentences):
  # splitting into sentences
  sentences = sent_tokenize(content)
  
  # cleaning sentences as above
  sentences_clean = [remove_punc(sent) for sent in sentences]
  sentences_clean = [keep_alpha(sent) for sent in sentences_clean]
  sentences_clean = [sent.lower() for sent in sentences_clean]
  
  # removing stopwords
  sentences_clean = [remove_stopwords(r.split()) for r in sentences_clean]
    
  # making sentence vectors
  sentence_vectors = []
  for i in sentences_clean:
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

summary = summarize(X_test[0],5)
for sent in summary:
  print(sent)
