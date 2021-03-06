# -*- coding: utf-8 -*-
"""TensorFlow with GPU

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/notebooks/gpu.ipynb

# Tensorflow with GPU

This notebook provides an introduction to computing on a [GPU](https://cloud.google.com/gpu) in Colab. In this notebook you will connect to a GPU, and then run some basic TensorFlow operations on both the CPU and a GPU, observing the speedup provided by using the GPU.

## Enabling and testing the GPU

First, you'll need to enable GPUs for the notebook:

- Navigate to Edit→Notebook Settings
- select GPU from the Hardware Accelerator drop-down

Next, we'll confirm that we can connect to the GPU with tensorflow:
"""
# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from google.colab import drive
drive.mount('/content/drive')

"""## Observe TensorFlow speedup on GPU relative to CPU

This exp constructs a typical convolutional neural network layer over a
random image and manually places the resulting ops on either the CPU or the GPU
to compare execution speed.
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
  
# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
gpu()

# Run the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

## Set up working directory
import os
work_dir = os.getcwd()
print("The working directory is :", work_dir)


## Importing the necessary modules in current namespace
import os, sys, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime, os
import zipfile


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
from tensorflow.keras import initializers, regularizers, constraints
from pickle import dump,load

# %load_ext tensorboard

## Variables initialization
NO_SAMP       = 10000 # number of samples from the original data.
SIZE_VOCAB    = 50000 # max vocabulary size
DIM_EMBEDDED  = 300   # embedding dimension from the embedding matrix glove
DIM_LATENT    = 128   # latent dimentionality of the encoding space

SIZE_BATCH    = 64             # batch size for each training pass
NO_OF_EPOCS   = 50             # number of epochs to be trained
VALIDATION_SPLIT_RATIO = 0.2   # propertion of the validation sample

text_container         = []  # placeholder to save in input text (Story)
target_container       = []  # placeholder to save in target text (Summary)
tt_container = []  # placeholder to save in target text offset by 1 (Summary)
path_to_zip_file = '/content/drive/My Drive/dataset.zip'
directory_to_extract_to = '/content/sample_data'

# Extract the dataset from google drive folder
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)
    
    
    
"""
Encoder-Decoder Class with Attention comprised of two sub-models:

Encoder: The encoder is responsible for stepping through the input time steps and encoding the entire sequence into a fixed length vector called a context vector.
Decoder: The decoder is responsible for stepping through the output time steps while reading from the context vector.

Let E = encoder be the encoder's output tensor of shape (-1, INPUT_LENGTH,64), and let E[i] (encoder[:,i,:]) be the encoder's output vector (of shape (-1, 64)) for i-th English character. This is source-side information (H-s in the paper).

Let D = decoder be the decoder's output tensor of shape (-1, OUTPUT_LENGTH,64), and let D[j] (decoder[:,j,:]) be the decoder's output vector (of shape (-1, 64)) for j-th Katakana character. This is target-side information (H-t in the paper).

Let O = output be the final Katakana output of shape (-1, OUTPUT_LENGTH, output_dict_size). and let O[j] (output[:,j,:]) be the output (probability distribution of shape (-1, output_dict_size)) of j-th Katakana character.
"""
class seq_seq_attention(Layer):
    def __init__(self,**kwargs):
        super(seq_seq_attention,self).__init__(**kwargs)

    def build(self,input_shape):
        """
        Matrices for creating the context vector
        """
        self.W=self.add_weight(shape=(input_shape[-1],1),initializer="normal", name="att_weight")
        self.b=self.add_weight(shape=(input_shape[1],1),initializer="zeros",name="att_bias")        
        super(seq_seq_attention, self).build(input_shape)

    def call(self,x):
        """
        Function which does the computation and is passed through a softmax layer to calculate the attention probabilities and context vector.
        """
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def get_out_shape_tf(self,input_shape):
        """
        For Keras internal compatibility checking.
        """
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        """
        The get_config() method collects the input shape and other information about the model.
        """
        return super(seq_seq_attention,self).get_config()
        
        
 # Methods 
 
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

# load num_stories
# /content/sample_data/dataset
o_dir = os.path.join(work_dir, 'sample_data/dataset', 'stories_text_summarization_dataset_train') 
num_stories = loadStory(o_dir)
print('Loaded Stories %d' % len(num_stories))

print(num_stories[0])


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

stories_cleaned[0]

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


# Creating the plot of story distribution
word_cnt = []
summary_cnt = []


for i in text_container:
      word_cnt.append(len(str(i).split()))

for i in target_container:
      summary_cnt.append(len(str(i).split()))

length_df = pd.DataFrame({'text':word_cnt, 'summary':summary_cnt})
length_df.hist(bins = 30)
plt.show()

print(max(summary_cnt))
print(max(word_cnt))

# Sample out the first N inputs 
def sample(seq_len_threshold):
    idx = [i for i,v in enumerate(text_container) if len(v) <= seq_len_threshold]
    input_texts_subsample = [text_container[i] for i in idx]
    target_texts_subsample = [target_container[i] for i in idx]
    target_texts_inputs_subsample = [tt_container[i] for i in idx]
    return input_texts_subsample,target_texts_subsample,target_texts_inputs_subsample 
    
text_container,target_container,tt_container = sample(seq_len_threshold=1200)

print(len(text_container))
print(len(target_container))
print(len(tt_container))

# Tokenize inputs 
tkn_i = Tokenizer(num_words = SIZE_VOCAB)
tkn_i.fit_on_texts(text_container)
inpt_sq = tkn_i.texts_to_sequences(text_container)

print(text_container[2])
print(" ")
print(inpt_sq[2])

print(len(text_container[2].split()))
print(len(inpt_sq[2]))


w2indx = tkn_i.word_index
print("Found % unique input tokens" % len(w2indx))
print("")
print(w2indx)

max_len_input = max(len(s) for s in inpt_sq)
max_len_input

# Tokenize the target feeds
tkn_target = Tokenizer(num_words=SIZE_VOCAB, filters='')

tkn_target.fit_on_texts(target_container + tt_container) ##inefficient
trgt_sq = tkn_target.texts_to_sequences(target_container)
trgt_sq_i = tkn_target.texts_to_sequences(tt_container)

print(target_container[2])
print(" ")
print(trgt_sq[2])
print(" ")
print(trgt_sq_i[2])

print(len(target_container[2].split()))
print(len(trgt_sq[2]))
print(len(trgt_sq_i[2]))

# get the word to index mapping for the output language
word2indx_outputs = tkn_target.word_index
print('Found %s unique output tokens.' % len(word2indx_outputs))
print("")
print(word2indx_outputs)

num_words_output = len(word2indx_outputs) + 1 
num_words_output

max_len_target = max(len(s) for s in trgt_sq)
max_len_target

encoder_inputs = pad_sequences(inpt_sq, maxlen= max_len_input)
print("encoder_inputs shape:", encoder_inputs.shape)
print("encoder_inputs[0]:", encoder_inputs[0])

decoder_inputs = pad_sequences(trgt_sq_i, maxlen= max_len_target, padding='post')
print("decoder_inputs shape:", decoder_inputs.shape)
print("decoder_inputs[0]:", decoder_inputs[0])

decoder_targets = pad_sequences(trgt_sq, maxlen= max_len_target, padding='post')
print("decoder_targets shape:", decoder_targets.shape)
print("decoder_targets[0]:", decoder_targets[0])

path_to_zip_file = '/content/drive/My Drive/glove.6B.zip'
directory_to_extract_to = '/content/sample_data'

import zipfile
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

"""
Download pre-trained word vectors
The links below contain word vectors obtained from the respective corpora. If you want word vectors trained on massive web datasets, you need only download one of these text files! Pre-trained word vectors are made available under the Public Domain Dedication and License.

Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download): glove.42B.300d.zip
Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): glove.840B.300d.zip
Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download): glove.6B.zip
Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 200d vectors, 1.42 GB download): glove.twitter.27B.zip

If the web datasets above don't match the semantics of your end use case, you can train word vectors on your own corpus.

$ git clone http://github.com/stanfordnlp/glove
$ cd glove && make
$ ./demo.sh
"""

## store all the pre-trained word vectors. the source file is a space separated file. save vectors to dictionary
print("loading word vectors.....")
word2vec = {}
with open(os.path.join(work_dir, "/content/sample_data/", "glove.6B.%sd.txt" % DIM_EMBEDDED), encoding='utf') as f:
    ## word vec[0], word vec[1], word vec[2] .....
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vector
print('Found %s word vectors.' % len(word2vec))

# Prepare embedding matrix (Vocab Size x Dimension)
print("Filling pre-trained embeddings.....")
vocab_size = min(SIZE_VOCAB, len(w2indx) + 1)
embedding_matrix = np.zeros((vocab_size, DIM_EMBEDDED))

for word, indx in w2indx.items():
    if indx < SIZE_VOCAB:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[indx] = embedding_vector
            
print(embedding_matrix[1:2])

"""
The encoder-decoder model provides a pattern for using recurrent neural networks to address challenging sequence-to-sequence prediction problems such as machine translation.

The function takes 3 arguments, as follows:

n_input: The cardinality of the input sequence, e.g. number of features, words, or characters for each time step.
n_output: The cardinality of the output sequence, e.g. number of features, words, or characters for each time step.
n_units: The number of cells to create in the encoder and decoder models, e.g. 128 or 256.
The function then creates and returns 3 models, as follows:

train: Model that can be trained given source, target, and shifted target sequences.
inference_encoder: Encoder model used when making a prediction for a new source sequence.
inference_decoder Decoder model use when making a prediction for a new source sequence.
"""
decoder_targets_one_hot = np.zeros(
  (
    len(text_container),
    max_len_target,
    num_words_output
  ),
  dtype='float32'
)

# assign the values
for i, d in enumerate(decoder_targets):
    for t, word in enumerate(d):
        if word != 0:
            decoder_targets_one_hot[i, t, word] = 1
            
decoder_targets_one_hot[1]

# Creating Embedding Layer 
embedding_layer = Embedding(vocab_size, 
                            DIM_EMBEDDED, 
                            weights=[embedding_matrix], 
                            input_length = max_len_input, 
                            trainable=False)






"""
Model Building Tensorflow 2.0 version compatible 
input_text_bgru = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedding_layer_bgru = Embedding(len(tokenizer.word_index) + 1,
                                    300,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
g = embedding_layer_bgru(input_text_bgru)
g = SpatialDropout1D(0.4)(g)
g = Bidirectional(GRU(64, return_sequences=True))(g)
att = Attention(MAX_SEQUENCE_LENGTH)(g)
g = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "he_uniform")(g)
avg_pool1 = GlobalAveragePooling1D()(g)
max_pool1 = GlobalMaxPooling1D()(g)
g = concatenate([att,avg_pool1, max_pool1])
g = Dense(128, activation='relu')(g)
bgru_output = Dense(2, activation='softmax')(g)

query_value_attention_seq = tf.keras.layers.Attention()(
    [query_seq_encoding, value_seq_encoding])
"""
## building encoder part
enc_i    = Input((max_len_input,))
enc_e = embedding_layer(enc_i)

## bidirectional LSTM with different dropouts rates
## keras.layers.recurrent.Recurrent(return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, implementation=0)
"""
weights: list of Numpy arrays to set as initial weights. The list should have 3 elements, of shapes: [(input_dim, output_dim), (output_dim, output_dim), (output_dim,)].
return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
return_state: Boolean. Whether to return the last state in addition to the output.
go_backwards: Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.
stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.
implementation: one of {0, 1, or 2}. If set to 0, the RNN will use an implementation that uses fewer, larger matrix products, thus running faster on CPU but consuming more memory. If set to 1, the RNN will use more matrix products, but smaller ones, thus running slower (may actually be faster on GPU) while consuming less memory. If set to 2 (LSTM/GRU only), the RNN will combine the input gate, the forget gate and the output gate into a single matrix, enabling more time-efficient parallelization on the GPU.

Note: RNN dropout must be shared for all gates, resulting in a slightly reduced regularization.
input_dim: dimensionality of the input (integer). This argument (or alternatively, the keyword argument input_shape) is required when using this layer as the first layer in a model.
input_length: Length of input sequences, to be specified when it is constant. This argument is required if you are going to connect Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed). Note that if the recurrent layer is not the first layer in your model, you would need to specify the input length at the level of the first layer (e.g. via the input_shape argument)
"""
enc_LSTM           = LSTM(DIM_LATENT, return_state=True, return_sequences=True,dropout=0.2)
enc_LSTM_r       = LSTM(DIM_LATENT,return_state=True,return_sequences=True,dropout=0.05,go_backwards=True)

encoder_output_, encoder_h, encoder_c = enc_LSTM(enc_e)
encoder_output_r, encoder_hr, encoder_cr = enc_LSTM_r(enc_e)

encoder_h_final = Add()([encoder_h, encoder_hr])
encoder_c_final = Add()([encoder_c, encoder_cr])
encoder_output_final = Add()([encoder_output_, encoder_output_r])

##keep only the states to pass to the decoder
encoder_states    = [encoder_h_final, encoder_c_final]

## building decoder; using encoder_states as initial state
decoder_input_    = Input((max_len_target,))
decoder_embedding = Embedding(num_words_output, DIM_EMBEDDED)
decoder_embedding_x = decoder_embedding(decoder_input_)
decoder_lstm      = LSTM(DIM_LATENT, return_state=True, return_sequences=True, dropout=0.2)
decoder_output_, decoder_h, decoder_c = decoder_lstm(decoder_embedding_x, initial_state=encoder_states)

##seq_seq_attention
atten = seq_seq_attention()(encoder_output_final)
decoder_output_final = Add()([decoder_output_, atten])

decoder_dense     = Dense(num_words_output, activation='softmax')
decoder_output_   = decoder_dense(decoder_output_final)

model = Model(inputs= [enc_i, decoder_input_], outputs = decoder_output_)

model.summary()

plot_model(model, to_file=os.path.join(work_dir, "sample_data", 'model_plot.png'), show_shapes=True, show_layer_names=True)

"""
The purpose of loss functions is to compute the quantity that a model should seek to minimize during training.
"""
'''
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(layers.Activation('softmax'))

loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss_fn, optimizer='adam')
'''
def custom_loss(y_true, y_pred):
  # both are of shape N x T x K
  mask = K.cast(y_true > 0, dtype='float32')
  out = mask * y_true * K.log(y_pred)
  return -K.sum(out) / K.sum(mask)

## Define Accuracy 
def acc(y_true, y_pred):
  # both are of shape N x T x K
  targ = K.argmax(y_true, axis=-1)
  pred = K.argmax(y_pred, axis=-1)
  correct = K.cast(K.equal(targ, pred), dtype='float32')

  # 0 is padding, don't include those
  mask = K.cast(K.greater(targ, 0), dtype='float32')
  n_correct = K.sum(mask * correct)
  n_total = K.sum(mask)
  return n_correct / n_total

## compile model
model.compile(optimizer='adam', loss=custom_loss, metrics= [acc])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)

## initialize tensorboard
logdir = os.path.join(work_dir, "Outputs", "logs\logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard = TensorBoard(log_dir = logdir, histogram_freq=1, write_images=True)

## train the model
r = model.fit([ encoder_inputs, decoder_inputs], 
                 decoder_targets_one_hot, 
                 batch_size       = SIZE_BATCH,
                 epochs           = NO_OF_EPOCS,
                 validation_split = VALIDATION_SPLIT_RATIO,
                 callbacks        = [tensorboard])

model.save(os.path.join(work_dir, "Outputs",'base_model_1.h5'))

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs

## we need to create another model that takes the RNN states from encoder and previous word as input and accept a T=1 sequence

##the encoder will be standalone, from this we will get the hidden states
encoder_model = Model(enc_i, encoder_states)

decoder_input_h = Input(shape=(DIM_LATENT,))
decoder_input_c = Input(shape=(DIM_LATENT,))

decoder_states_input_ = [decoder_input_h, decoder_input_c] 

decoder_input_single = Input(shape=(1,))
decoder_input_single_embedding = decoder_embedding(decoder_input_single)

decoder_outputs, decoder_h, decoder_c = decoder_lstm(decoder_input_single_embedding, initial_state=decoder_states_input_)

decoder_states = [decoder_h, decoder_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_input_single] + decoder_states_input_,
                     [decoder_outputs] + decoder_states)


print("The encoder model")
encoder_model.summary()
print("")
print("The decoder model")
decoder_model.summary()
print("")

plot_model(encoder_model, to_file=os.path.join(work_dir, 'Outputs','encoder_model_plot.png'), show_shapes=True, show_layer_names=True)

indx2word_inputs= {v:k for k, v in w2indx.items()}
indx2word_outputs = {v:k for k, v in word2indx_outputs.items()}

"""
A RNN layer (or stack thereof) acts as "encoder": it processes the input sequence and returns its own internal state. Note that we discard the outputs of the encoder RNN, only recovering the state. This state will serve as the "context", or "conditioning", of the decoder in the next step.
Another RNN layer (or stack thereof) acts as "decoder": it is trained to predict the next characters of the target sequence, given previous characters of the target sequence. Specifically, it is trained to turn the target sequences into the same sequences but offset by one timestep in the future, a training process called "teacher forcing" in this context. Importantly, the encoder uses as initial state the state vectors from the encoder, which is how the decoder obtains information about what it is supposed to generate. Effectively, the decoder learns to generate targets[t+1...] given targets[...t], conditioned on the input sequence.
"""
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2indx_outputs['<sos>']

    # if we get this we break
    eos = word2indx_outputs['<eos>']

    # Create the translation
    output_sentence = []
    for _ in range(max_len_target):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Get next word
        idx = np.argmax(output_tokens[0, 0, :])

        # End sentence of EOS
        if eos == idx:
            break

        word = ''
        if idx > 0:
            word = indx2word_outputs[idx]
            output_sentence.append(word)


        target_seq[0, 0] = idx
        states_value = [h, c]


    return ' '.join(output_sentence)
"""
1) Encode the input sequence into state vectors.
2) Start with a target sequence of size 1 (just the start-of-sequence character).
3) Feed the state vectors and 1-char target sequence to the decoder to produce predictions for the next character.
4) Sample the next character using these predictions (we simply use argmax).
5) Append the sampled character to the target sequence
6) Repeat until we generate the end-of-sequence character or we hit the character limit.
"""
while True:

    i = np.random.choice(len(text_container))
    input_seq = encoder_inputs[i:i+1]
    summary = decode_sequence(input_seq)
    print('-')
    print('Input:', text_container[i])
    print('summary:', summary)

    generate_summary = input("next? [Y/n]")
    if generate_summary and generate_summary.lower().startswith('n'):
        break
        

"""
Method-2 
pip install --upgrade transformers
pip install bert-extractive-summarizer
pip install spacy==2.1.3
pip install transformers==2.2.2
pip install neuralcoref
pip install transformers==2.2.0
pip install spacy==2.0.12
python -m spacy download en_core_web_md

from summarizer import Summarizer,TransformerSummarizer


from transformers import *
# Load model, model config and tokenizer via Transformers
custom_config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
custom_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=custom_config)

from summarizer import Summarizer

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences,num_beams):
    batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60).to(torch_device)
    translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text
    
    
context = "Some are born great, some achieve greatness, and some have greatness thrust upon them"
num_return_sequences=5
num_beams=10
get_response(context,num_return_sequences,num_beams)


from summarizer import Summarizer

body = '''
a man angry that a family was talking during a movie threw popcorn at the son and then shot the father in the arm according to police in philadelphia pennsylvania james joseph cialella was charged with attempted murder aggravated assault and weapons charges james joseph cialella was charged with attempted murder aggravated assault and weapons violations a police report said cialella told the family sitting in front of him in the theater on christmas day to be quiet police said an argument ensued while others at the riverview movie theatre watched the curious case of benjamin button starring brad pitt and cate blanchett the philadelphia inquirer reported cialella then approached the family from the left side of the aisle and shot the father who was not identified as he was standing between cialella and his family according to the police report the victim was taken to jefferson hospital with a gunshot wound to his left arm police said cialella was carrying a keltec handgun clipped inside his sweatpants police said he was arrested and taken into custody
summary: the crashes is a injured in a cocaine a hospital of a state of the injuries and a injuries and a way and help the vocabulary
next?
'''

model = Summarizer()
result = model(body, min_length=60)
full = ''.join(result)
print(full)


body = '''
        a gunman who opened fire on his coworkers at a walmart in reno nevada was arrested without incident friday after a sixhour standoff with authorities reno police said three employees were injured and two remained hospitalized friday night in serious to stable condition lt mohammad rafaqat told cnn john gillane will be charged with three counts of attempted murder in the shootings rafaqat said the incident began at am am et when gillane walked into the store and opened fire on employees before barricading himself in an office it ended at pm pm et after hours of police negotiations a possible motive for the attack was not immediately clear rafaqat said our thoughts and prayers are with our employees at this time walmart spokesman dan fogleman said cnns sara pratley contributed to this report
        '''
        
        
bert_model = Summarizer()
bert_summary = ''.join(bert_model(body, min_length=60))
print(bert_summary)


GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
full = ''.join(GPT2_model(body, min_length=60))
print(full)


model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
full = ''.join(model(body, min_length=60))
print(full)
"""
