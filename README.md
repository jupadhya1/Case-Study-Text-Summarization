# Case-Study-Text-Summarization
 Build a text summarization system to highlight a summary of a given document (news article). The training and testing data set is provided for the same. Generate both extractive as well as abstractive summary separately.  (If not able to implement the abstractive text summarization, please do send the one-page document explaining the approach)

# Prerequiste to execute the code
pip install --upgrade transformers

pip install bert-extractive-summarizer 

pip install spacy==2.1.3 

pip install transformers==2.2.2 

pip install neuralcoref 

pip install transformers==2.2.0 

pip install spacy==2.0.12 

python -m spacy download en_core_web_md 

# Dataset link: 
https://drive.google.com/file/d/1VGthRzHtBSIO182zMCMiqY-YV-D0mLLG/view?usp=sharing

Data-set contains new articles (documents) for training and testing. The summary (or highlight) in the training set is represented as @highlight.

# Case 1: Abstractive Summary
Steps to execute the code 

$ git clone http://github.com/stanfordnlp/glove

$ cd glove && make

$ ./demo.sh

$ python Generate_Abstractive_Summary.py


BERT,GPT-2 & PEGASUS transformer approach 
bert_model = Summarizer()
bert_summary = ''.join(bert_model(body, min_length=60))
print(bert_summary)


GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
full = ''.join(GPT2_model(body, min_length=60))
print(full)


model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
full = ''.join(model(body, min_length=60))
print(full)

