# Case-Study-Text-Summarization
 Build a text summarization system to highlight a summary of a given document (news article). The training and testing data set is provided for the same. Generate both extractive as well as abstractive summary separately.  (If not able to implement the abstractive text summarization, please do send the one-page document explaining the approach)

# Prerequiste to execute the code
pip install --upgrade transformers \n
pip install bert-extractive-summarizer \n
pip install spacy==2.1.3 \n
pip install transformers==2.2.2 \n
pip install neuralcoref \n
pip install transformers==2.2.0 \n
pip install spacy==2.0.12 \n
python -m spacy download en_core_web_md \n

# Dataset link: 
https://drive.google.com/file/d/1VGthRzHtBSIO182zMCMiqY-YV-D0mLLG/view?usp=sharing

Data-set contains new articles (documents) for training and testing. The summary (or highlight) in the training set is represented as @highlight.

# Case 1: Abstractive Summary
Steps to execute the code 

$ git clone http://github.com/stanfordnlp/glove

$ cd glove && make

$ ./demo.sh

$ python Generate_Abstractive_Summary.py


