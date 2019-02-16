#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:02:05 2018

@author: pxu3
"""
import csv
import os
import re
import itertools
import pandas as pd
import numpy as np
import tensorflow as tf
np.random.seed(1234)
from itertools import groupby
import contractions
import spacy
from collections import Counter
import chardet 
from os.path import join, exists, split

nlp = spacy.load('en')

# Functions for text pre-processing

def unescapematch(matchobj):
    """
    Converts from hex to unicode: \u201c -> '
    """
    escapesequence = matchobj.group(0)
    digits = escapesequence[2:6]
    ordinal = int(digits, 16)
    char = chr(ordinal)
    return char

def replace_url_phone(text):
    """
    Accepts a text string and replaces:
    1) emails with emailid
    2) urls with url
    3) phone numbers with phonenumber
    """
    email_regex = ("([A-Za-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`""{|}~-]+)*(@)(?:[A-Za-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|""\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)")
    url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    phone_regex =  ("([+]\d{12}|[+]?\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)[-\.\s]??\d{3}[\-\.\s]??\d{4}|[+]\d{1,2}[-\.\s]??\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4})")
    text = re.sub(email_regex, 'emailid',text)
    text = re.sub(url_regex, 'url',text)
    text = re.sub(phone_regex, 'phonenumber', text)
    return text
 
def clean_string(string):
    """
    Cleans string
    1) replaces e.g: \u201c -> '
    2) replaces contractions lile I'm -> I am
    3) replaces emailids, urls and phone numbers
    4) places a space between words and punctuation
    5) replaces symbols with words like $ -> dollar
    
    """
    string = re.sub(r'(\\u[0-9A-Fa-f]{4})', unescapematch, string)
    # remove remaining hexcodes
    string = re.sub(r'[^\x00-\x7f]',r'', string)
    string = contractions.fix(string)
    string = replace_url_phone(string)
    # add space between punctuation and text
    string = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", string)
    string = re.sub('\$', " dollar ", string)
    string = re.sub('\%', " percent ", string)
    string = re.sub('\&', " and ", string)
    string = re.sub('\"'," quote ", string)
    string = string.replace("\\","")
    # remove multiple instances of punctuation
    re.sub(r'(\W)(?=\1)', '', string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def check_encoding(filepath): 
    """
    Check encoding of a file 
    """
    rawdata = open(filepath, 'rb').read()
    result = chardet.detect(rawdata)
    charenc = result['encoding']
    return charenc

def replace_entity(sentence):
    """
    Replaces specific entities in text
    Ex: Harry lives in Pennsylvania -> PERSON lives in GPE 
    
    """
    doc = nlp(sentence)
    sentence_ = []
    for ent in doc.ents:
        ent.merge()
    for token in doc:
        
        if token.ent_type_ in ['ORG','GPE','LOC','NORP','CARDINAL','FACILITY','MONEY','PERSON','DATE','TIME',\
                      'PERCENT','QUANTITY','ORDINAL']:
            sentence_.append(token.ent_type_+'_ent')
            
        else:
            sentence_.append(token.orth_)
        # combining detection of compound entities like State-College (ORG ORG ORG) -> ORG
        sentence_set = [x[0] for x in groupby(sentence_)]
    return sentence_set


# Functions to load data and produce embeddings
def load_data_and_labels(folderpath,filename):
    """
    Loads data, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    with open(os.path.join(folderpath, filename), 'r',encoding="utf-8") as datafile:
         reader = csv.reader(datafile,delimiter='\t')
         first_iteration = 1
         x_text = []
         y = []
         for row in reader:
             if first_iteration==1:
                 print('The columns in the file are:\n ',row)
                 #text_index = row.index('CommentText')
                 text_index = row.index('SENTENCES')
                 class_index = row.index('binary_labels')
                 first_iteration =0
             else:
                #text = clean_string(row[text_index])
                comment = eval(row[text_index])
                labels = eval(row[class_index])
                if len(comment)>0:
                   for i in range(len(comment)):
                       sentence = clean_string(comment[i])
                       sentence = replace_entity(sentence)
                       sentence = [word.lower() for word in sentence]
                       label = labels[i][0]
                       x_text.append(sentence)
                       y.append(label)
         return x_text,y
         
        
def pad_sequences(sentences, pad_len, padding_word="<PAD/>"):
    """
    Pads the sentences to make the length equal to the sentence with maximum number of words
    """
    if pad_len is not None:
        sequence_length = pad_len
    else:
        sequence_length = max(len(x) for x in sentences)
    
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence)<sequence_length:
           num_padding = sequence_length - len(sentence)
           new_sentence = sentence + [padding_word] * num_padding
        else:
           num_padding = 0
           new_sentence = [x for x in sentence[0:pad_len]]
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return word_counts, vocabulary, vocabulary_inv

def load_data(folderpath,filename):
    """
    Loads the preprocessed data.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(folderpath,filename)
    sentences_padded = pad_sequences(sentences, padding_word="<PAD/>", pad_len=30)
    vocab_size,vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, vocab_size] 

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_trainable_dataset(folderpath,filename):
    
    x_text, y, vocabulary, vocabulary_inv_list, vocab_size = load_data(folderpath,filename)
    
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    
    #y = y.argmax(axis=1)

    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    
    x_text = x_text[shuffle_indices]
    y = y[shuffle_indices]
    
    train_len = int(len(x_text) * 0.8)
    
    x_train = x_text[:train_len]
    y_train = y[:train_len]
    x_test = x_text[train_len:]
    y_test = y[train_len:]
    
    return x_train, y_train, x_test, y_test, vocabulary, vocabulary_inv

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev
    
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]