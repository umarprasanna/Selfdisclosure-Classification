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
from itertools import groupby
import contractions
import spacy
from collections import Counter
import chardet 


nlp = spacy.load('en')


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
    # remove multiple instances of punctuation
    re.sub(r'(\W)(?=\1)', '', string)
    # add space between punctuation and text
    string = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub('\$', " dollar ", string)
    string = re.sub('\%', " percent ", string)
    string = re.sub('\&', " and ", string)
    return string.strip()
                                   
def check_encoding(filepath): 
    """
    Check encoding of a file 
    """
    rawdata = open(filepath, 'rb').read()
    result = chardet.detect(rawdata)
    charenc = result['encoding']
    return charenc


def load_data(folderpath,filename,target):
    """
    Generates one line at a time of the comments and labels from file
    If target=True, labels are also yielded
    """
    with open(os.path.join(folderpath, filename), 'r',encoding="utf-8") as datafile:
         reader = csv.reader(datafile,delimiter='\t')
         first_iteration = 1
         for row in reader:
             if first_iteration==1:
                 print('The columns in the file are:\n ',row)
                 text_index = row.index('CommentText')
                 class_index = row.index('Category')
                 first_iteration =0
             else:
                 if target==True:
                    text = clean_string(row[text_index])
                    if len(text)>0:
                       text = replace_entity(text)
                       text = [word.lower() for word in text]
                       label = eval(row[class_index])
                       yield text,label
                 else:
                    text = clean_string(row[text_index])
                    if len(text)>0:
                       text = replace_entity(text)
                       text = [word for word in text]
                       yield text
    

                      
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
            sentence_.append(token.string)
        # combining detection of compound entities like State-College (ORG ORG ORG) -> ORG
        sentence_set = [x[0] for x in groupby(sentence_)]
    return sentence_set



def pad_sequences(sentences, padding_word="<PAD/>", pad_len=None):
    if pad_len is not None:
        sequence_length = pad_len
    else:
        sequence_length = max(len(x) for x in sentences)
    
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
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