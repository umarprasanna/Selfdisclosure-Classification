#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:15:05 2018

@author: pxu3
"""
import numpy as np
np.random.seed(1234)
from gensim.models import Word2Vec,KeyedVectors
from get_data import load_data, check_encoding


filepath='/Users/pxu3/Desktop/Spring 2019/Research/Data/Data.txt'
if check_encoding(filepath) != 'utf-8':
   
    with open(filepath, 'rb') as source_file:
         with open('/Users/pxu3/Desktop/Spring 2019/Research/Data/Data.txt', 'w+b') as dest_file:
              contents = source_file.read()
              dest_file.write(contents.decode(check_encoding(filepath)).encode('utf-8'))
              

print('Loading data and pretrained word2vec model....')
sentences = load_data('/Users/pxu3/Desktop/Spring 2019/Research/Data/','Data.txt',target=False)
google_wv = KeyedVectors.load_word2vec_format('/Users/pxu3/Desktop/Deep Learning/1Bword2vec/GoogleNews-vectors-negative300.bin', binary=True)
print('Finished.....')

model = Word2Vec(size=300, min_count=2, iter=10)
model.build_vocab(sentences)
training_examples_count = model.corpus_count
# below line will make it 1, so saving it before
model.build_vocab([list(google_wv.vocab.keys())], update=True)
model.intersect_word2vec_format("/Users/pxu3/Desktop/Deep Learning/1Bword2vec/GoogleNews-vectors-negative300.bin",binary=True, lockf=1.0)
model.train(sentences,total_examples=training_examples_count, epochs=model.iter)


#model.save("word2vec_model2")
#model1 = Word2Vec.load("word2vec_model")
#model.wv.save("word2vec_model_vectors2")

