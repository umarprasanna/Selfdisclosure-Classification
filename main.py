#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:15:05 2018

@author: pxu3
"""

# Necessary Imports
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
from gensim.models import word2vec
from gensim.models import Word2Vec,KeyedVectors
from copy import deepcopy
from multiprocessing import cpu_count
from get_data import load_data, check_encoding

nlp = spacy.load('en')

#filepath='/Users/pxu3/Desktop/Spring 2019/Research/Data/Data.txt'
#if check_encoding(filepath) != 'utf-8':
   
#    with open(filepath, 'rb') as source_file:
#         with open('/Users/pxu3/Desktop/Spring 2019/Research/Data/Data.txt', 'w+b') as dest_file:
#              contents = source_file.read()
#              dest_file.write(contents.decode(check_encoding(filepath)).encode('utf-8'))
              

