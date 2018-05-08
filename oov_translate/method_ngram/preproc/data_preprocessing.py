import subprocess as sp
import os
import sys 
import xml.etree.ElementTree as et
import heapq
from random import shuffle
import kenlm

dir_path = os.path.dirname(os.path.realpath(__file__))+"/"
sys.path.insert(0, dir_path+'../../../oov_translate')
from config import *
from utils import *

def train_lm(ngram, train_data, lm_model):
    '''
    param:
        ngram: length of "markovity"
        train_data: path to training data
        lm_model: path to lm file
    return:
        language model in kenLM format
    '''
#     if os.path.exists(lm_model):
#         print("Language model exists at: "+lm_model)
#     else:
    stdout, stderr = sh(lm_builder+\
                        " -o "+str(ngram)+\
                        " < "+train_data+\
                        " > "+lm_model)
    print("Language model generated at: "+lm_model)
    return kenlm.Model(lm_model)

def get_perplexity(lm_model, test_data):
    '''
    param:
        lm_model: path to lm file
        test_data: path to test data
    return:
        perp_w_oov: perplexity including OOVs
        perp_wo_oov: perplexity excluding OOVs
        oov_num: nubmer of OOVs
        token_num: number of tokens
    '''
    stdout, _ = sh(query_perplexity+' '+lm_model+\
                   ' < '+test_data+\
                   ' | tail -n 4')
    stdout = stdout.strip().split('\n')
    perp_w_oov = float(stdout[0].split(':')[1].strip())
    perp_wo_oov = float(stdout[1].split(':')[1].strip())
    oov_num = int(stdout[2].split(':')[1].strip())
    token_num = int(stdout[3].split(':')[1].strip())
    return perp_w_oov, perp_wo_oov, oov_num, token_num

def get_cross_entropy(lm, sent):
    '''
    param:
        lm: language model in kenLM format
        sent: stripped sentence
    return:
        cross entropy estimation based on
    https://courses.engr.illinois.edu/cs498jh/Slides/Lecture04.pdf
    '''
    return -lm.score(sent)*1.0/len(sent.split(' '))


