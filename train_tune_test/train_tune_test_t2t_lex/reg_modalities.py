from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys 
import numpy as np
import random

from tensor2tensor.utils import expert_utils
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.utils import modality
from tensor2tensor.data_generators import wmt
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.data_generators import generator_utils
import tensorflow as tf
from collections import defaultdict
from seq2seq.data import vocab
#from reg_problems import *

def pad_to_lex_cap(lex_list, lex_cap):
    return np.tile(lex_list, int(lex_cap/len(lex_list))+1)[:lex_cap]

def get_lex_to_emb_table(lex_list_list, emb_dict):
    emb_table = []
    for lex_list in lex_list_list:
        emb_list = [emb_dict[lex] if lex in emb_dict else np.random.normal(0, 0.1, (300)) for lex in lex_list]
        emb_table.append(np.array(emb_list, dtype=float))
    return emb_table

def read_vocab(vocab_file):
    vocab = []
    with open(vocab_file) as f:
        for l in f:
            vocab.append(l.strip())
    return vocab

@registry.register_symbol_modality("lex_modality")
class LexModality(modality.Modality):
    '''
    transforms data to a space interpretable by t2t models.
    Input: embedding
    Output: linear transformation + softmax

    input_modalities and target_modality are configured in hparams

    utils.T2TModel._create_modalities has registry.create_modality(), which is defined in utils.registry.py
    modalities used by Text2TextProblem are: input_modality: SYMBOL, target_modality: SYMBOL

    Modality template: at utils.modality.py
    - top_dimensionality (property, not implemented): vocab_size
    - name (property)
    - _body_input_depth (property): hidden_size
    - _model_hparams (initialized from outside)
    - _vocab_size (initialized from outside)

    - bottom (not implemented): transform one shard of input; called on inputs entering the model
    - bottom_sharded: transform the inputs
    - top (not implemented): generate predictions/logits for one shard of output; called on model outputs to generate predictions/logits
    - top_sharded: generate prediction/logits for all shards
    - targerts_bottom: transform one shard of targets; called on targets entering the model (e.g., decoder)
    - targets_bottom_sharded: transform the targets
    - loss: compute loss numerator and denominator for one shhard of output; called on predictions (outputs of top) and targets
    - loss_sharded: compute loss for all shards
    '''
    def __init__(self, model_hparams, vocab_size=None):
        super(LexModality, self).__init__(model_hparams, vocab_size)

        # load the lexicon
        lexicon_dict_file = model_hparams.lexicon_dict_file
        tf.logging.info("loading the lexicon from: "+lexicon_dict_file+" ...")
        lexicon_dict = vocab.get_lexicon_dict(lexicon_dict_file)
        self.lexicon_dict = lexicon_dict
        tf.logging.info("lexicon loaded!")

        # load the glove mat
        glove_dict_file = model_hparams.glove_dict_file
        tf.logging.info("loading the glove mat from: "+glove_dict_file+" ...")
        glove_dict = vocab.load_glove_dict(glove_dict_file)
        self.glove_dict = glove_dict
        tf.logging.info("glove mat loaded!")

        #print(model_hparams.values())
        #print(model_hparams.problems[0].values())
        #print(os.path.join(model_hparams.data_dir, model_hparams.problem_instances[0].vocab_file))
        #input("hah")
        ## source and target vocab lists
        source_vocab_file = os.path.join(model_hparams.data_dir, model_hparams.problem_instances[0].vocab_file+".vie")
        target_vocab_file = os.path.join(model_hparams.data_dir, model_hparams.problem_instances[0].vocab_file+".eng")
        print("reading source vocab from: "+source_vocab_file)
        source_vocab = read_vocab(source_vocab_file)
        self.source_vocab_size = len(source_vocab)
        print("source_vocab_size: "+str(self.source_vocab_size)) # 8003
        print("reading target vocab from: "+target_vocab_file)
        target_vocab = read_vocab(target_vocab_file)
        self.target_vocab_size = len(target_vocab)
        print("target_vocab_size: "+str(self.target_vocab_size)) # 8003
        #source_vocab = vocab.read_vocab(model_hparams.src_vocab)
        #target_vocab = vocab.read_vocab(model_hparams.tgt_vocab)

        # get src_emb_table (src_vocab_size * glove_dim * lex_cap)
        src_to_lex_table = [pad_to_lex_cap(lexicon_dict[src_word], model_hparams.lex_cap) if src_word in lexicon_dict else ["_not_in_lexicon"]*model_hparams.lex_cap for src_word in source_vocab]
        lex_to_emb_table = get_lex_to_emb_table(src_to_lex_table, glove_dict)
        src_emb_init = np.array(lex_to_emb_table)
        src_emb_table = tf.get_variable(name="W", initializer=tf.constant(src_emb_init, dtype=tf.float32), trainable=True)
        self.src_emb_table = src_emb_table

        # get tgt_emb_table (tgt_vocab_size * glove_dim)
        tgt_to_emb_table = [glove_dict[tgt_word] if tgt_word in glove_dict else np.random.normal(0, 0.1, (300)) for tgt_word in target_vocab]
        tgt_emb_init = np.array(tgt_to_emb_table)
        tgt_emb_table = tf.get_variable(name="WT", initializer=tf.constant(tgt_emb_init, dtype=tf.float32), trainable=True)
        self.tgt_emb_table = tgt_emb_table


    @property
    def name(self):
        return "lex_modality_%d_%d_%d" % (self.source_vocab_size, self.target_vocab_size, self._body_input_depth)

    @property
    def top_dimensionality(self):
        return self.target_vocab_size

    def _get_weights(vocab_size, self):
        '''
        create or get concantenated embedding or softmax variable
        return: a list of self._num_shards tensors
        '''
        num_shards = self._model_hparams.symbol_modality_num_shards
        shards = []
        for i in range(num_shards):
            shard_size = (vocab_size // num_shards) + (1 if i < vocab_size % num_shards else 0)
            emb_mat_init = tf.random_normal_initializer(0.0, self._body_input_depth**-0.5)
            emb_mat = tf.get_variable(
                "weights_%d" % i, 
                [shard_size, self._body_input_depth],
                initializer=emb_mat_init)
            shards.append(emb_mat)
        if num_shards == 1:
            ret = shards[0]
        else:
            ret = tf.concat(shards, 0)
        ret = expert_utils.convert_gradient_to_tensor(ret)
        return ret

    def _embed_src(self, x, emb_mat, name, reuse=None, to_squeeze=True):
        with tf.variable_scope(name, reuse=reuse):
            print(x.shape.as_list()) # ==> [None, None, 1, 1]
            if to_squeeze:
                x = tf.squeeze(x, axis=3)
            print(x.shape.as_list()) # ==> [None, None, 1]
            emb_mat = expert_utils.convert_gradient_to_tensor(emb_mat)
            ret = tf.gather(emb_mat, x)
            print(ret.shape.as_list()) # ==> [None, None, 1, 4, 300]
            if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
                ret *= self._body_input_depth**0.5
            if to_squeeze:
                pads = tf.expand_dims(tf.to_float(tf.not_equal(x,0)), -1)
                pads = tf.expand_dims(pads, -1)
                print(pads.shape.as_list()) # ==> [None, None, 1, 1, 1]
                ret *= pads
            print(ret.shape.as_list()) # ==> [None, None, 1, 4, 300]
            return ret

    def _embed_tgt(self, y, emb_mat, name, reuse=None, to_squeeze=True):
        with tf.variable_scope(name, reuse=reuse):
            if to_squeeze:
                y = tf.squeeze(y, axis=3)
            emb_mat = expert_utils.convert_gradient_to_tensor(emb_mat)
            ret = tf.gather(amb_mat, y)
            if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
                ret *= self._body_input_depth**0.5
            if to_squeeze:
                ret *= tf.expand_dims(tf.to_float(tf.not_equal(y, 0)), -1)
            return ret

    def bottom(self, x):
        print(x.shape.as_list()) # ==> [None, None, 1, 1]
        res = self._embed_src(x, self.src_emb_table, "input_emb", reuse=None, to_squeeze=True)
        print(res.shape.as_list()) # ==> [None, None, None, 4, 300]
        return res

    def targets_bottom(self, y):
        print(y.shape.as_list())
        #input("targets_bottom")
        return self._embed_tgt(y, self.tgt_emb_table, "target_emb", reuse=None, to_squeeze=True)

    def top(self, body_output, _):
        '''
        input: [batch, p0, p1, body_input_depth]
        output: [batch, p0, p1, ?, vocab_size]
        '''
        with tf.variable_scope("softmax", reuse=None):
            var = self._get_weights(self.target_vocab_size)
            print(var.shape.as_list())
            input("var")
            if self._model_hparams.factored_logits and (self._model_hparams.mode == tf.estimator.ModeKeys.TRAIN):
                body_output = tf.expand_dims(body_output, 3)
                logits = common_layers.FactoredTensor(body_output, var)
            else:
                shape = tf.shape(body_output)[:-1]
                body_output = tf.reshape(body_output, [-1, self._body_input_depth])
                logits = tf.matmul(body_output, var, transpose_b=True)
                logits = tf.reshape(logits, tf.concat([shape, [1, self.target_vocab_size]], 0))
            return logits
 
