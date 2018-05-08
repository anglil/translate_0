from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections import OrderedDict
import os
import sys 
import numpy as np
import random
import yaml
from six.moves import xrange
import string

from tensor2tensor.utils import expert_utils
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.utils import modality
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.data_generators import generator_utils
import tensorflow as tf
from collections import defaultdict
from seq2seq.data import vocab




dir_path = os.path.dirname(os.path.realpath(__file__))
config_file = os.path.join(dir_path, "config.yml")
config = yaml.load(open(config_file))
s = config["s"]
t = config["t"]
emb_trainable = False if "embuntrainable" in config["model_params"] else True
emb_random = True if "embrandom" in config["model_params"] else False
lex_cluster = True if "lexcluster" in config["model_params"] else False
use_align = True if "usealign" in config["model_params"] else False
method = config['model_params']
model_name = config["model_params"].split("_")[0]
is_simple_model = (model_name=="t2tsimple")
train_src = config["train_src"]
train_tgt = config["train_tgt"]

synonym_api = config['synonym_api']
sys.path.insert(0, synonym_api)
from thesaurus import Word

nltk_data_dir = config['synonym_api2']
import nltk
nltk.data.path.append(nltk_data_dir)
from nltk.corpus import wordnet as wn


train_tgt_token_set = set()
with open(train_tgt) as f:
    for l in f:
        l = l.strip().split(' ')
        for tok in l:
            train_tgt_token_set.add(tok)

punctuation_set = set(string.punctuation)

def is_punc(mystr):
    res = True
    for c in mystr:
        if c not in punctuation_set:
            res = False
            break
    return res

def get_alignment_dict_ordered():
    alignment_dict = dict()
    alignment_file = os.path.abspath(os.path.join(train_src, "..", "..", "..", "model", "train", "model", "aligned.grow-diag-final-and"))
    print("alignment_file: "+alignment_file)
    with open(train_src) as f_src, open(train_tgt) as f_tgt, open(alignment_file) as f_a:
        for l_src in f_src:
            l_src = l_src.strip().split(' ')
            l_tgt = f_tgt.readline().strip().split(' ')
            l_a = f_a.readline().strip().split(' ')
            for pair in l_a:
                pos_src = int(pair.split('-')[0])
                pos_tgt = int(pair.split('-')[1])
                assert(pos_src<len(l_src))
                assert(pos_tgt<len(l_tgt))
                src_word = l_src[pos_src].lower() # src_word in lower case
                tgt_word = l_tgt[pos_tgt].lower() # tgt_word in lower case
    
                if is_punc(src_word):
                    alignment_dict[src_word] = {src_word:1}
                elif src_word not in alignment_dict:
                    alignment_dict[src_word] = {tgt_word:1}
                elif tgt_word in alignment_dict[src_word]:
                    alignment_dict[src_word][tgt_word] += 1
                else:
                    alignment_dict[src_word][tgt_word] = 1
    alignment_dict_ordered = dict()
    for k,v in alignment_dict.items():
        alignment_dict_ordered[k] = OrderedDict(sorted(v.items(),key=lambda t:-t[1]))
        alignment_dict_ordered[k] = list(alignment_dict_ordered[k].keys())
    return alignment_dict_ordered


def get_synonyms(word, src="wordnet"):
    synonyms = set()
    if src == "wordnet":
        for ss in wn.synsets(word):
            synonyms |= set(ss.lemma_names())
    elif src == "thesaurus":
        try:
            w = Word(word)
        except:
            return synonyms
        try:
            syn = w.synonyms(relevance=[2,3])
        except:
            return synonyms
        for s in syn:
            if len(s.split(' ')) == 1:
                synonyms.add(s.lower())
    return synonyms

def get_leader(word, word_clusters):
    if "pointer" not in word_clusters[word]:
        return word
    return get_leader(word_clusters[word]["pointer"], word_clusters)

def cluster_words_by_synonym(word_list):
    word_clusters = {}
    for i in range(len(word_list)):
        word = word_list[i]
        word_clusters[word] = {"cluster":{word}, "synonyms":get_synonyms(word)}
    for i in range(len(word_list)-1):
        word = word_list[i]
        word_source = get_leader(word, word_clusters)
        for j in range(i+1, len(word_list)):
            word_next = word_list[j]
            word_next_source = get_leader(word_next, word_clusters)
            if word_next_source in word_clusters[word_source]["synonyms"] and word_next_source not in word_clusters[word_source]["cluster"]:
                word_clusters[word_source]["cluster"] |= word_clusters[word_next_source]["cluster"]
                word_clusters[word_source]["synonyms"] |= word_clusters[word_next_source]["synonyms"]
                word_clusters[word_next_source]["pointer"] = word_source
    res = []
    for word in word_clusters:
        if "pointer" not in word_clusters[word]:
            res.append(word_clusters[word]["cluster"])
    return res


def pad_to_lex_cap(lex_list, lex_cap):
    return np.tile(lex_list, int(lex_cap/len(lex_list))+1)[:lex_cap]


def get_translation_candidates_by_target(translation_candidates, train_tgt_token_set, lex_cap, in_effect=True):
    '''
    when using the lexicon for source token initialization, only consider translated source tokens that appear in the target text
    '''
    if len(translation_candidates) < lex_cap or not in_effect:
        return translation_candidates
    ret = []
    for c in translation_candidates:
        if c in train_tgt_token_set:
            ret.append(c)
            if len(ret) == lex_cap:
                return ret
    ret_set = set(ret)
    for c in translation_candidates:
        if c not in ret_set:
            ret.append(c)
            ret_set.add(c)
    return ret

def src_to_tgt(source_vocab, alignment_dict_ordered, lexicon_dict, lex_cap, use_align=False, lex_cluster=False, is_simple_model=False):
    ret = []
    for src_word in source_vocab:
        src_word = src_word.lower()
        # use alignment
        if use_align:
            if src_word in alignment_dict_ordered:
                tgt_list = alignment_dict_ordered[src_word]
            elif src_word in lexicon_dict:
                tgt_list = get_translation_candidates_by_target(lexicon_dict[src_word], train_tgt_token_set, lex_cap)
            else:
                tgt_list = ["unk"]
        # use lexicon
        else:
            if src_word in lexicon_dict:
                tgt_list = get_translation_candidates_by_target(lexicon_dict[src_word], train_tgt_token_set, lex_cap)
            else:
                tgt_list = ["unk"]
        # take just one translation/alignment
        if is_simple_model:
            tgt_ret = tgt_list[0]
        # cluster by synonyms
        elif lex_cluster:
            tgt_ret = pad_to_lex_cap(cluster_words_by_synonym(tgt_list), lex_cap)
        # no clustering
        else:
            tgt_ret = pad_to_lex_cap(tgt_list, lex_cap)
        ret.append(tgt_ret)
    return ret

def tgt_to_emb(src_to_tgt_table, glove_dict, lex_cluster=False, is_simple_model=False):
    if is_simple_model:
        assert(lex_cluster==False)
        tgt_to_emb_table = [glove_dict[tgt_word] if tgt_word in glove_dict else np.random.normal(0, 0.1, (300)) for tgt_word in src_to_tgt_table]
    elif lex_cluster:
        tgt_to_emb_table = get_tgtcluster_to_emb_table(src_to_tgt_table, glove_dict)
    else:
        tgt_to_emb_table = get_tgt_to_emb_table(src_to_tgt_table, glove_dict)
    return tgt_to_emb_table


def get_tgt_to_emb_table(lex_list_list, emb_dict):
    emb_table = []
    for lex_list in lex_list_list:
        emb_list = [emb_dict[lex] if lex in emb_dict else np.random.normal(0, 0.1, (300)) for lex in lex_list]
        emb_table.append(np.array(emb_list, dtype=float))
    return emb_table

def get_tgtcluster_to_emb_table(lexcluster_list_list, emb_dict):
    print("lexcluster_list_list shape: "+str((np.array(lexcluster_list_list)).shape))
    emb_table = []
    ctr = 0
    ctr_no_emb_from_lex = 0
    ctr_no_emb_from_emb = 0
    for lexcluster_list in lexcluster_list_list:
        lex_cap = len(lexcluster_list)
        if lexcluster_list[0] == "_not_in_lexicon":
            emb_list = np.random.normal(0, 0.1, (lex_cap, 300))
            #assert((np.array(emb_list)).shape==(4,300))
            ctr_no_emb_from_lex += 1
        else:
            emb_list = []
            for lexcluster in lexcluster_list:
                emb_array = []
                for lex in lexcluster:
                    if lex in emb_dict:
                        emb_array.append(np.array(emb_dict[lex]))
                if len(emb_array) != 0:
                    emb_list.append(np.mean(np.array(emb_array), 0))
            if len(emb_list) != lex_cap:
                if len(emb_list) != 0:
                    emb_list = np.tile(emb_list, (int(lex_cap/len(emb_list))+1, 1))[:lex_cap]
                else:
                    emb_list = np.random.normal(0, 0.1, (lex_cap, 300))
                    #assert((np.array(emb_list)).shape==(4,300))
                    ctr_no_emb_from_emb += 1
            #else:
                #assert((np.array(emb_list)).shape==(4,300))
        emb_table.append(np.array(emb_list, dtype=float))
        ctr += 1
    print("ctr: %d" % ctr)
    print("ctr_no_emb_from_lex: %d" % ctr_no_emb_from_lex)
    print("ctr_no_emb_from_emb: %d" % ctr_no_emb_from_emb)
    print("emb_table shape: "+str((np.array(emb_table)).shape))
    return emb_table
                

def read_vocab(vocab_file):
    vocab = []
    with open(vocab_file) as f:
        for l in f:
            vocab.append(l.strip())
    return vocab

def read_vocab2(model_hparams):
    _share_vocab = True if config["share_vocab"].lower()=="true" else False
    _use_subword_tokenizer = True if config["use_subword_tokenizer"].lower()=="true" else False
    data_dir = model_hparams.data_dir
    vocab_file = model_hparams.problem_instances[0].vocab_file
    if _use_subword_tokenizer:
        if _share_vocab:
            src_vocab = tgt_vocab = text_encoder.SubwordTextEncoder(os.path.join(data_dir, vocab_file))
        else:
            src_vocab = text_encoder.SubwordTextEncoder(os.path.join(data_dir, vocab_file+"."+s))
            tgt_vocab = text_encoder.SubwordTextEncoder(os.path.join(data_dir, vocab_file+"."+t))
    else:
        if _share_vocab:
            src_vocab = tgt_vocab = text_encoder.TokenTextEncoder(os.path.join(data_dir, vocab_file), replace_oov="UNK")
        else:
            src_vocab = text_encoder.TokenTextEncoder(os.path.join(data_dir, vocab_file+"."+s), replace_oov="UNK")
            tgt_vocab = text_encoder.TokenTextEncoder(os.path.join(data_dir, vocab_file+"."+t), replace_oov="UNK")
    src_v = collections.OrderedDict(sorted(src_vocab._id_to_token.items(),key=lambda t:t[0]))
    tgt_v = collections.OrderedDict(sorted(tgt_vocab._id_to_token.items(),key=lambda t:t[0]))
    return [v for k,v in src_v.items()], [v for k,v in tgt_v.items()]


@registry.register_symbol_modality("lex_modality")
class LexModality(modality.Modality):

    def __init__(self, model_hparams, vocab_size=None):
        super(LexModality, self).__init__(model_hparams, vocab_size)

        ## source and target vocab lists
        #source_vocab_file = os.path.join(model_hparams.data_dir, model_hparams.problem_instances[0].vocab_file+"."+s)
        #target_vocab_file = os.path.join(model_hparams.data_dir, model_hparams.problem_instances[0].vocab_file+"."+t)
        #print("reading source vocab from: "+source_vocab_file)

        #source_vocab = read_vocab(source_vocab_file)
        #self.source_vocab_size = len(source_vocab)
        #print("source_vocab_size: "+str(self.source_vocab_size)) # 8003
        #print("reading target vocab from: "+target_vocab_file)
        #target_vocab = read_vocab(target_vocab_file)
        #self.target_vocab_size = len(target_vocab)
        #print("target_vocab_size: "+str(self.target_vocab_size)) # 8003



        source_vocab, target_vocab = read_vocab2(model_hparams)
        self.source_vocab_size = len(source_vocab)
        self.target_vocab_size = len(target_vocab)

        model_dir = os.path.join(model_hparams.data_dir, '..', '..', method)
        ckpt_file = os.path.join(model_dir, 'checkpoint')
        print("tensorboard --port 6006 --logdir="+model_dir)
        print("ckpt_file: "+ckpt_file)

        if emb_random or (os.path.exists(ckpt_file) and (not emb_random)):
            if is_simple_model:
                self.src_emb_init = np.random.rand(self.source_vocab_size,300)
            else:
                self.src_emb_init = np.random.rand(self.source_vocab_size,model_hparams.lex_cap,300)
            self.tgt_emb_init = np.random.rand(self.target_vocab_size,300)
        else:
            # load the lexicon
            if tf.get_collection("lexicon_dict") == []:
                lexicon_dict_file = model_hparams.bilingual_lexicon
                tf.logging.info("loading the lexicon from: "+lexicon_dict_file+" ...")
                lexicon_dict = vocab.get_lexicon_dict(lexicon_dict_file)
                tf.add_to_collection("lexicon_dict", lexicon_dict)
                tf.logging.info("lexicon loaded!")
            else:
                tf.logging.info("lexicon_dict exists in graph collections")

            # load the glove mat
            if tf.get_collection("glove_dict") == []:
                glove_dict_file = model_hparams.glove_mat
                tf.logging.info("loading the glove mat from: "+glove_dict_file+" ...")
                glove_dict = vocab.load_glove_dict(glove_dict_file)
                tf.add_to_collection("glove_dict", glove_dict)
                tf.logging.info("glove mat loaded!")
            else:
                tf.logging.info("glove_dict exists in graph collections")

            # get src_emb_table (src_vocab_size * lex_cap * glove_dim)
            # get tgt_emb_table (tgt_vocab_size * glove_dim)
            lexicon_dict = tf.get_collection("lexicon_dict")
            assert lexicon_dict != []
            lexicon_dict = lexicon_dict[0]

            glove_dict = tf.get_collection("glove_dict")
            assert glove_dict != []
            glove_dict = glove_dict[0]

            alignment_dict_ordered = get_alignment_dict_ordered()
                   
            src_to_tgt_table = src_to_tgt(source_vocab, alignment_dict_ordered, lexicon_dict, model_hparams.lex_cap, use_align, lex_cluster, is_simple_model)
            lex_to_emb_table = tgt_to_emb(src_to_tgt_table, glove_dict, lex_cluster, is_simple_model)

            self.src_emb_init = np.array(lex_to_emb_table)
            print("src_emb_init shape: "+str(self.src_emb_init.shape))

            tgt_to_emb_table = [glove_dict[tgt_word] if tgt_word in glove_dict else np.random.normal(0, 0.1, (300)) for tgt_word in target_vocab]
            self.tgt_emb_init = np.array(tgt_to_emb_table)
            print("tgt_emb_init shape: "+str(self.tgt_emb_init.shape))

    @property
    def name(self):
        return "lex_modality_%d_%d_%d" % (self.source_vocab_size, self.target_vocab_size, self._body_input_depth)

    @property
    def top_dimensionality(self):
        return self.target_vocab_size

    def _get_weights(self, src_mat_np, vocab_size):
        num_shards = self._model_hparams.symbol_modality_num_shards
        shards = []
        pos = 0
        for i in xrange(num_shards):
            shard_size = (vocab_size // num_shards) + (1 if i < vocab_size % num_shards else 0)
            var_name = "weights_%d" % i
            src_mat_np_shard = src_mat_np[pos:pos+shard_size]
            print(src_mat_np_shard.shape)
            raise Exception("gg")
            initializer = tf.constant(src_mat_np_shard, dtype=tf.float32)
            #initializer = lambda shape=[shard_size,src_mat_np.shape[1],300], dtype=tf.float32, partition_info=None: src_mat_np_shard
            pos += shard_size
            src_mat_shard = tf.get_variable(
                var_name,
                initializer=initializer,
                trainable=emb_trainable)
                #shape=[shard_size,src_mat_np.shape[1],300],
                #dtype=tf.float32,
            shards.append(src_mat_shard)
        if num_shards == 1:
            ret = shards[0]
        else:
            ret = tf.concat(shards, 0)
        ret = expert_utils.convert_gradient_to_tensor(ret)
        return ret

    def _get_weights_top(self, tgt_mat_np, vocab_size):
        num_shards = self._model_hparams.symbol_modality_num_shards
        shards = []
        pos = 0
        for i in xrange(num_shards):
            shard_size = (vocab_size // num_shards) + (1 if i < vocab_size % num_shards else 0)
            var_name = "weights_%d" % i
            tgt_mat_np_shard = tgt_mat_np[pos:pos+shard_size]
            #initializer = tf.constant(tgt_mat_np_shard, dtype=tf.float32)
            initializer = lambda shape=tgt_mat_np_shard.shape, dtype=tf.float32, partition_info=None: tgt_mat_np_shard
            pos += shard_size
            tgt_mat_shard = tf.get_variable(
                var_name,
                initializer=initializer,
                trainable=emb_trainable,
                shape=tgt_mat_np_shard.shape,
                dtype=tf.float32)
            shards.append(tgt_mat_shard)
        if num_shards == 1:
            ret = shards[0]
        else:
            ret = tf.concat(shards, 0)
        ret = expert_utils.convert_gradient_to_tensor(ret)
        return ret

    def bottom(self, x):
        with tf.variable_scope("input_emb", reuse=None):
            x = tf.squeeze(x, axis=3)

            #if is_simple_model:
            var = self._get_weights_top(self.src_emb_init, self.source_vocab_size)
            #else:
            #    var = self._get_weights(self.src_emb_init, self.source_vocab_size)
            ret = tf.gather(var, x)
            print("[bottom] input shape: "+str(x.shape.as_list())) #=>[None,None,1]
            print("[bottom] embedding matrix shape: "+str(var.shape.as_list())) #=>[8003,4,300]
            print("[bottom] output shape: "+str(ret.shape.as_list())) #=>[None,None,1,4,300]
            if self.src_emb_init.shape[-1]!=self._body_input_depth:
                ret = tf.layers.dense(ret, self._body_input_depth)
            
            if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
                ret *= self._body_input_depth**0.5

            if is_simple_model:
                ret *= tf.expand_dims(tf.to_float(tf.not_equal(x, 0)), -1)
            else:
                pads = tf.expand_dims(tf.to_float(tf.not_equal(x,0)), -1)
                pads = tf.expand_dims(pads, -1)
                ret *= pads
            return ret

    def _targets_bottom(self, y, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            y = tf.squeeze(y, axis=3)

            var = self._get_weights_top(self.tgt_emb_init, self.target_vocab_size)
            ret = tf.gather(var, y)
            print("[targets] input shape: "+str(y.shape.as_list())) #=>[None,None,1]
            print("[targets] embedding matrix shape: "+str(var.shape.as_list())) #=>[8003,300]
            print("[targets] output shape: "+str(ret.shape.as_list())) #=>[None,None,1,300]
            if self.tgt_emb_init.shape[-1]!=self._body_input_depth:
                ret = tf.layers.dense(ret, self._body_input_depth)

            if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
                ret *= self._body_input_depth**0.5

            ret *= tf.expand_dims(tf.to_float(tf.not_equal(y, 0)), -1)
            return ret

    def targets_bottom(self, y):
        if self._model_hparams.shared_embedding_and_softmax_weights: 
            try:
                return self._targets_bottom(y, "shared", reuse=True)
            except ValueError:
                return self._targets_bottom(y, "shared", reuse=None)
        else:
            return self._targets_bottom(y, "target_emb", reuse=None)

    def top(self, body_output, _):
        if self._model_hparams.shared_embedding_and_softmax_weights:
            scope_name = "shared"
            reuse = True
        else:
            scope_name = "softmax"
            reuse = False
        with tf.variable_scope(scope_name, reuse=reuse):
            var = self._get_weights_top(self.tgt_emb_init, self.target_vocab_size)
            if self.tgt_emb_init.shape[-1]!=self._body_input_depth:
                var = tf.layers.dense(var, self._body_input_depth)

            if self._model_hparams.factored_logits and (self._model_hparams.mode == tf.estimator.ModeKeys.TRAIN):
                body_output = tf.expand_dims(body_output, 3)
                logits = common_layers.FactoredTensor(body_output, var)
            else:
                shape = tf.shape(body_output)[:-1]
                body_output = tf.reshape(body_output, [-1, self._body_input_depth])
                logits = tf.matmul(body_output, var, transpose_b=True)
                logits = tf.reshape(logits, tf.concat([shape, [1, self.target_vocab_size]], 0))
            print("[top] input shape: "+str(body_output.shape.as_list())) #=>[None,300]
            print("[top] embedding matrix shape: "+str(var.shape.as_list())) #=>[8003,300]
            print("[top] output shape: "+str(logits.shape.as_list())) #=>[None,None,1,1,8003]
            return logits

