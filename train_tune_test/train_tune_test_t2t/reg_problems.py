from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import wmt
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.data_generators import generator_utils
import tensorflow as tf
from collections import defaultdict
import yaml


FLAGS = tf.flags.FLAGS
EOS = text_encoder.EOS_ID

dir_path = os.path.dirname(os.path.realpath(__file__))
config_file = os.path.join(dir_path, "config.yml")
config = yaml.load(open(config_file))
s = config["s"]
t = config["t"]
st = config["st"]
train_src = config["train_src"]
dev_src = config["dev_src"]
train_tgt = config["train_tgt"]
dev_tgt = config["dev_tgt"]
_share_vocab = True if config["share_vocab"].lower()=="true" else False
_use_subword_tokenizer = True if config["use_subword_tokenizer"].lower()=="true" else False
_vocab_size = int(config["vocab_size"])

# specify problem.SpaceID value for each language programmatically
langs = ["amh", "ben", "hau", "som", "il3", "vie", "yor", "eng"]
assert(s in langs)
assert(t in langs)
problem.SpaceID.ENG_TOK = problem.SpaceID.EN_TOK
problem.SpaceID.ENG_BPE_TOK = problem.SpaceID.EN_BPE_TOK
id_names = [attr for attr, value in vars(problem.SpaceID).items() if not attr.startswith("__")]
id_values = [value for attr, value in vars(problem.SpaceID).items() if not attr.startswith("__")]
id_cur = int(max(id_values)+1)
for lang in langs:
    bpe_tok = lang.upper()+"_BPE_TOK"
    if bpe_tok not in id_names:
        setattr(problem.SpaceID, bpe_tok, id_cur)
        id_cur += 1
    tok = lang.upper()+"_TOK"
    if tok not in id_names:
        setattr(problem.SpaceID, tok, id_cur)
        id_cur += 1

def generator_fn(file_list):
    for filepath in file_list:
        with tf.gfile.GFile(filepath, mode="r") as source_file:
            #file_byte_budget = 3.5e5 if filepath.endswith("en") else 7e5
            for line in source_file:
                #if file_byte_budget <= 0:
                #    break
                line = line.strip()
                #file_byte_budget -= len(line)
                yield line

def generate_bpe_vocab(file_list, targeted_vocab_size):
    token_counts = defaultdict(int)
    for item in generator_fn(file_list):
        for tok in tokenizer.encode(text_encoder.native_to_unicode(item)):
            token_counts[tok] += 1
    vocab = text_encoder.SubwordTextEncoder.build_to_target_size(
        targeted_vocab_size, 
        token_counts, 
        1, 
        1e3)
    return vocab

def generate_tok_vocab(file_list, targeted_vocab_size):
    #if len(file_list) > 1:
    #    raise Exception("A shared regular vocab of two languages is not supported (and possibly doesn't make much sense).")
    #if len(file_list) == 0:
    #    raise Exception("No file given.")
    #filepath = file_list[0]
    #with tf.gfile.GFile(filepath, mode="r") as source_file:
    #    if sys.version_info[0] >= 3:
    #        data = source_file.read().replace("\n", " ").split()
    #    else:
    #        data = source_file.read().decode("utf-8").replace("\n", " ").split()
    data = []
    for line in generator_fn(file_list):
        for tok in line.split(" "):
            data.append(tok)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    words = words[:targeted_vocab_size]
    words += ("UNK",)
    vocab = text_encoder.TokenTextEncoder(None, vocab_list=words, replace_oov="UNK")
    return vocab

# only apply on training data, on both the source and target sides
# based on generator_utils.get_or_generate_vocab
def get_vocab(file_list, vocab_filepath, use_subword_tokenizer, targeted_vocab_size):
    if tf.gfile.Exists(vocab_filepath):
        tf.logging.info("Found vocab file: %s", vocab_filepath)
        if use_subword_tokenizer:
            vocab = text_encoder.SubwordTextEncoder(vocab_filepath)
        else:
            vocab = text_encoder.TokenTextEncoder(vocab_filepath, replace_oov="UNK")
        return vocab

    tf.logging.info("Generating vocab file: %s", vocab_filepath)
    if use_subword_tokenizer:
        vocab = generate_bpe_vocab(file_list, targeted_vocab_size)
    else:
        vocab = generate_tok_vocab(file_list, targeted_vocab_size)
    vocab.store_to_file(vocab_filepath)
    return vocab

def replace_oov_with_unk(infile, outfile, token_to_id_dict):
    with open(infile) as f, open(outfile, 'w') as fw:
        for line in f:
            l = line.strip().split(' ')
            l_new = [(tok if tok in token_to_id_dict else "UNK") for tok in l]
            fw.write(' '.join(l_new)+"\n")

# params:
# share_vocab: True or False
# targeted_vocab_size: e.g. 8000, 32000
# use_subword_tokenizer: True or False
@registry.register_problem
class TranslateSrctgtLrlp(wmt.TranslateProblem):
    @property
    def input_space_id(self):
        return getattr(problem.SpaceID, s.upper()+"_BPE_TOK")
    @property
    def target_space_id(self):
        return getattr(problem.SpaceID, t.upper()+"_BPE_TOK")
    @property
    def is_character_level(self):
        return False
    @property
    def use_subword_tokenizer(self):
        return _use_subword_tokenizer
    @property
    def num_shards(self):
        return 10
    @property
    def vocab_name(self):
        return "vocab."+s+t
    @property
    def targeted_vocab_size(self):
        return _vocab_size
    @property
    def vocab_file(self):
        return "%s.%d" % (self.vocab_name, self.targeted_vocab_size)
    @property
    def has_inputs(self):
        return True # set to False for language models

    # don't have to use tmp_dir in generator() because generate_data() (which is the function actually being called) doesn't use tmp_dir outside of generator().
    def generator(self, data_dir, tmp_dir, is_training_set):
        source_file = dev_src
        target_file = dev_tgt
        if is_training_set:
            source_file = train_src
            target_file = train_tgt

        if _share_vocab:
            vocab_obj = get_vocab([train_src, train_tgt], os.path.join(data_dir, self.vocab_file), self.use_subword_tokenizer, self.targeted_vocab_size)
            src_vocab_obj = tgt_vocab_obj = vocab_obj
        else:
            src_vocab_obj = get_vocab([train_src], os.path.join(data_dir, self.vocab_file+"."+s), self.use_subword_tokenizer, self.targeted_vocab_size)
            tgt_vocab_obj = get_vocab([train_tgt], os.path.join(data_dir, self.vocab_file+"."+t), self.use_subword_tokenizer, self.targeted_vocab_size)
        #if not self.use_subword_tokenizer:
        #    source_file_with_unk = source_file+"."+st+"."+str(self.targeted_vocab_size)
        #    target_file_with_unk = target_file+"."+st+"."+str(self.targeted_vocab_size)
        #    replace_oov_with_unk(source_file, source_file_with_unk, vocab_obj._token_to_id)
        #    replace_oov_with_unk(target_file, target_file_with_unk, vocab_obj._token_to_id)
        #    source_file = source_file_with_unk
        #    target_file = target_file_with_unk
        #return wmt.token_generator(source_file, target_file, vocab_obj, EOS)
        return wmt.bi_vocabs_token_generator(source_file, target_file, src_vocab_obj, tgt_vocab_obj, EOS)

    def feature_encoders(self, data_dir):
        if self.use_subword_tokenizer:
            if _share_vocab:
                src_vocab = tgt_vocab = text_encoder.SubwordTextEncoder(os.path.join(data_dir, self.vocab_file))
            else:
                src_vocab = text_encoder.SubwordTextEncoder(os.path.join(data_dir, self.vocab_file+"."+s))
                tgt_vocab = text_encoder.SubwordTextEncoder(os.path.join(data_dir, self.vocab_file+"."+t))
        else:
            if _share_vocab:
                src_vocab = tgt_vocab = text_encoder.TokenTextEncoder(os.path.join(data_dir, self.vocab_file), replace_oov="UNK")
            else:
                src_vocab = text_encoder.TokenTextEncoder(os.path.join(data_dir, self.vocab_file+"."+s), replace_oov="UNK")
                tgt_vocab = text_encoder.TokenTextEncoder(os.path.join(data_dir, self.vocab_file+"."+t), replace_oov="UNK")
        return {
            "inputs": src_vocab,
            "targets": tgt_vocab,
        }

# ----

#@registry.register_problem
#class TranslateSrctgtLrlp32000bpe(TranslateSrctgtLrlp8000bpe):
#    @property
#    def targeted_vocab_size(self):
#        return 32000
#
## ----
#
#@registry.register_problem
#class TranslateSrctgtLrlp80008000bpe(TranslateSrctgtLrlp8000bpe):
#    @property
#    def use_subword_tokenizer(self):
#        return True
#    
#    def generator(self, data_dir, tmp_dir, is_training_set):
#        source_file = dev_src
#        target_file = dev_tgt
#        if is_training_set:
#            source_file = train_src
#            target_file = train_tgt
#            
#        src_vocab_obj = get_vocab([train_src], os.path.join(data_dir, self.vocab_file+"."+s), self.use_subword_tokenizer, self.targeted_vocab_size)
#        tgt_vocab_obj = get_vocab([train_tgt], os.path.join(data_dir, self.vocab_file+"."+t), self.use_subword_tokenizer, self.targeted_vocab_size)
#        if not self.use_subword_tokenizer:
#           source_file_with_unk = source_file+"."+s+"."+str(self.targeted_vocab_size)
#           target_file_with_unk = target_file+"."+t+"."+str(self.targeted_vocab_size)
#           replace_oov_with_unk(source_file, source_file_with_unk, src_vocab_obj._token_to_id)
#           replace_oov_with_unk(target_file, target_file_with_unk, tgt_vocab_obj._token_to_id)
#           source_file = source_file_with_unk
#           target_file = target_file_with_unk
#        return wmt.bi_vocabs_token_generator(source_file, target_file, src_vocab_obj, tgt_vocab_obj, EOS)
#
#
## ----
#
#@registry.register_problem
#class TranslateSrctgtLrlp80008000(TranslateSrctgtLrlp80008000bpe):
#    @property
#    def use_subword_tokenizer(self):
#        return False







# generate_data(data_dir, tmp_dir)
# generate training and dev sets into data_dir
# additional files, e.g. vocabulary files, should also be written to data_dir

# downloads and other files can be written to tmp_dir
# if you have a training and dev generator, you can generate the training and dev sets with generator_utils.generate_dataset_and_shuffle

# use the self.training_filepaths and self.dev_filepaths functions to get sharded filesnames. If shuffled=False, the filenames will contain an "unshuffled" suffix; you should then shuffle the data shard-byshard with generator_utils.shuffle_dataset.

#class Problem
#def generate_data(data_dir, tmp_dir, task_id=-1): not implemented
#def hparams
#def dataset_filenames
#def feature_encoders
#def example_reading_spec
#def preprocess_examples
#def eval_metrics
#def training_filepaths(data_dir, num_shards, shuffled)
#def dev_filepaths(data_dir, num_shards, shuffled)
#def test_filepaths(data_dir, num_shards, shuffled)
#def __init__
#def internal_build_encoders
#def internal_hparams
#def maybe_reverse_features
#def maybe_copy_features
#
#class Text2TextProblem(Problem)
#def @property is_character_level: not implemented
#def @property targeted_vocab_size: not implemented
#def @property use_train_shards_for_dev
#def @property input_space_id: not implemented
#def @property target_space_id: not implemented
#def @property num_shards: not implemented
#def @property num_dev_shards
#def @property vocab_name: not implemented
#def @property vocab_file
#def @property use_subword_tokenizer: not implemented
#def @property has_inputs
#def generator(data_dir, tmp_dir, is_training): not implemented
#def generate_data(data_dir, tmp_dir, task_id=-1)
#def feature_encoders(data_dir)
#def hparams(defaults, unused_model_hparams)
#def eval_metrics()
#
#class TranslateProblem
#def @property is_character_level
#def @property num_shards
#def @property vocab_name
#def @property use_subword_tokenizer

