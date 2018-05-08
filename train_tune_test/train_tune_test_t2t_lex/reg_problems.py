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

FLAGS = tf.flags.FLAGS
EOS = text_encoder.EOS_ID

train_src="/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/train/src_raw.vie.train.y1r1.v2"
train_tgt="/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/train/ref_raw.eng.train.y1r1.v2"
dev_src="/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/dev/src_raw.vie.dev.y1r1.v2"
dev_tgt="/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/dev/ref_raw.eng.dev.y1r1.v2"

def generate_bpe_vocab(file_list, targeted_vocab_size):
    def generator_fn():
        for filepath in file_list:
            with tf.gfile.GFile(filepath, mode="r") as source_file:
                #file_byte_budget = 3.5e5 if filepath.endswith("en") else 7e5
                for line in source_file:
                    #if file_byte_budget <= 0:
                    #    break
                    line = line.strip()
                    #file_byte_budget -= len(line)
                    yield line
    token_counts = defaultdict(int)
    for item in generator_fn():
        for tok in tokenizer.encode(text_encoder.native_to_unicode(item)):
            token_counts[tok] += 1
    vocab = text_encoder.SubwordTextEncoder.build_to_target_size(targeted_vocab_size, token_counts, 1, 1e3)

    return vocab

def generate_tok_vocab(file_list, targeted_vocab_size):
    if len(file_list) > 1:
        raise Exception("A shared regular vocab of two languages is not supported (and possibly doesn't make much sense).")
    if len(file_list) == 0:
        raise Exception("No file given.")
    filepath = file_list[0]
    with tf.gfile.GFile(filepath, mode="r") as source_file:
        if sys.version_info[0] >= 3:
            data = source_file.read().replace("\n", " ").split()
        else:
            data = source_file.read().decode("utf-8").replace("\n", " ").split()
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    words = words[:targeted_vocab_size]
    words += ("UNK",)
    vocab = text_encoder.TokenTextEncoder(None, vocab_list=words)
    return vocab

# only apply on training data, on both the source and target sides
# based on generator_utils.get_or_generate_vocab
def get_vocab(file_list, vocab_filepath, use_subword_tokenizer, targeted_vocab_size):
    if tf.gfile.Exists(vocab_filepath):
        tf.logging.info("Found vocab file: %s", vocab_filepath)
        if use_subword_tokenizer:
            vocab = text_encoder.SubwordTextEncoder(vocab_filepath)
        else:
            vocab = text_encoder.TokenTextEncoder(vocab_filepath)
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

@registry.register_problem
class TranslateIl3engLrlp8000bpe(wmt.TranslateProblem):
    @property
    def input_space_id(self):
        problem.SpaceID.IL3_BPE_TOK = 29
        return problem.SpaceID.IL3_BPE_TOK
    @property
    def target_space_id(self):
        problem.SpaceID.ENG_BPE_TOK = problem.SpaceID.EN_BPE_TOK
        return problem.SpaceID.ENG_BPE_TOK

    @property
    def is_character_level(self):
        return False
    @property
    def use_subword_tokenizer(self):
        return True

    @property
    def num_shards(self):
        return 10
    @property
    def vocab_name(self):
        return "vocab.vieeng"
    @property
    def targeted_vocab_size(self):
        return 8000
    @property
    def vocab_file(self):
        return "%s.%d" % (self.vocab_name, self.targeted_vocab_size)

    # don't have to use tmp_dir in generator() because generate_data() (which is the function actually being called) doesn't use tmp_dir outside of generator().
    def generator(self, data_dir, tmp_dir, is_training_set):
        source_file = dev_src
        target_file = dev_tgt
        if is_training_set:
            source_file = train_src
            target_file = train_tgt

        vocab_obj = get_vocab([train_src, train_tgt], os.path.join(data_dir, self.vocab_file), self.use_subword_tokenizer, self.targeted_vocab_size)
        if not self.use_subword_tokenizer:
            source_file_with_unk = source_file+".vie-eng."+str(self.targeted_vocab_size)
            target_file_with_unk = target_file+".vie-eng."+str(self.targeted_vocab_size)
            replace_oov_with_unk(source_file, source_file_with_unk, vocab_obj._token_to_id)
            replace_oov_with_unk(target_file, target_file_with_unk, vocab_obj._token_to_id)
            source_file = source_file_with_unk
            target_file = target_file_with_unk
        return wmt.token_generator(source_file, target_file, vocab_obj, EOS)

    #def hparams(self, defaults, unused_model_hparams):
    #    p = defaults

    #    if self.has_inputs:
    #        source_vocab_size = self._encoders["inputs"].vocab_size
    #        p.input_modality = {"inputs":(registry.Modalities.SYMBOL, source_vocab_size)}
    #        p.input_space_id = self.input_space_id
    #    target_vocab_size = self._encoders["targets"].vocab_size
    #    p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
    #    p.target_space_id = self.target_space_id


@registry.register_problem
class TranslateVieengLrlp8000bpe(TranslateIl3engLrlp8000bpe):
    @property
    def input_space_id(self):
        problem.SpaceID.VIE_BPE_TOK = 30
        return problem.SpaceID.VIE_BPE_TOK
        
@registry.register_problem
class TranslateIl3engLrlp32000bpe(TranslateIl3engLrlp8000bpe):
    @property
    def targeted_vocab_size(self):
        return 32000

@registry.register_problem
class TranslateVieengLrlp32000bpe(TranslateIl3engLrlp32000bpe):
    @property
    def input_space_id(self):
        problem.SpaceID.VIE_BPE_TOK = 30
        return problem.SpaceID.VIE_BPE_TOK

@registry.register_problem
class TranslateIl3engLrlp80008000bpe(TranslateIl3engLrlp8000bpe):
    @property
    def use_subword_tokenizer(self):
        return True
    
    def generator(self, data_dir, tmp_dir, is_training_set):
        source_file = dev_src
        target_file = dev_tgt
        if is_training_set:
            source_file = train_src
            target_file = train_tgt
            
        src_vocab_obj = get_vocab([train_src], os.path.join(data_dir, self.vocab_file+".vie"), self.use_subword_tokenizer, self.targeted_vocab_size)
        tgt_vocab_obj = get_vocab([train_tgt], os.path.join(data_dir, self.vocab_file+".eng"), self.use_subword_tokenizer, self.targeted_vocab_size)
        if not self.use_subword_tokenizer:
           source_file_with_unk = source_file+".vie."+str(self.targeted_vocab_size)
           target_file_with_unk = target_file+".eng."+str(self.targeted_vocab_size)
           replace_oov_with_unk(source_file, source_file_with_unk, src_vocab_obj._token_to_id)
           replace_oov_with_unk(target_file, target_file_with_unk, tgt_vocab_obj._token_to_id)
           source_file = source_file_with_unk
           target_file = target_file_with_unk
        return wmt.bi_vocabs_token_generator(source_file, target_file, src_vocab_obj, tgt_vocab_obj, EOS)

    def feature_encoders(self, data_dir):
        if self.use_subword_tokenizer:
            src_vocab = text_encoder.SubwordTextEncoder(os.path.join(data_dir, self.vocab_file+".vie"))
            tgt_vocab = text_encoder.SubwordTextEncoder(os.path.join(data_dir, self.vocab_file+".eng"))
        else:
            src_vocab = text_encoder.TokenTextEncoder(os.path.join(data_dir, self.vocab_file+".vie"))
            tgt_vocab = text_encoder.TokenTextEncoder(os.path.join(data_dir, self.vocab_file+".eng"))
        return {
            "inputs": src_vocab,
            "targets": tgt_vocab,
        }

@registry.register_problem
class TranslateVieengLrlp80008000bpe(TranslateIl3engLrlp80008000bpe):
    @property
    def input_space_id(self):
        problem.SpaceID.VIE_BPE_TOK = 30
        return problem.SpaceID.VIE_BPE_TOK

@registry.register_problem
class TranslateIl3engLrlp80008000(TranslateIl3engLrlp80008000bpe):
    @property
    def use_subword_tokenizer(self):
        return False
    @property
    def input_space_id(self):
        problem.SpaceID.IL3_TOK = 31
        return problem.SpaceID.IL3_TOK
    @property
    def target_space_id(self):
        problem.SpaceID.ENG_TOK = problem.SpaceID.EN_TOK
        return problem.SpaceID.ENG_TOK

@registry.register_problem
class TranslateVieengLrlp80008000(TranslateIl3engLrlp80008000):
    @property
    def input_space_id(self):
        problem.SpaceID.VIE_TOK = 32
        return problem.SpaceID.VIE_TOK

