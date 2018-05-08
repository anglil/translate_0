from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer
import tensorflow as tf
import six
from six.moves import xrange
from itertools import chain
import collections

FLAGS = tf.flags.FLAGS
EOS = text_encoder.EOS_ID

# python version > 2

class SubwordTupleEncoder(text_encoder.TextEncoder):
    '''
    __init__ -> _load_from_file -> _load_from_file_object -> _init_subtokens_from_list, _init_alphabet_from_tokens
    build_to_target_size -> build_from_token_counts -> _init_subtokens_from_list, _init_alphabet_from_tokens, _escaped_token_to_subtoken_strings
    '''
    @property
    def vocab_size(self):
        return len(self._all_subtoken_strings)

    def __init__(self, filename=None):
        self._alphabet = set()
        if filename is not None:
            self._load_from_file(filename)
        super(SubwordTupleEncoder, self).__init__(num_reserved_ids=None)

    def _load_from_file_object(self, f):
        '''vocab file -> {subtoken:id}, {alphabet}'''
        subtoken_strings = []
        for line in f:
            s = line.strip()
            if ((s.startswith("'") and s.endswith("'")) or (s.startswith("\"") and s.endswith("\""))):
                s = s[1:-1]
            subtoken_strings.append(s)
        self._init_subtokens_from_list(subtoken_strings)
        self._init_alphabet_from_tokens(subtoken_strings)

    def _init_subtokens_from_list(self, subtoken_strings, num_reserved_ids=0):
        '''list of subtokens -> {subtoken:id}'''
        if num_reserved_ids == 0:
            self._all_subtoken_strings = subtoken_strings
        elif num_reserved_ids == text_encoder.NUM_RESERVED_TOKENS:
            self._all_subtoken_strings = text_encoder.RESERVED_TOKENS + subtoken_strings
        else:
            raise ValueError("Unexpected value for num_reserved_ids.")

        self._max_subtoken_len = max([len(s) for s in subtoken_strings])
        self._subtoken_string_to_id = {s:i+reserved for i,s in enumerate(subtoken_strings) if s}
        # initialize the cache
        self._cache_size = 2**20
        self._cache = [(None,None)]*self._cache_size

    def _init_alphabet_from_tokens(self, tokens):
        '''iterable of tokens/subtokens -> {alphabet}'''
        self._alphabet = {c for token in tokens for c in token}
        self._alphabet |= text_encoder._ESCAPE_CHARS

    def build_to_target_size(cls, target_size, token_counts, min_val, max_val, num_iterations=4):
        '''{token:count} -> vocab object, constrained by target_size'''
        if min_val > max_val:
            raise ValueError("(minimum token count) lower bound is greater than upper bound.")
        if target_size < 1:
            raise ValueError("target size must be positive.")

        def bisect(min_val, max_val):
            cur_min_count = (max_val+min_val)//2
            tf.logging.info("Trying min_count %d" % cur_min_count)
            subtokenizer = cls()
            subtokenizer.build_from_token_counts(token_counts, cur_min_count, num_iterations)
            if (abs(subtokenizer.vocab_size-target_size)/(target_size*1.0)<0.01) or (min_val>=max_val) or (cur_min_count<2):
                return subtokenizer
            if subtokenizer.vocab_size > target_size:
                other_subtokenizer = bisect(cur_min_count+1, max_val)
            else:
                other_subtokenizer = bisect(min_val, cur_min_count-1)
            if other_subtokenizer is None:
                return subtokenizer
            if abs(other_subtokenizer.vocab_size-target_size)<abs(subtokenizer.vocab_size-target_size):
                return other_subtokenizer
            return subtokenizer

        return bisect(min_val, max_val)

    def build_from_token_counts(self, token_counts, min_count, num_iterations=4, num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS):
        '''{token:count} -> vocab object, constrained by min_count'''
        if num_reserved_ids == 0:
            alphabet_tokens = six.iterkeys(token_counts)
        elif num_reserved_ids == text_encoder.NUM_RESERVED_TOKENS:
            alphabet_tokens = chain(six.iterkeys(token_counts), [t for t in text_encoder.RESERVED_TOKENS])
        else:
            raise ValueError("Unexpected value for num_reserved_ids.")

        self._init_alphabet_from_tokens(alphabet_tokens)
        self._init_subtokens_from_list(list(self._alphabet), reserved=num_reserved_ids)

        if min_count < 1:
            min_count = 1
        for i in xrange(num_iterations):
            tf.logging.info("Iteration {0}".format(i))
            subtoken_counts = collections.defaultdict(int)
            for token, count in six.iteritems(token_counts):
                escaped_token = _escape_token(token, self._alphabet)
                subtokens = self._escaped_token_to_subtoken_strings(escaped_token)
                start = 0
                for subtoken in subtokens:
                    for end in xrange(start+1, len(escaped_token)+1):
                        new_subtoken = escaped_token[start:end]
                        subtoken_counts[new_subtoken] += count
                    start += len(subtoken)

            len_to_subtoken_strings = []
            for subtoken_string, count in six.iteritems(subtoken_counts):
                len_subtoken = len(subtoken_string)
                if count >= min_count:
                    while len(len_to_subtoken_strings) <= len_subtoken:
                        len_to_subtoken_strings.append(set())
                    len_to_subtoken_strings[len_subtoken].add(subtoken_string)

            new_subtoken_strings = []
            for len_subtoken in xrange(len(len_to_subtoken_strings)-1, 0, -1):
                subtoken_strings = len_to_subtoken_strings[len_subtoken]
                for subtoken_string in subtoken_strings:
                    count = subtoken_counts[subtoken_string]
                    if count >= min_count:
                        if subtoken_string not in self._alphabet:
                            new_subtoken_strings.append((count, subtoken_string))
                        for l in xrange(1, len_subtoken):
                            subtoken_counts[subtoken_string[:l]] -= count

            new_subtoken_strings.extend((subtoken_counts.get(a,0),a) for a in self._alphabet)
            new_subtoken_strings.sort(reverse=True)
            
            self._init_subtokens_from_list([subtoken for _,subtoken in new_subtoken_strings], reserved=num_reserved_ids)
            tf.logging.info("vocab_size = %d" % self.vocab_size)

    def _escaped_token_to_subtoken_strings(self, escaped_token):
        '''escaped token -> list of subtoken strings'''
        ret = []
        start = 0
        token_len = len(escaped_token)
        while start < token_len:
            for end in xrange(min(token_len,start_self._max_subtoken_len), start, -1):
                subtoken = escaped_token[start:end]
                if subtoken in self._subtoken_string_to_id:
                    ret.append(subtoken)
                    start = end
                    break
                else:
                    assert False, "subtoken not found in subtoken vocab."
        return ret

    
    #def encode(self, raw_text):
    #    '''string -> list of subtoken ids'''
    #    return self._tokens_to_subtoken_ids(tokenizer.encode(raw_text))
    #
    #def decode(self, subtokens):
    #    '''list of subtoken ids -> string'''
    #    return tokenzer.decode(self._subtoken_ids_to_tokens(subtokens)))

    #
    #def _tokens_to_subtoken_ids(self, tokens):
    #    '''list of tokens -> list of subtoken ids'''
    #    ret = []
    #    for token in tokens:
    #        ret.extend(self._token_to_subtoken_ids(token))
    #    return ret

    #def _token_to_subtoken_ids(self, token):
    #    '''token -> list of subtoken ids'''


