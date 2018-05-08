from tensor2tensor.data_generators import text_encoder
from reg_problems import replace_oov_with_unk
import sys

infile = sys.argv[1]
outfile = sys.argv[2]
vocab_file = sys.argv[3]

vocab_obj = text_encoder.TokenTextEncoder(vocab_file)
replace_oov_with_unk(infile, outfile, vocab_obj._token_to_id)

