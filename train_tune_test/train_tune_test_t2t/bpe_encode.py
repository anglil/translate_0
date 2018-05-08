from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import os
import sys 

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer
import tensorflow as tf

task = sys.argv[1]
in_file = sys.argv[2]
out_file = sys.argv[3]
vocab_file = sys.argv[4]
print("task: "+task)
print("in_file: "+in_file)
print("out_file: "+out_file)
print("vocab_file: "+vocab_file)

if not os.path.exists(out_file):
    vocab_obj = text_encoder.SubwordTextEncoder(vocab_file)
    
    if task == "encode":
        with open(in_file) as f_in, open(out_file, "w") as f_out:
            for l_in in f_in:
                l_in = l_in.strip()
        
                # token to subtoken_id
                l_in_ids = vocab_obj.encode(l_in)
    
                # subtoken_id to subtoken
                l_out = []
                for i in l_in_ids:
                    l_out.append(vocab_obj._subtoken_id_to_subtoken_string(i))
                l_out = " ".join(l_out)
                f_out.write(l_out+"\n")
        print("encoding done.")
    
    elif task == "decode":
        with open(in_file) as f_in, open(out_file, "w") as f_out:
            for l_in in f_in:
                l_in = l_in.strip().split(" ")
    
                # subtoken to subtoken_id
                l_in_subtokens = []
                for i in l_in:
                    if i in vocab_obj._subtoken_string_to_id:
                        l_in_subtokens.append(vocab_obj._subtoken_string_to_id[i])
                    elif i+" " in vocab_obj._subtoken_string_to_id:
                        l_in_subtokens.append(vocab_obj._subtoken_string_to_id[i+" "])
                        l_in_subtokens.append(vocab_obj._subtoken_string_to_id[" "])
                    elif " "+i in vocab_obj._subtoken_string_to_id:
                        l_in_subtokens.append(vocab_obj._subtoken_string_to_id[" "])
                        l_in_subtokens.append(vocab_obj._subtoken_string_to_id[" "+i])
    
                # subtoken_id to token
                l_out = vocab_obj.decode(l_in_subtokens)
                f_out.write(l_out+"\n")
        print("decoding done.")

    else:
        raise ValueError("task "+task+" not supported!")
else:
    print("out_file exists at: "+out_file)
