import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))+"/"
from config import *
from utils import *
from oov_candidates_preprocessing import *

print("cd "+os.path.join(dir_path,"method_pmi2")+"; "+
    "java "+
    "-cp "+":".join([".",palmetto_jar,hppc_jar])+" "+
    "compute_pmi "+" ".join([
        "apply_pmi",
        onebest_file,
        candidate_list_file,
        function_words_file,
        punctuations_file,
        pmi_mat_dir,
        context_words_record_file,
        eng_vocab_file,
        candidate_list_file_out,
        str(num_candidates)])
    )

