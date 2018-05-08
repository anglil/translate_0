import sys
import os

dir_path = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.insert(0, dir_path+'../../')
from config import *
from utils import *

'''
onebest_file
candidate_list_file
'''

print("cd "+os.path.join(dir_path,"method_pmi2")+"; "+
    "java "+
    "-cp "+":".join([".",palmetto_jar,hppc_jar])+" "+
    "compute_pmi "+" ".join([
        "collect_pmi",
        onebest_file,
        candidate_list_file,
        function_words_file,
        punctuations_file,
        pmi_mat_dir,
        context_words_record_file,
        eng_vocab_file,
        index_path,
        context_scale,
        window_mechanism,
        str(pmi_mat_capacity)])
    )
