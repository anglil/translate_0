import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))+"/"
sys.path.insert(0, dir_path+'../../')
from config import *
from utils import *
from oov_candidates_preprocessing import *
ocp = oov_candidates_preprocessing()

assert("pmi" in tmp_dir)
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

function_words_file = data_dir+"function_words_file"
if not os.path.exists(function_words_file):
    with open(function_words_file, 'w') as f:
        for word in function_words:
            f.write(word+"\n")

punctuations_file = data_dir+"punctuations_file"
if not os.path.exists(punctuations_file):
    with open(punctuations_file, 'w') as f:
        for punctuation in punctuations:
            f.write(punctuation+"\n")

# compile
print("--------")
print("compile everything pmi-related: ")
print("cd "+os.path.join(dir_path,"method_pmi2")+"; "+\
    "javac "+\
    "-cp "+":".join([".",palmetto_jar,hppc_jar])+" "+\
    "*.java")
