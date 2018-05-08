import sys 
import os

dir_path = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.insert(0, dir_path+'../../')
sys.path.insert(0, dir_path+'../preproc')
from config import *
from utils import *
from data_preprocessing import train_lm

if __name__ == "__main__":

    ngram = 4
    lm_name = str(ngram)+"gram"

    train_final = os.path.join(tmp_dir, os.path.basename(train_ref_file)+".final_"+lm_name)
    lm_final_path = os.path.join(tmp_dir, "lm_"+lm_name+"_final")

    if not os.path.exists(lm_final_path+".binary"):
        train_lm(ngram, train_final, lm_final_path)
        print("language model created at: "+lm_final_path)
        sh(build_binary+" "+lm_final_path+" "+lm_final_path+".binary")
        print("language model binarized at: "+lm_final_path+".binary")
    else:
        print("binarized language model exists at: "+lm_final_path+".binary")

