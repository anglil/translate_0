import sys
import os

dir_path = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.insert(0, dir_path+'../../')
from config import *
from utils import *


def train_charlm(combined_selected_non_domain_and_in_domain_doc, dev_ref_file, \
                 charlm_num_layer, charlm_input_dim, charlm_hidden_dim, \
                 charlm_model_file, charlm_dict_file, charlm_ppl_file, charlm_log_file):
    lr = 0.1

    model_dir = os.path.join(tmp_dir,"models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    gpu_id = -1
    
    forward_mem = 9192
    backward_mem = 1024
    param_mem = 512

    cmd0 = "cd "+"/home/ec2-user/kklab/Projects/lrlp/scripts/oov_translate/method_dclm3/train"+"; "
    cmd0 += "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"+boost_dir+"; "
    cmd0 += "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"+dynet_dir+"; "

    if gpu_id == -1:
        cmd = cmd0 + "./charlm "+\
        "--dynet-mem "+",".join([str(forward_mem),str(backward_mem),str(param_mem)])+" "+\
        "train "+\
        combined_selected_non_domain_and_in_domain_doc+" "+\
        dev_ref_file+" "+\
        str(charlm_num_layer)+" "+\
        str(charlm_input_dim)+" "+\
        str(charlm_hidden_dim)+" "+\
        charlm_model_file+" "+\
        charlm_dict_file+" "+\
        charlm_ppl_file+" "+\
        charlm_log_file+" "+\
        str(lr)
    else:
        cmd = cmd0 + "./charlm "+\
        "--dynet-gpu-ids "+str(gpu_id)+" "\
        "--dynet-mem "+",".join([str(forward_mem),str(backward_mem),str(param_mem)])+" "+\
        "train "+\
        combined_selected_non_domain_and_in_domain_doc+" "+\
        dev_ref_file+" "+\
        str(charlm_num_layer)+" "+\
        str(charlm_input_dim)+" "+\
        str(charlm_hidden_dim)+" "+\
        charlm_model_file+" "+\
        charlm_dict_file+" "+\
        charlm_ppl_file+" "+\
        charlm_log_file+" "+\
        str(lr)
    
    print("charlm:")
    print(cmd+"\n")

    
def train_dclm(combined_selected_non_domain_with_unk_and_in_domain_doc, dev_ref_file, model_type, \
               dclm_num_layer, dclm_input_dim, dclm_hidden_dim, dclm_align_dim, dclm_len_thresh, \
               dclm_model_file, dclm_dict_file, dclm_ppl_file, dclm_log_file):
    lr = 0.1
    
    model_dir = os.path.join(tmp_dir,"models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    gpu_id = -1
    
    forward_mem = 9192
    backward_mem = 1024
    param_mem = 512

    cmd0 = "cd "+"/home/ec2-user/kklab/Projects/lrlp/scripts/oov_translate/method_dclm3/train"+"; "
    cmd0 += "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"+boost_dir+"; "
    cmd0 += "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"+dynet_dir+"; "

    if gpu_id == -1:
        cmd = cmd0 + "./dclm "+\
        "--dynet-mem "+",".join([str(forward_mem),str(backward_mem),str(param_mem)])+" "+\
        "train "+\
        combined_selected_non_domain_with_unk_and_in_domain_doc+" "+\
        dev_ref_file+" "+\
        model_type+" "+\
        str(dclm_num_layer)+" "+\
        str(dclm_input_dim)+" "+\
        str(dclm_hidden_dim)+" "+\
        str(dclm_align_dim)+" "+\
        str(dclm_len_thresh)+" "+\
        dclm_model_file+" "+\
        dclm_dict_file+" "+\
        dclm_ppl_file+" "+\
        dclm_log_file+" "+\
        str(lr)
    else:
        cmd = cmd0 + "./dclm "+\
        "--dynet-gpu-ids "+str(gpu_id)+" "\
        "--dynet-mem "+",".join([str(forward_mem),str(backward_mem),str(param_mem)])+" "+\
        "train "+\
        combined_selected_non_domain_with_unk_and_in_domain_doc+" "+\
        dev_ref_file+" "+\
        model_type+" "+\
        str(dclm_num_layer)+" "+\
        str(dclm_input_dim)+" "+\
        str(dclm_hidden_dim)+" "+\
        str(dclm_align_dim)+" "+\
        str(dclm_len_thresh)+" "+\
        dclm_model_file+" "+\
        dclm_dict_file+" "+\
        dclm_ppl_file+" "+\
        dclm_log_file+" "+\
        str(lr)

    print(model_type+":")
    print(cmd+"\n")


if __name__ == "__main__":
    num_of_doc_to_select = 20000
    combined_selected_non_domain_and_in_domain_doc = os.path.join(tmp_dir,"combined_selected_non_domain_"+str(num_of_doc_to_select)+"_and_in_domain_doc")
    combined_selected_non_domain_with_unk_and_in_domain_doc = os.path.join(tmp_dir,"combined_selected_non_domain_with_unk_"+str(num_of_doc_to_select)+"_and_in_domain_doc")

    # train charlm
    # input: combined_selected_non_domain_and_in_domain_doc, dev_ref_file, charlm_num_layer, charlm_input_dim, charlm_hidden_dim
    # output: charlm_model_file, charlm_dict_file, charlm_ppl_file, charlm_log_file
    charlm_num_layer = 2
    charlm_input_dim = 48
    charlm_hidden_dim = 48
    charlm_model_file = tmp_dir+"models/"+'_'.join(["charlm",t,str(charlm_num_layer),str(charlm_input_dim),str(charlm_hidden_dim)])+".model"
    charlm_dict_file = tmp_dir+"models/"+'_'.join(["charlm",t,str(charlm_num_layer),str(charlm_input_dim),str(charlm_hidden_dim)])+".dict"
    charlm_ppl_file = tmp_dir+"models/"+'_'.join(["charlm",t,str(charlm_num_layer),str(charlm_input_dim),str(charlm_hidden_dim)])+".ppl"
    charlm_log_file = tmp_dir+"models/"+'_'.join(["charlm",t,str(charlm_num_layer),str(charlm_input_dim),str(charlm_hidden_dim)])+".log"
    train_charlm(
        combined_selected_non_domain_and_in_domain_doc, \
        dev_ref_file, \
        charlm_num_layer, \
        charlm_input_dim, \
        charlm_hidden_dim, \
        charlm_model_file, \
        charlm_dict_file, \
        charlm_ppl_file, \
        charlm_log_file)
    print('--------')
    
    # train dclm
    # input: combined_selected_non_domain_with_unk_and_in_domain_doc, dev_ref_file, model_type, dclm_num_layer, dclm_input_dim, dclm_hidden_dim, dclm_align_dim, dclm_len_thresh
    # output: dclm_model_file, dclm_dict_file, dclm_ppl_file, dclm_log_file
    dclm_num_layer = 2
    dclm_input_dim = 48
    dclm_hidden_dim = 48
    dclm_align_dim = 48 # only for adclm (however for convenience, this param is attached to every model name)
    dclm_len_thresh = 4
    for model_type in {"rnnlm", "adclm", "ccdclm", "codclm"}:
        dclm_model_file = tmp_dir+"models/"+'_'.join([model_type,t,str(dclm_num_layer),str(dclm_input_dim),str(dclm_hidden_dim),str(dclm_align_dim),str(dclm_len_thresh)])+".model"
        dclm_dict_file = tmp_dir+"models/"+'_'.join([model_type,t,str(dclm_num_layer),str(dclm_input_dim),str(dclm_hidden_dim),str(dclm_align_dim),str(dclm_len_thresh)])+".dict"
        dclm_ppl_file = tmp_dir+"models/"+'_'.join([model_type,t,str(dclm_num_layer),str(dclm_input_dim),str(dclm_hidden_dim),str(dclm_align_dim),str(dclm_len_thresh)])+".ppl"
        dclm_log_file = tmp_dir+"models/"+'_'.join([model_type,t,str(dclm_num_layer),str(dclm_input_dim),str(dclm_hidden_dim),str(dclm_align_dim),str(dclm_len_thresh)])+".log"
        train_dclm(
            combined_selected_non_domain_with_unk_and_in_domain_doc, \
            dev_ref_file, \
            model_type, \
            dclm_num_layer, \
            dclm_input_dim, \
            dclm_hidden_dim, \
            dclm_align_dim, \
            dclm_len_thresh, \
            dclm_model_file, \
            dclm_dict_file, \
            dclm_ppl_file, \
            dclm_log_file)
    print('--------')
