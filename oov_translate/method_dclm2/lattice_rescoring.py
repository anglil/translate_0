import sys
import os

sys.path.insert(0, '../')
from config import *
from utils import *

def rescore_lattice(onebest_file, candidate_list_file, model_type, dclm_num_layer, dclm_input_dim, dclm_hidden_dim, dclm_align_dim, dclm_model_file, dclm_dict_file, decoder_type, beam_size, include_charlm, charlm_model_file, charlm_dict_file, charlm_num_layer, charlm_input_dim, charlm_hidden_dim, res_file):
    
    forward_mem_test = 9192
    
    cmd0 = "cd "+os.getcwd()+"; "
    cmd0 += "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"+boost_dir+"; "
    cmd0 += "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"+dynet_dir+"; "

    if model_type == "rnnlm":
        cmd = cmd0 + "./latticedec0 "+\
        "--dynet-mem "+str(forward_mem_test)+" "+\
        str(dclm_num_layer)+" "+\
        str(dclm_input_dim)+" "+\
        str(dclm_hidden_dim)+" "+\
        str(beam_size)+" "+\
        dclm_model_file+" "+\
        dclm_dict_file+" "+\
        onebest_file+" "+\
        candidate_list_file+" "+\
        res_file

    elif model_type == "adclm":
        cmd = cmd0 + "./latticedec1 "+\
        "--dynet-mem "+str(forward_mem_test)+" "+\
        str(dclm_num_layer)+" "+\
        str(dclm_input_dim)+" "+\
        str(dclm_hidden_dim)+" "+\
        str(dclm_align_dim)+" "+\
        str(beam_size)+" "+\
        dclm_model_file+" "+\
        dclm_dict_file+" "+\
        onebest_file+" "+\
        candidate_list_file+" "+\
        res_file

    elif model_type == "ccdclm":
        cmd = cmd0 + "./latticedec2 "+\
        "--dynet-mem "+str(forward_mem_test)+" "+\
        str(dclm_num_layer)+" "+\
        str(dclm_input_dim)+" "+\
        str(dclm_hidden_dim)+" "+\
        str(beam_size)+" "+\
        dclm_model_file+" "+\
        dclm_dict_file+" "+\
        onebest_file+" "+\
        candidate_list_file+" "+\
        res_file

    elif model_type == "codclm":
        cmd = cmd0 + "./latticedec3 "+\
        "--dynet-mem "+str(forward_mem_test)+" "+\
        str(dclm_num_layer)+" "+\
        str(dclm_input_dim)+" "+\
        str(dclm_hidden_dim)+" "+\
        str(beam_size)+" "+\
        dclm_model_file+" "+\
        dclm_dict_file+" "+\
        onebest_file+" "+\
        candidate_list_file+" "+\
        res_file

    ### beam search or context vector 
    cmd += " "+decoder_type
    
    ### add char-level model into dclm to help with the decoding
    if include_charlm:
        cmd += " "+str(include_charlm)
        cmd += " "+charlm_model_file
        cmd += " "+charlm_dict_file
        cmd += " "+str(charlm_num_layer)
        cmd += " "+str(charlm_input_dim)
        cmd += " "+str(charlm_hidden_dim)

    print(cmd)
