import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))+"/"
sys.path.insert(0, dir_path+'../../../oov_translate')
from config import *
from utils import *
from oov_candidates_preprocessing import *

def rescore_lattice(onebest_file, candidate_list_file, model_type, dclm_num_layer, dclm_input_dim, dclm_hidden_dim, dclm_align_dim, dclm_model_file, dclm_dict_file, decoder_type, beam_size, include_charlm, charlm_model_file, charlm_dict_file, charlm_num_layer, charlm_input_dim, charlm_hidden_dim, res_file):
    
    forward_mem_test = 9192
    
    cmd0 = "cd "+"/home/ec2-user/kklab/Projects/lrlp/scripts/oov_translate/method_dclm3/train/"+"; "
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

if __name__ == "__main__":
    ocp = oov_candidates_preprocessing()

    charlm_num_layer = 2
    charlm_input_dim = 48
    charlm_hidden_dim = 48
    charlm_model_file = os.path.join(tmp_dir,"models",'_'.join(["charlm",t,str(charlm_num_layer),str(charlm_input_dim),str(charlm_hidden_dim)])+".model")
    charlm_dict_file = os.path.join(tmp_dir,"models",'_'.join(["charlm",t,str(charlm_num_layer),str(charlm_input_dim),str(charlm_hidden_dim)])+".dict")

    dclm_num_layer = 2
    dclm_input_dim = 48
    dclm_hidden_dim = 48
    dclm_align_dim = 48
    dclm_len_thresh = 4

    beam_size = 4

    # dev, test, syscomb, eval
    for dataset in ["test"]:#{"dev", "test", "domain", "eval"}:
        # extracted, aligned_extracted, eng_vocab, extracted_eng_vocab, aligned
        for candidate_source in [["masterlexicon"],["extracted"],["extracted","googletranslate"],["aligned"],["masterlexicon", "aligned"],["extracted","aligned","masterlexicon"],["extracted", "aligned"],["googletranslate", "masterlexicon"],["masterlexicon", "extracted", "masterlexicon"]]:
        #for candidate_source in [["masterlexicon","googletranslate","alignedhyp"],["masterlexicon","googletranslate","aligned","alignedhyp"]]:
            # oov file prep
            _, _, onebest_file, _, _, candidate_list_file = ocp.init(dataset, candidate_source)
            #prefix = "-".join(sorted(candidate_source)) if "alignedhyp" not in candidate_source else "-".join(sorted(candidate_source))+"-True"
            prefix = os.path.basename(candidate_list_file).split(".")[0]
            if "False" in prefix:
                mt = "_t2t_dim512_layer2_lr0.2_dropout0.1_bpe8000"

            #onebest_file = os.path.join(res_dir,dataset,".".join(["onebest"+mt,t,dataset,yrv]))
            #print("onebest_file: "+onebest_file)
            #assert(os.path.exists(onebest_file))
            #print("----")

            #candidate_list_file = os.path.join(res_dir,dataset,"oov",".".join([prefix,t,dataset,yrv]))
            #print("candidate_list_file: "+candidate_list_file)
            #assert(os.path.exists(candidate_list_file))
            #print("--------")

            # adclm, ccdclm, codclm, rnnlm
            for model_type in ["adclm"]:
                dclm_model_file = tmp_dir+"models/"+'_'.join([model_type,t,str(dclm_num_layer),str(dclm_input_dim),str(dclm_hidden_dim),str(dclm_align_dim),str(dclm_len_thresh)])+".model"
                dclm_dict_file = tmp_dir+"models/"+'_'.join([model_type,t,str(dclm_num_layer),str(dclm_input_dim),str(dclm_hidden_dim),str(dclm_align_dim),str(dclm_len_thresh)])+".dict"
                # beam (beam search), context (comparing contexts), embed (add hisorical embeddings)
                for decoder_type in ["context"]:
                    # True, False
                    for include_charlm in {False}:
                        print(dataset)
                        print(candidate_source)
                        print(model_type)
                        print("----")

                        ref_file = os.path.join(res_dir,dataset,".".join([ref_label+raw,t,dataset,yrv]))
                        print("ref_file: "+ref_file)
                        assert(os.path.exists(ref_file))
                        print("----")

                        res_attr = "_".join(["onebest"+mt,prefix,model_type,decoder_type,str(include_charlm)])
                        res_file = os.path.join(res_dir,dataset,".".join([res_attr,t,dataset,yrv]))
                        #if s!="vie":
                        #    print("res_file: "+res_file)
                        #else:
                        #    print("res_file: "+res_file+".add")
                        #print('----')
                        
                        if not os.path.exists(res_file):
                            rescore_lattice(
                                onebest_file,
                                candidate_list_file,
                                model_type,
                                dclm_num_layer,
                                dclm_input_dim,
                                dclm_hidden_dim,
                                dclm_align_dim,
                                dclm_model_file,
                                dclm_dict_file,
                                decoder_type,
                                beam_size,
                                include_charlm,
                                charlm_model_file,
                                charlm_dict_file,
                                charlm_num_layer,
                                charlm_input_dim,
                                charlm_hidden_dim,
                                res_file)
                        else:
                            print("res_file exists at: "+res_file)

                        #if s=="vie":
                        #    from add_empty_doc import myadd
                        #    if os.path.exists(res_file):
                        #        myadd(ref_file, res_file, res_file+".add")
                        #    res_file = res_file+".add"

                        if os.path.exists(res_file):
                            stdout, _ = sh(bleu_getter+" -lc "+ref_file+" < "+res_file)
                            print(stdout)

                        #if os.path.exists(res_file) and dataset == "eval":
                        #    print("res_file for "+dataset+" exists at: "+res_file)
                        #    res_file_xml = corpus_dir+".".join([dataset_name+"-uw-oov", candidate_source, st, dataset, yrv, "xml"])
                        #    write_translation_to_xml(data_in_domain_xml, res_file, res_file_xml)
                        #    print("xml written to: "+res_file_xml)
                        #    print('--------\n')
