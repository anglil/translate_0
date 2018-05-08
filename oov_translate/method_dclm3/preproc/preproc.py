import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))+"/"
sys.path.insert(0, dir_path+'../../')
from config import *
from utils import *
from oov_candidates_preprocessing import *
ocp = oov_candidates_preprocessing()

# directory in which to store oov translation models
assert("dclm" in tmp_dir)
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

from data_preprocessing import *

if os.path.exists(train_ref_file):
    '''
    get seed for data selection
        input: dev_ref_file, test_nbest_file
        output: proc_seed_in_domain
    '''
    proc_seed_in_domain = tmp_dir+"stemmed_seed."+t+".dev_test."+yrv # no stopwords, no punctuations
    #get_seed_for_data_selection2(
    get_seed_for_data_selection3(
        dev_ref_file,
        test_1best_file,
        proc_seed_in_domain)
    print('--------')
    '''
    get target for data selection
        input: wiki_dump_train
        output: doc_non_domain, doc_non_domain_proc
    '''
    doc_non_domain = wiki_dump_train+".nonproc"
    doc_non_domain_proc = wiki_dump_train+".proc" # no stopwords, no punctuations
    get_target_for_data_selection2(
        wiki_dump_train,
        doc_non_domain,
        doc_non_domain_proc)
    print('--------')
    '''
    build vocab
        input: train_ref_file, dev_ref_file, test_1best_file, candidate_list_file
        output: vocab set
    '''
    _, _, _, _, _, dev_candidate_list_file = ocp.init("dev", ["googletranslate", "masterlexicon", "extracted", "aligned"])
    _, _, _, _, _, test_candidate_list_file = ocp.init("test", ["googletranslate", "masterlexicon", "extracted", "aligned"])
    dclm_vocab = build_vocab3(
        [train_ref_file, dev_ref_file, test_1best_file],
        [dev_candidate_list_file, test_candidate_list_file])
    print('--------')
    '''
    select doc with high jaccard idx
        input: proc_seed_in_domain, doc_non_domain, doc_non_domain_proc, selection threshold, dclm_vocab
        output: selected_non_domain_doc_file, selected_non_domain_doc_with_unk_file
    '''
    num_of_doc_to_select = 20000
    selected_non_domain_doc_file = tmp_dir+"selected_non_domain_doc_file_"+str(num_of_doc_to_select) # for char-lm
    selected_non_domain_doc_with_unk_file = tmp_dir+"selected_non_domain_doc_with_unk_file_"+str(num_of_doc_to_select) # for dclm
    select_doc_with_high_jaccard_idx3(
        proc_seed_in_domain,
        doc_non_domain,
        doc_non_domain_proc,
        num_of_doc_to_select,
        dclm_vocab,
        selected_non_domain_doc_file,
        selected_non_domain_doc_with_unk_file)
    print('--------')
    '''
    merge files with boundary

    input: train_ref_file, (unseq_ref_file), selected_non_domain_doc_file/selected_non_domain_doc_with_unk_file
    output: combined_selected_non_domain_and_in_domain_doc/combined_selected_non_domain_with_unk_and_in_domain_doc
    '''
    combined_selected_non_domain_and_in_domain_doc = tmp_dir+"combined_selected_non_domain_"+str(num_of_doc_to_select)+"_and_in_domain_doc"
    combined_selected_non_domain_with_unk_and_in_domain_doc = tmp_dir+"combined_selected_non_domain_with_unk_"+str(num_of_doc_to_select)+"_and_in_domain_doc"
    throw_long_sentences = True
    merge_files_with_boundary(
        [train_ref_file,selected_non_domain_doc_file],
        combined_selected_non_domain_and_in_domain_doc, 
        throw_long_sentences)
    merge_files_with_boundary(
        [train_ref_file,selected_non_domain_doc_with_unk_file],
        combined_selected_non_domain_with_unk_and_in_domain_doc, 
        throw_long_sentences)
    print('--------')

else:
    import shutil
    print("existing trained models:")
    print("/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.08.il5-eng.y2r1.v1/oov_trans_dclm/combined_selected_non_domain_20000_and_in_domain_doc")
    print("/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.08.il5-eng.y2r1.v1/oov_trans_dclm/combined_selected_non_domain_with_unk_20000_and_in_domain_doc")
    #print("/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.08.il5-eng.y2r1.v1/oov_trans_dclm/models/")
    print("/home/ec2-user/kklab/Projects/lrlp/experiment_isi-sbmt-amh-edoov.il5-eng.y2r1.v3/oov_trans_dclm/models/")
    print("--------")
    print("/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.08.il6-eng.y2r1.v1/oov_trans_dclm/combined_selected_non_domain_20000_and_in_domain_doc")
    print("/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.08.il6-eng.y2r1.v1/oov_trans_dclm/combined_selected_non_domain_with_unk_20000_and_in_domain_doc")
    #print("/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.08.il6-eng.y2r1.v1/oov_trans_dclm/models/")
    print("/home/ec2-user/kklab/Projects/lrlp/experiment_isi-sbmt.il6-eng.y2r1.v4/oov_trans_dclm/models/")
    print("--------")

    combined_selected_non_domain_and_in_domain_doc_old = input("existing combined_selected_non_domain_and_in_domain_doc: ")
    combined_selected_non_domain_with_unk_and_in_domain_doc_old = input("existing combined_selected_non_domain_with_unk_and_in_domain_doc: ")
    model_and_dict_dir_old = input("existing model_and_dict_dir: ")

    num_of_doc_to_select = 20000
    combined_selected_non_domain_and_in_domain_doc = tmp_dir+"combined_selected_non_domain_"+str(num_of_doc_to_select)+"_and_in_domain_doc"
    combined_selected_non_domain_with_unk_and_in_domain_doc = tmp_dir+"combined_selected_non_domain_with_unk_"+str(num_of_doc_to_select)+"_and_in_domain_doc"
    model_and_dict_dir = tmp_dir+"models/"

    shutil.copy(combined_selected_non_domain_and_in_domain_doc_old, combined_selected_non_domain_and_in_domain_doc)
    shutil.copy(combined_selected_non_domain_with_unk_and_in_domain_doc_old, combined_selected_non_domain_with_unk_and_in_domain_doc)
    shutil.copytree(model_and_dict_dir_old, model_and_dict_dir)

