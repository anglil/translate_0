import numpy as np
import operator
from numpy import linalg as LA
import collections
import re
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import diags
import heapq
import sys
import os

dir_path = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.insert(0, dir_path+'../../oov_translate')
from config import *
from utils import *
from oov_candidates_preprocessing import *


def load_glove(dim):
    '''
    load glove into a dictionary
    '''
    glove_dict = dict()
    glove_file = glove_dir+"glove.6B."+str(dim)+"d.txt"
    with open(glove_file) as f:
        for line in f:
            l = line.strip().split(' ')
            word = l[0]
            vec = [float(l[i+1]) for i in range(dim)]
            glove_dict[word] = vec
                
    return glove_dict


def get_vec(word, glove_dict):
    '''
    query the dictionary for a word's vector representation
    '''
    if word in glove_dict:
        return glove_dict[word]
    return glove_dict['unk']


def get_similarity_cosine(vec1, vec2):
    '''
    compute the cosine distance between two vectors
    '''
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    d = np.dot(vec1, vec2)/(LA.norm(vec1)*LA.norm(vec2))
    return d


#def get_similarity_pmi(w1, w2):
    


def compose_sent_bucket(sent_bucket, candidate_bucket):
    '''
    sent_tok: [words]
    oov_pos: [[pos],[pos],...]
    '''
    assert(len(sent_bucket)==len(candidate_bucket))
    sent_tok = []
    oov_pos = []
    ctr = 0
    for i in range(len(sent_bucket)):
        for j in range(len(sent_bucket[i])):
            if j in candidate_bucket[i]:
                ctr0 = []
                for k in range(len(candidate_bucket[i][j])):
                    sent_tok.append(candidate_bucket[i][j][k])
                    ctr0.append(ctr)
                    ctr += 1
                oov_pos.append(ctr0)
            else:
                sent_tok.append(sent_bucket[i][j])
                ctr += 1
    return sent_tok, oov_pos


def pagerank_init(sent_tok, oov_pos, dist_measure, glove_dict):
    '''
    sent_tok: [tok]
    oov_pos: [[oov_pos, ...], [oov_pos, ...]]
    complete_graph: boolean
    dist_measure: glove or pmi
    '''
    if dist_measure == "glove":
        glove_dict = load_glove(300)
    elif dist_measure == "pmi":
        import pmi

    oov_pos_set = set()
    for pos in oov_pos:
        oov_pos_set.add(pos)

    graph_size = len(sent_tok)
    An = lil_matrix((graph_size, graph_size))
    for i in range(graph_size):
        if i not in oov_pos_set:
            # context word
            word_vec = get_vec(sent_tok[i], glove_dict)
            for j in range(graph_size):
                if i != j:
                    # oov word or context word
                    neighbor_vec = get_vec(sent_tok[j], glove_dict)
                    An[i,j] = get_similarity_cosine(word_vec, neighbor_vec)
                    An[j,i] = An[i,j]
    return An


def pagerank_sparse(An, epsilon=0.0001):
    '''An is in lil_matrix format'''
    # regularize An
    graph_size = An.shape[0]
    outlink_rank_sum = An.sum(0)
    outlink_rank_sum = np.array(outlink_rank_sum)[0]
    for i in range(graph_size):
        if outlink_rank_sum[i] == 0:
            An[:,i] = [1.0/graph_size for i in range(graph_size)]
            outlink_rank_sum[i] = 1
    An = An.tocsc()
    An = An.dot(np.power(outlink_rank_sum, -1, dtype=float))

    # compute pagerank
    pagerank_cur = np.ones(graph_size)*1.0/graph_size
    pagerank_pre = np.zeros(graph_size)
    ctr = 0
    while abs(LA.norm(pagerank_cur)-LA.norm(pagerank_pre)) > epsilon:
        pagerank_pre = np.array(pagerank_cur)
        pagerank_cur = An.dot(pagerank_cur)
        pagerank_cur = pagerank_cur/LA.norm(pagerank_cur)
        ctr += 1
    return pagerank_cur


def pagerank_interpret(pagerank, sent_tok, oov_pos, sent_bucket, candidate_bucket):
    '''
    pagerank: numpy array
    sent_tok: [words]
    oov_pos: [[oov_pos],[oov_pos],..]
    sent_bucket: [[words],[words]..] with oov
    res: [[words],[words]..] without oov
    '''
    oov_trans = []
    assert(len(pagerank)==len(sent_tok))
    for i in range(len(oov_pos)):
        max_pos = -1
        max_score = -1
        for j in range(len(oov_pos[i])):
            if pagerank[oov_pos[i][j]] > max_score:
                max_pos = oov_pos[i][j]
        if max_pos != -1:
            oov_trans.append(sent_tok[max_pos])

    res = list(sent_bucket)
    ctr = 0
    for i in range(len(sent_bucket)):
        res.append([])
        for j in range(len(sent_bucket[i])):
            if j in candidate_bucket[i]:
                res[i][j] = oov_trans[ctr]
                ctr += 1
    return res


def pagerank_routine(sent_bucket, candidate_bucket):
    sent_tok, oov_pos = compose_sent_bucket(sent_bucket, candidate_bucket)
    # initialize pagerank (initialize transition matrix, or markov matrix)
    An = pagerank_init(sent_tok, oov_pos, complete_graph, dist_measure)
    # compute pagerank
    pagerank = pagerank_sparse(An)
    # interpret pagerank
    res = pagerank_interpret(pagerank, candidate_bucket)
    return res


def rescore_lattice(onebest_file, candidate_list_file, complete_graph, dist_measure, context_window, res_file):
    sent_bucket = [] # list of list
    candidate_bucket = [] # list of dict that maps word to list of candidates
    with open(onebest_file) as fo, open(candidate_list_file) as fc:#, open(res_file, 'w') as fr:
        for l_onebest in fo:
            l_onebest = l_onebest.strip()
            l_candidate = fc.readline().strip()
            if l_candidate != "=":
                onebest_tok = l_onebest.split(' ')
                candidate_map = {int(pair.split(":")[0]):pair.split(":")[1].split(",") for pair in l_candidate.split(" ")} if l_candidate != "" else {}

                sent_bucket.append(onebest_tok)
                candidate_bucket.append(candidate_map)

                if len(sent_bucket) == context_window:
                    trans = pagerank_routine(sent_bucket, candidate_bucket)

                    for item in trans:
                        fr.write(item+"\n")

                    sent_bucket = []
                    candidate_bucket = []

            else l_candidate == "=":
                if len(sent_bucket) != 0:
                    trans = pagerank_routine(sent_bucket, candidate_bucket)

                    for item in trans:
                        fr.write(item+"\n")

                    sent_bucket = []
                    candidate_bucket = []

                fr.write("=\n")

        if len(sent_bucket) != 0:
            trans = pagerank_routine(sent_bucket, candidate_bucket)

            for item in trans:
                fr.write(item+"\n")
                    


if __name__ == "__main__":
    ocp = oov_candidates_preprocessing()
    
    # dev, test, domain, eval
    for dataset in ["test"]:
        # extracted, eng_vocab, extracted_eng_vocab, glosbe
        for candidate_source in [["extracted","masterlexicon","googletranslate"],["extracted","masterlexicon","googletranslate","aligned"]]:
            # oov pos:w1,w2 file prep
            _, _, onebest_file, _, _, candidate_list_file = ocp.init(dataset, candidate_source)
            prefix = os.path.basename(candidate_list_file).split(".")[0]
            if "False" in prefix:
                mt = "_t2t_dim512_layer2_lr0.2_dropout0.1_bpe8000"

            dist_measure = "pmi" # pmi or glove
            context_window = 4 # number of sentences as the context

            ref_file = os.path.join(res_dir,dataset,".".join([ref_label+raw,t,dataset,yrv]))
            print("ref_file: "+ref_file)
            assert(os.path.exists(ref_file))
            print("----")

            res_attr = "_".join(["onebest"+mt,prefix,dist_measure,"context"+str(context_window)])
            res_file = os.path.join(res_dir,dataset,".".join([res_attr,t,dataset,yrv]))

            if not os.path.exists(res_file):
                rescore_lattice(
                    onebest_file, 
                    candidate_list_file,
                    complete_graph,
                    dist_measure,
                    context_window,
                    res_file)
            else:
                print("res_file exists at: "+res_file)

            if os.path.exists(res_file):
                stdout, _ = sh(bleu_getter+" -lc "+ref_file+" < "+res_file)
                print(stdout)


