
# coding: utf-8

# In[1]:

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

dir_path = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.insert(0, dir_path+'../../oov_translate')
from config import *
from utils import *
from oov_candidates_preprocessing import *


# In[2]:

tmp_dir = exp_dir+"oov_trans_pagerank/"
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)


# In[3]:

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

# #-------- load GloVe vectors --------
# print("loading GloVe...")
# glove_dict = load_glove(300)
# print("GloVe loaded.")



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



class WordGraph:
    def __init__(self,tra_tok,oov_pos,context,glove_dict,eng_vocab,ug_dict,candidate_source,add_aligned_oov,complete_graph):
        '''
        construct a complete or incomplete graph
        '''
        ### parameters for oov translation
        self.candidate_source = candidate_source
        self.add_aligned_oov = add_aligned_oov
        self.complete_graph = complete_graph
        
        ### parameters for pagerank
        self.decay = 0.85
        self.epsilon = 10**-4   
        
        ### only used in "eng_vocab", in place for oov_candidates
        ### a heap that stores the ranked oov candidates
        self.rank_heap = []
        
        #self.nodes = dict()
        
        oov_words = [tra_tok[i] for i in oov_pos]
        oov_words_set = set(oov_words)
        context_words = [tra_tok[i] for i in context]
        context_words_set = set(context_words)
        
        ### only used in "eng_vocab", 
        ### this is an upper bound for the number of oov words instead of a precise number
        self.oov_num = len(oov_words)
        
        self.glove_dict = glove_dict
        self.ug_dict = ug_dict
        self.eng_vocab = eng_vocab
        ### for eng_vocab: {candidate: score}
        ### for ug_dict: {oov:{candidate: score}}
        self.oov_candidates = get_oov_candidates_all(self.candidate_source,                                                      self.add_aligned_oov,                                                      self.ug_dict,                                                      self.eng_vocab,                                                      oov_words_set,                                                      context_words_set)
            
        ### form word_list
        ### 1. attach the context symbol with index to the context word
        self.word_list = [tra_tok[i]+" context"+str(i) for i in context]
        if self.candidate_source is "ug_dict":
            for oov in self.oov_candidates:
                if oov not in self.oov_candidates[oov]:
                    ### 2. attach the oov word a candidate word is referring to to the candidate
                    self.word_list += [candidate+" "+oov for candidate in self.oov_candidates[oov].keys() if candidate not in context_words_set]
        elif self.candidate_source is "eng_vocab":
            for candidate in self.oov_candidates:
                if candidate not in context_words_set:
                    ### 3. attach the symbolic oov word a candidate word is referring to to the candidate
                    self.word_list.append(candidate+" oov")
                    
        self.graph_size = len(self.word_list)
        self.pr = np.ones(self.graph_size)*1.0/self.graph_size      
        
#         ### establish the similarity between nodes (need to normalize the sum to 1)
#         for i in range(self.graph_size):
#             word = self.word_list[i]
#             if word not in self.nodes:
#                 self.nodes[word] = dict()
#             ### initialize the rank for every word node
#             self.nodes[word]['_rank_'] = 1.0/self.graph_size
#             if i != self.graph_size-1:
#                 for j in range(i+1, self.graph_size):
#                     neighbor = self.word_list[j]
#                     word_vec = get_vec(word.split(' ')[0], glove_dict)
#                     neighbor_vec = get_vec(neighbor.split(' ')[0], glove_dict)
#                     self.nodes[word][neighbor] = get_similarity_cosine(word_vec, neighbor_vec)
#                     if neighbor not in self.nodes:
#                         self.nodes[neighbor] = dict()
#                     ### inlink-outlink symmetry
#                     self.nodes[neighbor][word] = self.nodes[word][neighbor]
    
    
    
    def pagerank_sparse(self):
        '''
        pagerank sparse matrix version
        default: incomplete graph
        '''
        ### form the unnormalized matrix
        An = lil_matrix((self.graph_size, self.graph_size))
        for i in range(self.graph_size):
            word = self.word_list[i]
            word_pair = word.split(' ')
            ### default: not complete graph
            if re.search('^context.*', word_pair[1]) != None:
                word_vec = get_vec(word_pair[0], self.glove_dict)
                for j in range(self.graph_size):
                    if i != j:
                        neighbor = self.word_list[j]
                        neighbor_pair = neighbor.split(' ')
                        neighbor_vec = get_vec(neighbor_pair[0], self.glove_dict)
                        An[i,j] = get_similarity_cosine(word_vec, neighbor_vec)
                        An[j,i] = An[i,j]
        
        #print("form the normalizer")
        normalizer = []
        unit = 1.0/self.graph_size
        for i in range(self.graph_size):
            normalizer.append([unit])

        #print("check for 0 sum columns")
        outlink_rank_sum = An.sum(0)
        outlink_rank_sum = np.array(outlink_rank_sum)[0] # matrix to array
        for i in range(self.graph_size):
            if outlink_rank_sum[i] == 0:
                An[:,i] = normalizer
                outlink_rank_sum[i] = 1
                
        #print("normalize the matrix")
        ### convert the matrix in lil format to csr format for arithmetic and vector product operations
        An = An.tocsc()
        ### form the normalized matrix (normalize on columns: axis=0) 
        outlink_rank_sum_reciprocal = np.power(outlink_rank_sum, -1, dtype=float)
        outlink_rank_sum_reciprocal = diags(outlink_rank_sum_reciprocal, 0, format="csc") # construct the diagonals of a sparse matrix
        An = An.dot(outlink_rank_sum_reciprocal)
        
        #print("roll!")
        ### pagerank iterations
        pr_pre = 0
        ctr = 0
        while abs(LA.norm(self.pr)-LA.norm(pr_pre)) > self.epsilon:
            pr_pre = np.array(self.pr)
            self.pr = An.dot(self.pr)
            self.pr = self.pr/LA.norm(self.pr)
            ctr += 1
            

    def pagerank(self):
        '''
        pagerank non sparse matrix version
        '''
        ### form the unnormalized matrix
        An = -1*np.ones((self.graph_size, self.graph_size))
        for i in range(self.graph_size):
            for j in range(self.graph_size):
                if i != j:
                    ### column --> node's outgoing links
                    word = self.word_list[i]
                    neighbor = self.word_list[j]
                    word_pair = word.split(' ')
                    neighbor_pair = neighbor.split(' ')
                    if self.complete_graph == False:
                        if (re.search('^context.*', word_pair[1]) != None)                         or (re.search('^context.*', neighbor_pair[1]) != None):
                            if An[i][j] == -1 and An[i][j] == -1: 
                                word_vec = get_vec(word_pair[0], self.glove_dict)
                                neighbor_vec = get_vec(neighbor_pair[0], self.glove_dict)
                                An[i][j] = get_similarity_cosine(word_vec, neighbor_vec)
                                An[j][i] = An[i][j]
                        else:
                            An[i][j] = 0
                    else:
                        if An[i][j] == -1 and An[j][i] == -1: 
                            word_vec = get_vec(word_pair[0], self.glove_dict)
                            neighbor_vec = get_vec(neighbor_pair[0], self.glove_dict)
                            An[i][j] = get_similarity_cosine(word_vec, neighbor_vec)
                            An[j][i] = An[i][j]
                else:
                    An[i][j] = 0
        
        ### form the normalized matrix
        for i in range(self.graph_size):
            outlink_rank_sum = sum(An[:,i])
            ### if a node is a sink (never a case in this use scenario)
            if outlink_rank_sum == 0:
                An[:,i] = np.ones(self.graph_size)*1.0/self.graph_size
            ### normalize the outgoing links
            else:
                An[:,i] = [An[j][i]*1.0/outlink_rank_sum for j in range(self.graph_size)]
        if self.graph_size != 0:
            G = self.decay*An+(1.0-self.decay)/self.graph_size
        else:
            G = An

        ### pagerank iterations
        pr_pre = 0
        ctr = 0
        while abs(LA.norm(self.pr)-LA.norm(pr_pre)) > self.epsilon:
            pr_pre = np.array(self.pr)
            self.pr = G.dot(self.pr)
            self.pr = self.pr/LA.norm(self.pr)
            ctr += 1
   

    def update_oov_candidates(self):
        '''
        update rank for each candidate in oov_candidates
        '''
        if self.candidate_source is "ug_dict":
            for i in range(self.graph_size):
                pair = self.word_list[i].split(' ')
                candidate = pair[0]
                oov = pair[1]
                if oov in self.oov_candidates:
                    self.oov_candidates[oov][candidate] = self.pr[i]
        elif self.candidate_source is "eng_vocab":
            for i in range(self.graph_size):
                pair = self.word_list[i].split(' ')
                candidate = pair[0]
                sym = pair[1]
                if sym == "oov":
                    self.oov_candidates[candidate] = self.pr[i]
                    ### use a fixed size heap to store rank-candidate tuples
                    if len(self.rank_heap) < self.oov_num:
                        heapq.heappush(self.rank_heap, (self.pr[i], candidate))
                    else:
                        spilled = heapq.heappushpop(self.rank_heap, (self.pr[i], candidate))
        
        
        
    def translate_pagerank(self, tra_tok, oov_pos):
        '''
        translate oov words using their ranks derived by pagerank matrix version
        '''
        tra_tok_new = list(tra_tok)
        
        if self.candidate_source is "ug_dict":
            for i in range(len(tra_tok)):
                oov = tra_tok[i]
                if oov in self.oov_candidates and oov not in self.oov_candidates[oov]:
                    candidates = self.oov_candidates[oov]
                    best_candidate = max(candidates.keys(), key=(lambda k: candidates[k]))
                    tra_tok_new[i] = best_candidate
                    
        elif self.candidate_source is "eng_vocab":
            ### just sort a list of about the size of the number of oovs
            ### the last element is the one with the highest pagerank, so do pop later
            self.rank_heap = sorted(self.rank_heap)
            
            for i in range(len(tra_tok)):
                if i in oov_pos and tra_tok[i] not in self.oov_candidates:
                    best_candidate = self.rank_heap.pop()[1]
                    tra_tok_new[i] = best_candidate
        
        return ' '.join(tra_tok_new)
                    
    
    
    def rank_and_translate(self, tra_tok, oov_pos, context):
        '''
        rank + translate
        '''
        if self.candidate_source == "eng_vocab":
            if context != []:
                ### pagerank() or pagerank_sparse()
                self.pagerank_sparse()
        elif self.candidate_source == "ug_dict":
            self.pagerank()
        self.update_oov_candidates()
        res = self.translate_pagerank(tra_tok, oov_pos)
        return res
                  
            
#     ### pagerank iteration version
#     def rank(self):
#         num_iteration = 10
#         for i in range(num_iteration):
#             for node in self.nodes:
#                 rank_sum = 0
#                 for neighbor in self.nodes[node]:
#                     if neighbor != '_rank_':
#                         rank_sum += \
#                             1.0/sum(self.nodes[neighbor].values()) * \
#                             self.nodes[neighbor][node] * \
#                             self.nodes[neighbor]['_rank_']
#                 self.nodes[node]['_rank_'] = \
#                 (1.0-self.decay) * \
#                 1.0/self.graph_size + \
#                 self.decay * rank_sum
    
    
    
#     ### translate oov words using their ranks
#     def translate_rank(self, tra_tok, oov_candidates):
#         tra_tok_new = list(tra_tok)
        
#         if self.candidate_source is "ug_dict":
#             for i in range(len(tra_tok)):
#                 oov = tra_tok[i]
#                 if oov in oov_candidates and oov not in oov_candidates[oov]:
#                     candidates = oov_candidates[oov]
#                     best_candidate = None
#                     best_score = -float("inf")
#                     for candidate in candidates:
#                         score =  self.nodes[candidate+" "+oov]['_rank_']
#                         if score > best_score:
#                             best_score = score
#                             best_candidate = candidate
#                     tra_tok_new[i] = best_candidate
                    
#         elif self.candidate_source is "eng_vocab":
#             oov_real = {tra_tok[i]:None for i in oov_pos if tra_tok[i] not in oov_candidates}
#             candidate_real = {}
#             for word in oov_candidates:
#                 if word not in context_words_set:
#                     rank = self.nodes[word+" oov"]['_rank_']
#                     candidate_real[word] = rank
#             sorted_candidate_real = sorted(candidate_real.items(), key=operator.itemgetter(1))
#             ctr = 0
#             for oov in oov_real:
#                 candidate = sorted_candidate_real[ctr][0]
#                 oov_real[oov] = candidate
#                 ctr += 1
#             for i in range(len(tra_tok)):
#                 if tra_tok[i] in oov_real:
#                     tra_tok_new[i] = oov_real[tra_tok[i]]
                    
#         return ' '.join(tra_tok_new)



def oov_trans_pagerank(candidate_source,                        add_aligned_oov,                        complete_graph,                        res_file):
    '''
    params:
        candidate_source: ug_dict or eng_vocab 
        add_aligned_oov: True or False
        complete_graph: True or False
        res_file: path to oov translation result
    return:
        None
    '''
    eng_vocab = None
    ug_dict = None
    if candidate_source is "ug_dict":
        ug_dict = get_ug_dict(oov_candidates_file, 0)
    elif candidate_source is "eng_vocab":
        eng_vocab = get_eng_vocab(eng_vocab_file)
    
    print("loading GloVe...")
    glove_dict = load_glove(300)
    print("GloVe loaded.")

    multithread_routine([candidate_source, add_aligned_oov, complete_graph],                         res_file,                         [eng_vocab, ug_dict, glove_dict],                         tmp_dir,                         PAGERANK)

class PAGERANK (threading.Thread):
    def __init__(self,                  candidate_source,                  add_aligned_oov,                  complete_graph,                  res_file,                  ctr_lo,                  ctr_up,                  eng_vocab,                  ug_dict,                  glove_dict):
        
        threading.Thread.__init__(self)
        
        ### method params
        self.candidate_source = candidate_source
        self.add_aligned_oov = add_aligned_oov
        self.complete_graph = complete_graph
        
        ### one thread writes to one temporary file, later to be merged
        self.res_file = res_file 
        
        ### lower and upper bounds of instance indices
        self.ctr_lo = ctr_lo
        self.ctr_up = ctr_up
        
        ### established, cached resources, passed as arguments from outside
        self.eng_vocab = eng_vocab
        self.ug_dict = ug_dict
        self.glove_dict = glove_dict
        
    def run(self):
        ctr = 0
        with open(tra_file) as ft,         open(oov_file) as fo,         open(self.res_file, 'w') as fres:
            for l_tra in ft:
                l_oov = fo.readline()
                if ctr >= self.ctr_lo and ctr <= self.ctr_up:
                    ### unescaping html not happening
                    tra_tok, oov_pos, context = get_context_oov_pos(l_tra, l_oov)

                    word_graph = WordGraph(tra_tok,oov_pos,context,self.glove_dict,self.eng_vocab,self.ug_dict,self.candidate_source,self.add_aligned_oov,self.complete_graph)
                    res = word_graph.rank_and_translate(tra_tok, oov_pos, context)
                    
                    print(ctr)
                    print(res)
                    fres.write(res+'\n')

                ctr += 1


# In[5]:

# -------- hyperparameters specific to this method --------
### candidate_source is either "eng_vocab" or "ug_dict"
candidate_source = "ug_dict"
### whether to add aligned oov, only applied in "ug_dict"
add_aligned_oov = False
### whether the word graph is complete or not, only applied in "ug_dict"
complete_graph = False


# -------- write --------
res_file = ".".join([tra_file,"oovtranslated",candidate_source,"pagerank"])
if candidate_source is "ug_dict":
    if add_aligned_oov:
        res_file = ".".join([tra_file,"oovtranslated",candidate_source+"_withAlignedOov","pagerank"])
    else:
        res_file = ".".join([tra_file,"oovtranslated",candidate_source+"_withoutAlignedOov","pagerank"])
if complete_graph == False:
    ### not suitable for eng_vocab due to computational constraint
    res_file += "_incomplete_graph"

    
# -------- translate --------
oov_trans_pagerank(candidate_source, add_aligned_oov, complete_graph, res_file)




if __name__ == "__main__":
    ocp = oov_candidates_preprocessing()
    
    complete_graph = False
    
    # dev, test, domain, eval
    for dataset in {"dev", "test", "domain", "eval"}:
        # extracted, eng_vocab, extracted_eng_vocab, glosbe
        for candidate_source in {"extracted_eng_vocab"}:
            # oov pos:w1,w2 file prep
            data_in_domain_xml, _, onebest_file, _, _, candidate_list_file = ocp.init(dataset, candidate_source)
            for complete_graph in {True, False}:
                res_attr = "complete" if complete_graph else "incomplete"
                res_file = os.path.join(exp_dir,"translation",dataset,".".join([res_attr,t,dataset,yrv]))
                print('--------')
                rescore_lattice()



