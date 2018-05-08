#include <dynet/nodes.h>
#include <dynet/dynet.h>
#include <dynet/expr.h> 
   
#include "util.hpp"
#include "latticedec.h" 
#include "charlm.h"

using namespace dynet;
using namespace std;
 
dynet::Dict d;
int kEOS, kSOS;
 
float thres_dist = 0.5;
 
/* rnnlm */
template <class Builder> class RNNLanguageModel {
private:
    /* word embeddings VxK1 */
    LookupParameter p_c;
    /* recurrence weights VxK2 */
    Parameter p_R;
    /* bias Vx1 */
    Parameter p_bias;

    Builder builder;

public:
    RNNLanguageModel();

    /* constructor */
    RNNLanguageModel(
        Model& model,
        unsigned vocab_size,
        unsigned num_layer,
        unsigned input_dim,
        unsigned hidden_dim
    ) : builder(num_layer, input_dim, hidden_dim, model) {
        p_c = model.add_lookup_parameters(vocab_size, {input_dim});
        p_R = model.add_parameters({vocab_size, hidden_dim});
        p_bias = model.add_parameters({vocab_size});
    }


    // beam search
    unsigned decode_beam(const Doc doc,
        ComputationGraph& cg,
        Dict* dptr,
        const DocStr doc_candidate,
        const DocStr doc_str,
        ofstream& out,
        unsigned beam_size,

        bool include_charlm,
        charlm_lstm& charlm) {

        builder.new_graph(cg);
        Expression i_R = parameter(cg, p_R);
        Expression i_bias = parameter(cg, p_bias);
        Expression i_x_t, i_y_t, i_r_t, i_err;

        // loop through all sentences in the doc
        for (unsigned k=0;k<doc.size();k++) {
            builder.start_new_sequence();
            // sent with <s> and </s>
            auto sent = doc[k];
            unsigned sent_len = sent.size();

            // map oov_pos to oov candidates (in int)
            map<unsigned,set<int>> oov_candidates=get_oov_candidates(doc_candidate[k], dptr);
            // map oov_pos to oov candidates (in string)
            map<unsigned,vector<string>> oov_candidates_str=get_oov_candidates_str(doc_candidate[k]);

            // double: loss on the incoming arc
            map<double, set<expanded_node*>> path_cache;
            // string: node name
            map<string, expanded_node*> expanded_nodes;

            // initialize expanded_nodes and path_cache with the first node
            expanded_node* n00 = new expanded_node("n00", nullptr);
            expanded_nodes["n00"]=n00;

            // loss heap (keep track of paths with small losses)
            vector<double> loss_heap = {};
            make_heap(loss_heap.begin(), loss_heap.end());

            // initialize the outbound arc list (with respect to the original graph, rather than the expanded graph)
            vector<set<int>> outbound_arc(sent_len, set<int>());
            for (unsigned t=0; t<sent_len; t++) {
                if (oov_candidates.count(t)) {
                    outbound_arc[t] = oov_candidates[t];
                } else {
                    outbound_arc[t] = {sent[t]};
                }
            }

            // loop through all words in the sentence
            for (unsigned t=0; t<sent_len; t++) {

                unsigned i=0;

                // string: node name; used to keep track of newly created nodes at the frontier
                map<string, expanded_node*> expanded_nodes_new;
                unsigned arc_num = outbound_arc[t].size();

                // enumerate all nodes in expanded nodes
                for (auto e_node:expanded_nodes) {

                    // n_pre is nullptr when t = 0
                    auto n_pre = e_node.second;

                    // count expanded nodes
                    unsigned j = 0;

                    // enumerate all words in outbound arcs (with respect to the original graph)
                    for (auto arc_word:outbound_arc[t]) {

                        RNNPointer state_cur = (RNNPointer)(-1);
                        double loss = 0.0;
                        // context vector could come from the previous sentence (not in the case of RNNLM)
                        vector<float> context_vec(p_R.dim().rows(), 0.0);

                        if (t != 0) {
                            // p_c: vocab_suze * input_dim
                            i_x_t = lookup(cg, p_c, n_pre->get_in_arc()->get_word());
                            // hidden representation
                            i_y_t = builder.add_input(n_pre->get_in_arc()->get_state(), i_x_t);
                            // transform from hidden representation to vocab dimension
                            i_r_t = i_bias + i_R * i_y_t;
                            // compare the prediction against the reference
                            i_err = pickneglogsoftmax(i_r_t, arc_word);

                            state_cur = builder.state();
                            // dimension: hidden_dim
                            context_vec = as_vector(cg.incremental_forward(i_y_t));
                            // neg log probability
                            loss = as_scalar(cg.incremental_forward(i_err)) + n_pre->get_in_arc()->get_loss();
                        }

                        // find a path with less loss
                        if ((loss_heap.size()!=0 && loss<loss_heap.front()) || loss_heap.size() == 0) {

                            // create a new expanded node
                            // i: index of an expanded_node
                            // j: index of an outbound_arc
                            string node_name = "n"+to_string(t+1)+to_string(i*arc_num+j);
                            expanded_node* n_cur = new expanded_node(node_name, nullptr);
                            //cout<<"created node "<<node_name<<endl;

                            // create a new expanded arc
                            string arc_name = "a"+to_string(t+1)+to_string(i*arc_num+j);
                            expanded_arc* e_arc = new expanded_arc(arc_name, arc_word, n_pre, n_cur);
                            e_arc->set_state(state_cur);
                            e_arc->set_loss(loss);
                            e_arc->set_context_vec(context_vec);

                            // push the loss into the heap
                            loss_heap.push_back(loss);
                            push_heap(loss_heap.begin(), loss_heap.end());

                            // add the new expanded node into path_cache
                            if (!path_cache.count(loss)) {
                                path_cache[loss] = {n_cur};
                            } else {
                                path_cache.at(loss).insert(n_cur);
                            }
                            expanded_nodes_new[node_name] = n_cur;

                            // keep the beam size fixed
                            if (loss_heap.size()>beam_size) {

                                // obtain the node to be removed
                                double popped_loss = loss_heap.front();
                                pop_heap(loss_heap.begin(), loss_heap.end());
                                loss_heap.pop_back();
                                auto popped_node_ptr = path_cache.at(popped_loss).begin(); // this is why path_cache is used here
                                expanded_node* popped_node = *popped_node_ptr;

                                // remove one path from path_cache: map<double, set<expanded_node*>> path_cache
                                path_cache.at(popped_loss).erase(popped_node_ptr);
                                if (path_cache.at(popped_loss).size() == 0) {
                                    path_cache.erase(popped_loss);
                                }
                                expanded_nodes_new.erase(popped_node->get_name());

                                // remove the actual path from the expanded lattice
                                if (popped_node->get_in_arc()->get_out_node()->get_name() != n_pre->get_name()) {
                                    popped_node->del_path();
                                } else {
                                    popped_node->del_by_depth_1();
                                }
                            }
                        }
                        j++;
                    }
                    i++;
                }
                loss_heap.clear();
                path_cache.clear();
                expanded_nodes.clear();
                expanded_nodes = expanded_nodes_new;
            }
            vector<int> best_path = {};
            if (sent_len>0) {
                double best_loss = 9e+99;
                expanded_node* best_leaf = nullptr;
                for (auto nn:expanded_nodes) {
                    auto e_node = nn.second;
                    double loss = e_node->get_in_arc()->get_loss();
                    if (loss<best_loss) {
                        best_loss = loss;
                        best_leaf = e_node;
                    }
                }
                best_path = best_leaf->get_lineage();

                string char_out = "";
                // print the decoded sentence
                for (unsigned i=1;i<best_path.size()-1;i++) {
                    if (i>1) {cout<<" ";out<<" ";char_out=char_out+" ";}
                    string w;
                    if (oov_candidates.count(i)) {
                        w=d.convert(best_path[i]);
                        if (w=="UNK") {
                            if (!include_charlm) {
                                w=oov_candidates_str.at(i).back();
                            } else {
                                vector<string> char_outs = oov_candidates_str.at(i);
                                for (unsigned j=0;j<char_outs.size();++j) {
                                    char_outs[j] = char_out+char_outs[j];
                                }
                                cg.checkpoint();
                                auto selected = charlm.score_pick(cg, char_outs);
                                cg.revert();
                                cout<<"<<selected index: "<<selected.best_idx<<">>";
                                w=(oov_candidates_str.at(i))[selected.best_idx];
                            }
                        }
                    } else {
                        w=doc_str[k][i-1];
                    }
                    cout<<w;
                    out<<w;
                    char_out=char_out+w;
                }
                cout<<endl;
                out<<endl;
                char_out=char_out+"\n";
            }
        }
        return 0;
    }

    // use word embeddings as the hidden representation
    // unsigned decode_embed(
    //     const Doc doc,
    //     ComputationGraph& cg,
    //     Dict* dptr,

    //     const DocStr doc_candidate,
    //     const DocStr doc_str,

    //     ofstream& out,
    //     unsigned beam_size,

    //     bool include_charlm,
    //     charlm_lstm& charlm) {

    //     builder.new_graph(cg);

    //     /*p_c = model.add_lookup_parameters(vocab_size, {input_dim});
    //     p_R = model.add_parameters({vocab_size, hidden_dim});
    //     p_bias = model.add_parameters({vocab_size});*/
    //     Expression i_R = parameter(cg, p_R);
    //     Expression i_bias = parameter(cg, p_bias);
    //     //Expression i_e_t;
    //     Expression i_x_t, i_y_t, i_r_t, i_err;

    //     unsigned sent_len_all = 0;

    //     // loop through all the sentences in the doc
    //     for (unsigned k=0; k<doc.size(); k++) {
    //         builder.start_new_sequence();
    //         // sent includes <s> and </s>
    //         auto sent = doc[k];
    //         unsigned sent_len = sent.size();

    //         map<unsigned, set<int>> oov_candidates = get_oov_candidates(doc_candidate[k], dptr);
    //         map<unsigned, vector<string>> oov_candidates_str = get_oov_candidates_str(doc_candidate[k]);

    //         map<int, set<expanded_node*>> path_cache;
    //         map<string, expanded_node*> expanded_nodes_pre;

    //         expanded_node* n_pre = new expanded_node("n00", nullptr);
    //         expanded_nodes_pre["n00"] = n_pre;
    //         int a_pre_word = -1;
    //         path_cache[a_pre_word] = {n_pre};

    //         // length: <s> + sentence + </s>
    //         vector<set<int>> outbound_arc(sent_len, set<int>());
    //         for (unsigned t=0;t<sent_len;t++) {
    //             if (oov_candidates.count(t)) outbound_arc[t] = oov_candidates[t];
    //             else outbound_arc[t] = {sent[t]};
    //         }

    //         // set RNN state to -1 at the beginning of each sentence
    //         RNNPointer state_cur = (RNNPointer)(-1);

    //         for (unsigned t=0;t<sent_len;++t) {
    //             map<string, expanded_node*> expanded_nodes_cur;
    //             unsigned a_cur_num = outbound_arc[t].size();

    //             unsigned i = 0;
    //             for (auto n_pre_pair:expanded_nodes_pre) {
    //                 n_pre = n_pre_pair.second;

    //                 unsigned j = 0;
    //                 for (int a_cur_word:outbound_arc[t]) {
                        
    //                     double loss_cur = 0.0;
    //                     // hidden_dim
    //                     vector<float> context_vec(p_R.dim().rows(), 0.0);   

    //                     if (t!=0) {
    //                         a_pre_word = n_pre->get_in_arc()->get_word();
    //                         cout<<"word: "<<a_pre_word<<endl;
    //                         auto state_pre = n_pre->get_in_arc()->get_state();

    //                         // look up for the expression
    //                         //i_x_t = lookup(cg, p_c, a_pre_word);
    //                         cout<<"get here"<<endl;
    //                         // add history embeddings to the current and feed the sum to LSTM
    //                         Expression i_e_t;
    //                         vector<int> words_pre = n_pre->get_lineage();
    //                         unsigned l=0;
    //                         for (auto word_pre:words_pre) {
    //                             if (l==0)
    //                                 i_e_t = lookup(cg, p_c, word_pre);
    //                             else {
    //                                 i_e_t = i_e_t + lookup(cg, p_c, word_pre);
    //                                 cout<<"history way back"<<endl;
    //                             }
    //                             l++;
    //                         }
    //                         //i_e_t = i_x_t + n_pre->get_in_arc()->get_embed();
    //                         cout<<"get here 2"<<endl;
    //                         i_y_t = builder.add_input(state_pre, i_e_t / t);//float(sent_len_all+t));
    //                         cout<<"get here 3"<<endl;
    //                         i_r_t = i_bias + i_R * (i_y_t);
    //                         i_err = pickneglogsoftmax(i_r_t, a_cur_word);

    //                         state_cur = builder.state();
    //                         context_vec = as_vector(cg.incremental_forward(i_y_t));
    //                         loss_cur = as_scalar(cg.incremental_forward(i_err)) + n_pre->get_in_arc()->get_loss();
    //                     }

    //                     // if there isn't an expanded node in path_cache that shares the incoming word with the current expanded node
    //                     if (!path_cache.count(a_cur_word)) {
    //                         // remove the previous expanded node
    //                         if (t==0) 
    //                             path_cache.erase(a_pre_word);
    //                         else {
    //                             a_pre_word = n_pre->get_in_arc()->get_word();
    //                             if (path_cache.count(a_pre_word)) {
    //                                 path_cache.at(a_pre_word).erase(n_pre);
    //                                 if (path_cache.at(a_pre_word).size()==0)
    //                                     path_cache.erase(a_pre_word);
    //                             }
    //                         }

    //                         // create a new expanded node and arc
    //                         string n_cur_name = "n"+to_string(t+1)+to_string(i*a_cur_num+j);
    //                         expanded_node* n_cur = new expanded_node(n_cur_name, nullptr);
    //                         string a_cur_name = "a"+to_string(t+1)+to_string(i*a_cur_num+j);
    //                         expanded_arc* a_cur = new expanded_arc(a_cur_name, a_cur_word, n_pre, n_cur);

    //                         a_cur->set_state(state_cur);
    //                         a_cur->set_loss(loss_cur);
    //                         a_cur->set_context_vec(context_vec);
    //                         //a_cur->set_embed(i_e_t);

    //                         path_cache[a_cur_word] = {n_cur};
    //                         expanded_nodes_cur[n_cur_name] = n_cur;

    //                     // if there exists expanded nodes in path_cache that share the incoming word with the current expanded node
    //                     } else {
    //                         auto n_cur_competings = path_cache.at(a_cur_word);
    //                         bool to_merge = false;
    //                         for (auto n_cur_competing:n_cur_competings) {
    //                             if (!expanded_nodes_pre.count(n_cur_competing->get_name())) {
    //                                 vector<float> context_vec_competing = n_cur_competing->get_in_arc()->get_context_vec();
    //                                 float contexts_dist = dist(context_vec_competing, context_vec);

    //                                 if (contexts_dist<thres_dist) {
    //                                     double loss_competing = n_cur_competing->get_in_arc()->get_loss();
    //                                     if (loss_cur<loss_competing) {
    //                                         path_cache.at(a_cur_word).erase(n_cur_competing);
    //                                         expanded_nodes_cur.erase(n_cur_competing->get_name());
    //                                         if (n_cur_competing->get_in_arc()->get_out_node()->get_name() != n_pre->get_name())
    //                                             n_cur_competing->del_path();
    //                                         else
    //                                             n_cur_competing->del_by_depth_1();

    //                                         if (t==0)
    //                                             path_cache.erase(a_pre_word);
    //                                         else {
    //                                             a_pre_word = n_pre->get_in_arc()->get_word();
    //                                             if (path_cache.count(a_pre_word)) {
    //                                                 path_cache.at(a_pre_word).erase(n_pre);
    //                                                 if (path_cache.at(a_pre_word).size()==0)
    //                                                     path_cache.erase(a_pre_word);
    //                                             }
    //                                         }

    //                                         // create a new expanded node and arc
    //                                         string n_cur_name = "n"+to_string(t+1)+to_string(i*a_cur_num+j);
    //                                         expanded_node* n_cur = new expanded_node(n_cur_name, nullptr);
    //                                         string a_cur_name = "a"+to_string(t+1)+to_string(i*a_cur_num+j);
    //                                         expanded_arc* a_cur = new expanded_arc(a_cur_name, a_cur_word, n_pre, n_cur);

    //                                         a_cur->set_state(state_cur);
    //                                         a_cur->set_loss(loss_cur);
    //                                         a_cur->set_context_vec(context_vec);
    //                                         //a_cur->set_embed(i_e_t);

    //                                         if (!path_cache.count(a_cur_word))
    //                                             path_cache[a_cur_word]={n_cur};
    //                                         else
    //                                             path_cache.at(a_cur_word).insert(n_cur);
    //                                     } else {
    //                                         auto a_cur_siblings = n_pre->get_out_arcs();
    //                                         unsigned sibling_num = (*a_cur_siblings).size();
    //                                         if ((sibling_num==0) && (j==a_cur_num-1)) {
    //                                             if (t==0)
    //                                                 path_cache.erase(a_pre_word);
    //                                             else {
    //                                                 a_pre_word = n_pre->get_in_arc()->get_word();
    //                                                 if (path_cache.count(a_pre_word)) {
    //                                                     path_cache.at(a_pre_word).erase(n_pre);
    //                                                     if (path_cache.at(a_pre_word).size()==0)
    //                                                         path_cache.erase(a_pre_word);
    //                                                 }
    //                                             }

    //                                             n_pre->del_path();
    //                                         }
    //                                     }
    //                                     to_merge = true;
    //                                     break;
    //                                 }
    //                             }
    //                         }

    //                         if (!to_merge) {
    //                             if (t==0) 
    //                                 path_cache.erase(a_pre_word);
    //                             else {
    //                                 auto a_pre_word = n_pre->get_in_arc()->get_word();
    //                                 if (path_cache.count(a_pre_word)) {
    //                                     path_cache.at(a_pre_word).erase(n_pre);
    //                                     if (path_cache.at(a_pre_word).size()==0)
    //                                         path_cache.erase(a_pre_word);
    //                                 }
    //                             }

    //                             string n_cur_name = "n"+to_string(i+1)+to_string(i*a_cur_num+j);
    //                             expanded_node* n_cur = new expanded_node(n_cur_name, nullptr);
    //                             string a_cur_name = "a"+to_string(i+1)+to_string(i*a_cur_num+j);
    //                             expanded_arc* a_cur = new expanded_arc(a_cur_name, a_cur_word, n_pre, n_cur);
                                
    //                             a_cur->set_state(state_cur);
    //                             a_cur->set_loss(loss_cur);
    //                             a_cur->set_context_vec(context_vec);
    //                             //a_cur->set_embed(i_e_t);

    //                             if (!path_cache.count(a_cur_word))
    //                                 path_cache[a_cur_word]={n_cur};
    //                             else
    //                                 path_cache.at(a_cur_word).insert(n_cur);
    //                             expanded_nodes_cur[n_cur_name]=n_cur;
    //                         }
    //                     }
    //                     j++;
    //                 }
    //                 i++;
    //             }
    //             expanded_nodes_pre.clear();
    //             expanded_nodes_pre = expanded_nodes_cur;
    //         }

    //         vector<int> best_path = {};
    //         if (sent_len>0) {
    //             double best_loss = 9e+99;
    //             expanded_node* best_leaf = nullptr;
    //             for (auto n_pre_pair:expanded_nodes_pre) {
    //                 n_pre = n_pre_pair.second;
    //                 double loss_pre = n_pre->get_in_arc()->get_loss();
    //                 if (loss_pre<best_loss) {
    //                     best_loss = loss_pre;
    //                     best_leaf = n_pre;
    //                 }
    //             }
    //             best_path = best_leaf->get_lineage();

    //             string char_out = "";
    //             for (unsigned t=1;t<best_path.size()-1;++t) {
    //                 if (t>1){cout<<" ";out<<" ";char_out=char_out+" ";}
    //                 string w;
    //                 if (oov_candidates.count(t)) {
    //                     w=d.convert(best_path[t]);
    //                     if (w=="UNK") {
    //                         if (!include_charlm) {
    //                             w=oov_candidates_str.at(t).back();
    //                         } else {
    //                             vector<string> char_outs = oov_candidates_str.at(t);
    //                             for (unsigned j=0;j<char_outs.size();++j)
    //                                 char_outs[j]=char_out+char_outs[j];
    //                             cg.checkpoint();
    //                             auto selected = charlm.score_pick(cg, char_outs);
    //                             cg.revert();
    //                             cout<<"<<selected index: "<<selected.best_idx<<">>";
    //                             w=(oov_candidates_str.at(t))[selected.best_idx];
    //                         }
    //                     }
    //                 } else {
    //                     w=doc_str[k][t-1];
    //                 }
    //                 cout<<w;
    //                 out<<w;
    //                 char_out=char_out+w;
    //             }
    //             cout<<endl;
    //             out<<endl;
    //             char_out=char_out+"\n";
    //         }

    //         sent_len_all += sent_len;
    //     }
    //     return 0;
    // }

    /* build a rnnlm graph, which returns an Expression */
    unsigned decode_embed(const Doc doc, 
        ComputationGraph& cg, 
        Dict* dptr, 
        // candidates for each OOV words 
        const DocStr doc_candidate, 
        // corpus in string instead of one hot numbers
        const DocStr doc_str, 
        ofstream& out, 
        unsigned beam_size, 

        bool include_charlm, 
        charlm_lstm& charlm) {

        // reset RNN builder for new graph
        builder.new_graph(cg);
        // define expression
        Expression i_R = parameter(cg, p_R);
        /* word bias */
        Expression i_bias = parameter(cg, p_bias);

        Expression i_x_t, i_y_t, i_r_t, i_err;

        //vector<Expression> errs;

        vector<int> word_pre_doc = {};

        /* loop through all sentences in the doc */
        for (unsigned k = 0; k < doc.size(); k++) {
            // start a new sequence for each sentence
            builder.start_new_sequence();
            // for each sentence in this doc
            auto sent = doc[k];
            unsigned sent_len = sent.size();

            
            /* ---------------- */
            //cout << "new sequence!"<<endl;
            //cout<<"candidate size: "<<doc_candidate[k].size()<<endl;
            /*if (doc_candidate[k].empty()) {
                auto sent_str = doc_str[k];
                for (unsigned i=0;i<sent_str.size();i++) {
                    if (i>0) {cout<<" ";out<<" ";}
                    cout<<sent_str[i];
                    out<<sent_str[i];
                }
                cout<<endl;
                out<<endl;
                continue;
            }*/
            map<unsigned,set<int>> oov_candidates=get_oov_candidates(doc_candidate[k], dptr);
            map<unsigned,vector<string>> oov_candidates_str=get_oov_candidates_str(doc_candidate[k]);

            /* map the end point to the hole history to the probability of it and the vector representation of it */
            /* int: word on the incoming arc */
            map<int, set<expanded_node*>> path_cache;

            /* string: node name */
            map<string, expanded_node*> expanded_nodes;

            /* the first node in the lattice */
            expanded_node* n00 = new expanded_node("n00", nullptr);
            expanded_nodes["n00"]=n00;
            int null_word = -1;
            path_cache[null_word] = {n00};

            //cout<<"path_cache and expanded_nodes initialized"<<endl;

            /* initialize the outbound arc list */
            vector<set<int>> outbound_arc(sent_len, set<int>());
            for (unsigned t=0; t<sent_len; t++) {
                if (oov_candidates.count(t)) {
                    outbound_arc[t] = oov_candidates[t];
            } else {
                outbound_arc[t] = {sent[t]};
                }
            }
            //cout<<"outbound_arc initialized"<<endl;       

            /* ---------------- */
            RNNPointer state_cur = (RNNPointer)(-1);

            for (unsigned t = 0; t < sent_len; t++) {
                //cout<<"processing word "<<t<<endl;
                unsigned i = 0;
                map<string, expanded_node*> expanded_nodes_new;
                unsigned arc_num = outbound_arc[t].size();
                //cout<<"expanded_nodes_new created"<<endl;
                for (auto e_node:expanded_nodes) {
                    auto n_pre = e_node.second;
                    unsigned j = 0;
                    //bool skip_node = false;
                    unsigned outbound_arc_num = outbound_arc[t].size();
                    //cout<<"outbound_arc_num: "<<outbound_arc_num<<endl;
                    //cout<<"expanded_node found"<<endl;
                    for (auto arc_word:outbound_arc[t]) {
                        
                        //RNNPointer state_cur = (RNNPointer)(-1);
                        double loss = 0.0;
                        vector<float> context_vec(p_R.dim().rows(), 0.0);
                        
                        //cout << "start expanding lattice"<<endl;
                        
                        if (t != 0) {
                            //cout<<"n_pre: "<<n_pre->get_name()<<endl;
                            //i_x_t = lookup(cg, p_c, n_pre->get_in_arc()->get_word());
                            Expression i_e_t;
                            vector<int> words_pre = n_pre->get_lineage();
                            unsigned l=0;
                            for (auto word_pre0:words_pre) {
                                if (l==0)
                                    i_e_t = lookup(cg, p_c, word_pre0);
                                else {
                                    i_e_t = i_e_t + lookup(cg, p_c, word_pre0);
                                    //cout<<"history way back"<<endl;
                                }
                                l++;
                            }
                            for (auto ww:word_pre_doc)
                                i_e_t = i_e_t + lookup(cg, p_c, ww);
                            i_e_t = i_e_t / float(t+word_pre_doc.size());

                            i_y_t = builder.add_input(n_pre->get_in_arc()->get_state(), i_e_t);//i_x_t);
                            i_r_t = i_bias + i_R * i_y_t;
                            i_err = pickneglogsoftmax(i_r_t, arc_word);

                            /* ---------------- */
                            state_cur = builder.state();
                            context_vec = as_vector(cg.incremental_forward(i_y_t));
                            loss = as_scalar(cg.incremental_forward(i_err)) + n_pre->get_in_arc()->get_loss();
                            
                        }
                        //cout<<"state: "<<state_cur<<endl;
                        
                        /* create a new expanded node */
                        string node_name = "n"+to_string(t+1)+to_string(i*arc_num+j);
                        expanded_node* n_cur = new expanded_node(node_name, nullptr);
                        //cout << "build node ["<<node_name<<"] following: \"" << d.convert(arc_word) <<"\" and node ["<<n_pre->get_name()<<"]"<< endl;
                        
                        /* there isn't a path in cache that can represent the expanded node */
                        if (!path_cache.count(arc_word)) {
                            /* remove the previous expanded node */
                            if (t==0) {
                                path_cache.erase(null_word);
                            } else {
                                auto word_pre = n_pre->get_in_arc()->get_word();
                                if (path_cache.count(word_pre)) {
                                    path_cache.at(word_pre).erase(n_pre);
                                    if (path_cache.at(word_pre).size()==0) path_cache.erase(word_pre);
                                }
                            }
                            
                            /* create a new expanded arc */
                            string arc_name = "a"+to_string(t+1)+to_string(i*arc_num+j);
                            expanded_arc* e_arc = new expanded_arc(arc_name, arc_word, n_pre, n_cur);
                            e_arc->set_state(state_cur);
                            e_arc->set_loss(loss);
                            e_arc->set_context_vec(context_vec);
                            
                            if (!path_cache.count(arc_word)) path_cache[arc_word]={n_cur};
                            else path_cache[arc_word]={n_cur};
                            expanded_nodes_new[node_name]=n_cur;
                            
                        /* there is a path in cache that can represent the expanded node */
                        } else {
                            auto leaf_nodes = path_cache.at(arc_word);
                            bool to_merge = false;
                            for (auto leaf_node:leaf_nodes) {
                                if (!expanded_nodes.count(leaf_node->get_name())) {
                                    
                                    vector<float> context_vec_competing = leaf_node->get_in_arc()->get_context_vec();
                                    float contexts_dist = dist(context_vec_competing, context_vec);

                                    //cout<<"dist: "<<contexts_dist<<endl;
                                    if (contexts_dist<thres_dist) {
                                        //cout << "merge node [" << leaf_node->get_name() << "] and [" << node_name << "] following \"" <<d.convert(arc_word)<<"\" that emerges from node ["<< leaf_node->get_in_arc()->get_out_node()->get_name()<<"] and ["<<n_pre->get_name()<<"]"<<endl;
                                        double loss_competing = leaf_node->get_in_arc()->get_loss();
                                        //cout<<"loss at node ["<< leaf_node->get_name() <<"]: "<<loss_competing<<endl;
                                        //cout<<"loss at node ["<< node_name<<"]: "<<loss<<endl;
                                        
                                        /* replace the path with the expanded node */
                                        if (loss < loss_competing) {
                                            /* remove the competing expanded node */
                                            path_cache.at(arc_word).erase(leaf_node);
                                            expanded_nodes_new.erase(leaf_node->get_name());
                                            
                                            //cout << "remove node [" << leaf_node->get_name() << "]" << endl;
                                            if (leaf_node->get_in_arc()->get_out_node()->get_name() != n_pre->get_name()) {
                                                leaf_node->del_path();
                                            } else {
                                                leaf_node->del_by_depth_1();
                                            }
                                            
                                            /* remove the previous expanded node */
                                            if (t==0) {
                                                path_cache.erase(null_word);
                                            } else {
                                                auto word_pre = n_pre->get_in_arc()->get_word();
                                                if (path_cache.count(word_pre)) {
                                                    path_cache.at(word_pre).erase(n_pre);
                                                    if (path_cache.at(word_pre).size()==0) path_cache.erase(word_pre);
                                                }
                                            }
                                            
                                            /* create a new expanded arc */
                                            string arc_name = "a"+to_string(t+1)+to_string(i*arc_num+j);
                                            expanded_arc* e_arc = new expanded_arc(arc_name, arc_word, n_pre, n_cur);
                                            e_arc->set_state(state_cur);
                                            e_arc->set_loss(loss);
                                            e_arc->set_context_vec(context_vec);
                                            
                                            if (!path_cache.count(arc_word)) path_cache[arc_word]={n_cur};
                                            else path_cache.at(arc_word).insert(n_cur);
                                            expanded_nodes_new[node_name]=n_cur;

                                        /* keep the existing path */
                                        } else {
                                            //*n_cur=*leaf_node;
                                            
                                            auto out_arcs = n_pre->get_out_arcs();
                                            unsigned out_arcs_num = (*out_arcs).size();
                                            if ((out_arcs_num == 0) && (j == outbound_arc_num-1)) {
                                                //cout << "remove node [" << n_pre->get_name() << "]" << endl;
                                                
                                                /* remove the previous expanded node */
                                                if (t==0) {
                                                    path_cache.erase(null_word);
                                                } else {
                                                    auto word_pre = n_pre->get_in_arc()->get_word();
                                                    if (path_cache.count(word_pre)) {
                                                        path_cache.at(word_pre).erase(n_pre);
                                                        if (path_cache.at(word_pre).size()==0) path_cache.erase(word_pre);
                                                    }
                                                }
                                                
                                                n_pre->del_path();                                                 
                                                //skip_node = true;
                                            }
                                        }
                                        to_merge = true;
                                        break;
                                    } else {
                                        //cout << "not merge node [" << leaf_node->get_name() << "] and [" << node_name << "] following \"" <<d.convert(arc_word)<<"\" that emerges from node ["<< leaf_node->get_in_arc()->get_out_node()->get_name()<<"] and ["<<n_pre->get_name()<<"]"<<endl;
                                    }
                                }
                            }
                            /* there isn't a path in cache that can represent the expanded node */
                            if (!to_merge) {
                                //cout << "not merge [" << node_name << "] with any node following \"" <<d.convert(arc_word)<<"\""<< endl;
                                /* remove the previous expanded node */
                                if (t==0) {
                                    path_cache.erase(null_word);
                                } else {
                                    auto word_pre = n_pre->get_in_arc()->get_word();
                                    if (path_cache.count(word_pre)) {
                                        path_cache.at(word_pre).erase(n_pre);
                                        if (path_cache.at(word_pre).size()==0) path_cache.erase(word_pre);
                                    }
                                }
                                
                                /* create a new expanded arc */
                                string arc_name = "a"+to_string(t+1)+to_string(i*arc_num+j);
                                expanded_arc* e_arc = new expanded_arc(arc_name, arc_word, n_pre, n_cur);
                                e_arc->set_state(state_cur);
                                e_arc->set_loss(loss);
                                e_arc->set_context_vec(context_vec);
                                
                                if (!path_cache.count(arc_word)) path_cache[arc_word]={n_cur};
                                else path_cache.at(arc_word).insert(n_cur);
                                expanded_nodes_new[node_name]=n_cur;
                            }
                        }
                        /*n00->dfs();
                        cout<<"--------"<<endl;
                        cout << "path_cache: "<<endl; for (auto pp:path_cache)
                        {cout<<"word: "<<d.convert(pp.first)<<" nodes: ";
                         for(auto kk:pp.second){cout<<kk->get_name()<<" ";}
                         cout<<endl;
                        }
                        cout<<"--------"<<endl;*/
                        
                        //if (skip_node == true) break;
                        j++;
                        /* ---------------- */
                    } //end of the loop through outbound_arc
                    //if (skip_node == true) continue;
                    i++;
                } //end of the loop through expanded_nodes
                expanded_nodes.clear();
                expanded_nodes = expanded_nodes_new;
                
                /*map<int, set<expanded_node*>> path_cache;
                map<string, expanded_node*> expanded_nodes;*/
                
                /*if ((beam_size == 1) && (t>0)) {
                    double best_loss = -1;
                    expanded_node* best_leaf = nullptr;
                    for (auto nn:expanded_nodes) {
                        auto e_node = nn.second;
                        double loss = e_node->get_in_arc()->get_loss();
                        if (loss<best_loss) {
                            best_loss = loss;
                            best_leaf = e_node;
                        }
                    }
                    expanded_nodes.clear();
                    expanded_nodes[best_leaf->get_name()]=best_leaf;
                    path_cache.clear();
                    path_cache[best_leaf->get_in_arc()->get_word()]={best_leaf};
                }*/
                
                //cout<<"expanded_nodes size: "<<expanded_nodes.size()<<endl;
                //cout << "expanded_nodes: "; for (auto nn:expanded_nodes){cout<<nn.first<<" ";}cout<<endl;
                
            } // end of the loop through sent by t
            
            /* ---------------- */
            //n00->dfs(0);
            vector<int> best_path = {};
            if (sent_len>0) {
                double best_loss = 9e+99;
                expanded_node* best_leaf = nullptr;
                for (auto nn:expanded_nodes) {
                    auto e_node = nn.second;
                    double loss = e_node->get_in_arc()->get_loss();
                    //cout<<"log loss: "<<loss<<endl;
                    if (loss<best_loss) {
                        best_loss = loss;
                        best_leaf = e_node;
                    }
                }
                best_path = best_leaf->get_lineage();
                
                string char_out = "";
                /* print the decoded sentence */
                for (unsigned i=1;i<best_path.size()-1;i++) {
                    word_pre_doc.push_back(best_path[i]);

                    if (i>1) {cout<<" ";out<<" ";char_out=char_out+" ";}
                    string w;
                    if (oov_candidates.count(i)) {
                        w=d.convert(best_path[i]);
                        if (w=="UNK") {

                            if (!include_charlm) {
                                //w=doc_str[k][i-1];
                                w=oov_candidates_str.at(i).back();
                            } else {
                                //cout<<"################"<< endl;
                                vector<string> char_outs = oov_candidates_str.at(i);
                                for (unsigned j=0;j<char_outs.size();++j) {
                                    char_outs[j] = char_out + char_outs[j];
                                }
                                // "selected" is a composed data structure here
                                cg.checkpoint();
                                auto selected = charlm.score_pick(cg, char_outs);
                                cg.revert();
                                cout << "<<selected index: "<<selected.best_idx<<">>";
                                w=(oov_candidates_str.at(i))[selected.best_idx];
                            }
                        }
                    } else {
                        w=doc_str[k][i-1];
                    }
                    cout<<w;
                    out<<w;
                    char_out=char_out+w;
                }
                cout<<endl;
                out<<endl;
                char_out=char_out+"\n";
            } // end of performing decoding for one sentence
            /* ---------------- */
            

        } // end of the loop through doc by k
        
        //Expression i_nerr = sum(errs);
        //return i_nerr;
        return 0;
    }

    /* build a rnnlm graph, which returns an Expression */
    unsigned decode_context(const Doc doc, 
        ComputationGraph& cg, 
        Dict* dptr, 
        // candidates for each OOV words 
        const DocStr doc_candidate, 
        // corpus in string instead of one hot numbers
        const DocStr doc_str, 
        ofstream& out, 
        unsigned beam_size, 

        bool include_charlm, 
        charlm_lstm& charlm) {

        // reset RNN builder for new graph
        builder.new_graph(cg);
        // define expression
        Expression i_R = parameter(cg, p_R);
        /* word bias */
        Expression i_bias = parameter(cg, p_bias);

        Expression i_x_t, i_y_t, i_r_t, i_err;

        //vector<Expression> errs;

        /* loop through all sentences in the doc */
        for (unsigned k = 0; k < doc.size(); k++) {
            // start a new sequence for each sentence
            builder.start_new_sequence();
            // for each sentence in this doc
            auto sent = doc[k];
            unsigned sent_len = sent.size();

            
            /* ---------------- */
            //cout << "new sequence!"<<endl;
            //cout<<"candidate size: "<<doc_candidate[k].size()<<endl;
            /*if (doc_candidate[k].empty()) {
                auto sent_str = doc_str[k];
                for (unsigned i=0;i<sent_str.size();i++) {
                    if (i>0) {cout<<" ";out<<" ";}
                    cout<<sent_str[i];
                    out<<sent_str[i];
                }
                cout<<endl;
                out<<endl;
                continue;
            }*/
            map<unsigned,set<int>> oov_candidates=get_oov_candidates(doc_candidate[k], dptr);
            map<unsigned,vector<string>> oov_candidates_str=get_oov_candidates_str(doc_candidate[k]);

            /* map the end point to the hole history to the probability of it and the vector representation of it */
            /* int: word on the incoming arc */
            map<int, set<expanded_node*>> path_cache;

            /* string: node name */
            map<string, expanded_node*> expanded_nodes;

            /* the first node in the lattice */
            expanded_node* n00 = new expanded_node("n00", nullptr);
            expanded_nodes["n00"]=n00;
            int null_word = -1;
            path_cache[null_word] = {n00};

            //cout<<"path_cache and expanded_nodes initialized"<<endl;

            /* initialize the outbound arc list */
            vector<set<int>> outbound_arc(sent_len, set<int>());
            for (unsigned t=0; t<sent_len; t++) {
                if (oov_candidates.count(t)) {
                    outbound_arc[t] = oov_candidates[t];
            } else {
                outbound_arc[t] = {sent[t]};
                }
            }
            //cout<<"outbound_arc initialized"<<endl;      	

            /* ---------------- */

            for (unsigned t = 0; t < sent_len; t++) {
                //cout<<"processing word "<<t<<endl;
                unsigned i = 0;
                map<string, expanded_node*> expanded_nodes_new;
                unsigned arc_num = outbound_arc[t].size();
                //cout<<"expanded_nodes_new created"<<endl;
                for (auto e_node:expanded_nodes) {
                    auto n_pre = e_node.second;
                    unsigned j = 0;
                    //bool skip_node = false;
                    unsigned outbound_arc_num = outbound_arc[t].size();
                    //cout<<"outbound_arc_num: "<<outbound_arc_num<<endl;
                    //cout<<"expanded_node found"<<endl;
                    for (auto arc_word:outbound_arc[t]) {
                        
                        RNNPointer state_cur = (RNNPointer)(-1);
                        double loss = 0.0;
                        vector<float> context_vec(p_R.dim().rows(), 0.0);
                        
                        //cout << "start expanding lattice"<<endl;
                        
                        if (t != 0) {
                            //cout<<"n_pre: "<<n_pre->get_name()<<endl;
                            i_x_t = lookup(cg, p_c, n_pre->get_in_arc()->get_word());
                            i_y_t = builder.add_input(n_pre->get_in_arc()->get_state(), i_x_t);
                            i_r_t = i_bias + i_R * i_y_t;
                            i_err = pickneglogsoftmax(i_r_t, arc_word);

                            /* ---------------- */
                            state_cur = builder.state();
                            context_vec = as_vector(cg.incremental_forward(i_y_t));
                            loss = as_scalar(cg.incremental_forward(i_err)) + n_pre->get_in_arc()->get_loss();
                            
                        }
                        //cout<<"state: "<<state_cur<<endl;
                        
                        /* create a new expanded node */
                        string node_name = "n"+to_string(t+1)+to_string(i*arc_num+j);
                        expanded_node* n_cur = new expanded_node(node_name, nullptr);
                        //cout << "build node ["<<node_name<<"] following: \"" << d.convert(arc_word) <<"\" and node ["<<n_pre->get_name()<<"]"<< endl;
                        
                        /* there isn't a path in cache that can represent the expanded node */
                        if (!path_cache.count(arc_word)) {
                            /* remove the previous expanded node */
                            if (t==0) {
                                path_cache.erase(null_word);
                            } else {
                                auto word_pre = n_pre->get_in_arc()->get_word();
                                if (path_cache.count(word_pre)) {
                                    path_cache.at(word_pre).erase(n_pre);
                                    if (path_cache.at(word_pre).size()==0) path_cache.erase(word_pre);
                                }
                            }
                            
                            /* create a new expanded arc */
                            string arc_name = "a"+to_string(t+1)+to_string(i*arc_num+j);
                            expanded_arc* e_arc = new expanded_arc(arc_name, arc_word, n_pre, n_cur);
                            e_arc->set_state(state_cur);
                            e_arc->set_loss(loss);
                            e_arc->set_context_vec(context_vec);
                            
                            if (!path_cache.count(arc_word)) path_cache[arc_word]={n_cur};
                            else path_cache[arc_word]={n_cur};
                            expanded_nodes_new[node_name]=n_cur;
                            
                        /* there is a path in cache that can represent the expanded node */
                        } else {
                            auto leaf_nodes = path_cache.at(arc_word);
                            bool to_merge = false;
                            for (auto leaf_node:leaf_nodes) {
                                if (!expanded_nodes.count(leaf_node->get_name())) {
                                    
                                    vector<float> context_vec_competing = leaf_node->get_in_arc()->get_context_vec();
                                    float contexts_dist = dist(context_vec_competing, context_vec);

                                    //cout<<"dist: "<<contexts_dist<<endl;
                                    if (contexts_dist<thres_dist) {
                                        //cout << "merge node [" << leaf_node->get_name() << "] and [" << node_name << "] following \"" <<d.convert(arc_word)<<"\" that emerges from node ["<< leaf_node->get_in_arc()->get_out_node()->get_name()<<"] and ["<<n_pre->get_name()<<"]"<<endl;
                                        double loss_competing = leaf_node->get_in_arc()->get_loss();
                                        //cout<<"loss at node ["<< leaf_node->get_name() <<"]: "<<loss_competing<<endl;
                                        //cout<<"loss at node ["<< node_name<<"]: "<<loss<<endl;
                                        
                                        /* replace the path with the expanded node */
                                        if (loss < loss_competing) {
                                            /* remove the competing expanded node */
                                            path_cache.at(arc_word).erase(leaf_node);
                                            expanded_nodes_new.erase(leaf_node->get_name());
                                            
                                            //cout << "remove node [" << leaf_node->get_name() << "]" << endl;
                                            if (leaf_node->get_in_arc()->get_out_node()->get_name() != n_pre->get_name()) {
                                                leaf_node->del_path();
                                            } else {
                                                leaf_node->del_by_depth_1();
                                            }
                                            
                                            /* remove the previous expanded node */
                                            if (t==0) {
                                                path_cache.erase(null_word);
                                            } else {
                                                auto word_pre = n_pre->get_in_arc()->get_word();
                                                if (path_cache.count(word_pre)) {
                                                    path_cache.at(word_pre).erase(n_pre);
                                                    if (path_cache.at(word_pre).size()==0) path_cache.erase(word_pre);
                                                }
                                            }
                                            
                                            /* create a new expanded arc */
                                            string arc_name = "a"+to_string(t+1)+to_string(i*arc_num+j);
                                            expanded_arc* e_arc = new expanded_arc(arc_name, arc_word, n_pre, n_cur);
                                            e_arc->set_state(state_cur);
                                            e_arc->set_loss(loss);
                                            e_arc->set_context_vec(context_vec);
                                            
                                            if (!path_cache.count(arc_word)) path_cache[arc_word]={n_cur};
                                            else path_cache.at(arc_word).insert(n_cur);
                                            expanded_nodes_new[node_name]=n_cur;

                                        /* keep the existing path */
                                        } else {
                                            //*n_cur=*leaf_node;
                                            
                                            auto out_arcs = n_pre->get_out_arcs();
                                            unsigned out_arcs_num = (*out_arcs).size();
                                            if ((out_arcs_num == 0) && (j == outbound_arc_num-1)) {
                                                //cout << "remove node [" << n_pre->get_name() << "]" << endl;
                                                
                                                /* remove the previous expanded node */
                                                if (t==0) {
                                                    path_cache.erase(null_word);
                                                } else {
                                                    auto word_pre = n_pre->get_in_arc()->get_word();
                                                    if (path_cache.count(word_pre)) {
                                                        path_cache.at(word_pre).erase(n_pre);
                                                        if (path_cache.at(word_pre).size()==0) path_cache.erase(word_pre);
                                                    }
                                                }
                                                
                                                n_pre->del_path();                                                 
                                                //skip_node = true;
                                            }
                                        }
                                        to_merge = true;
                                        break;
                                    } else {
                                        //cout << "not merge node [" << leaf_node->get_name() << "] and [" << node_name << "] following \"" <<d.convert(arc_word)<<"\" that emerges from node ["<< leaf_node->get_in_arc()->get_out_node()->get_name()<<"] and ["<<n_pre->get_name()<<"]"<<endl;
                                    }
                                }
                            }
                            /* there isn't a path in cache that can represent the expanded node */
                            if (!to_merge) {
                                //cout << "not merge [" << node_name << "] with any node following \"" <<d.convert(arc_word)<<"\""<< endl;
                                /* remove the previous expanded node */
                                if (t==0) {
                                    path_cache.erase(null_word);
                                } else {
                                    auto word_pre = n_pre->get_in_arc()->get_word();
                                    if (path_cache.count(word_pre)) {
                                        path_cache.at(word_pre).erase(n_pre);
                                        if (path_cache.at(word_pre).size()==0) path_cache.erase(word_pre);
                                    }
                                }
                                
                                /* create a new expanded arc */
                                string arc_name = "a"+to_string(t+1)+to_string(i*arc_num+j);
                                expanded_arc* e_arc = new expanded_arc(arc_name, arc_word, n_pre, n_cur);
                                e_arc->set_state(state_cur);
                                e_arc->set_loss(loss);
                                e_arc->set_context_vec(context_vec);
                                
                                if (!path_cache.count(arc_word)) path_cache[arc_word]={n_cur};
                                else path_cache.at(arc_word).insert(n_cur);
                                expanded_nodes_new[node_name]=n_cur;
                            }
                        }
                        /*n00->dfs();
                        cout<<"--------"<<endl;
                        cout << "path_cache: "<<endl; for (auto pp:path_cache)
                        {cout<<"word: "<<d.convert(pp.first)<<" nodes: ";
                         for(auto kk:pp.second){cout<<kk->get_name()<<" ";}
                         cout<<endl;
                        }
                        cout<<"--------"<<endl;*/
                        
                        //if (skip_node == true) break;
                        j++;
                        /* ---------------- */
                    } //end of the loop through outbound_arc
                    //if (skip_node == true) continue;
                    i++;
                } //end of the loop through expanded_nodes
                expanded_nodes.clear();
                expanded_nodes = expanded_nodes_new;
                
                /*map<int, set<expanded_node*>> path_cache;
                map<string, expanded_node*> expanded_nodes;*/
                
                /*if ((beam_size == 1) && (t>0)) {
                    double best_loss = -1;
                    expanded_node* best_leaf = nullptr;
                    for (auto nn:expanded_nodes) {
                        auto e_node = nn.second;
                        double loss = e_node->get_in_arc()->get_loss();
                        if (loss<best_loss) {
                            best_loss = loss;
                            best_leaf = e_node;
                        }
                    }
                    expanded_nodes.clear();
                    expanded_nodes[best_leaf->get_name()]=best_leaf;
                    path_cache.clear();
                    path_cache[best_leaf->get_in_arc()->get_word()]={best_leaf};
                }*/
                
                //cout<<"expanded_nodes size: "<<expanded_nodes.size()<<endl;
                //cout << "expanded_nodes: "; for (auto nn:expanded_nodes){cout<<nn.first<<" ";}cout<<endl;
                
			} // end of the loop through sent by t
            
            /* ---------------- */
            //n00->dfs(0);
            vector<int> best_path = {};
            if (sent_len>0) {
                double best_loss = 9e+99;
                expanded_node* best_leaf = nullptr;
                for (auto nn:expanded_nodes) {
                    auto e_node = nn.second;
                    double loss = e_node->get_in_arc()->get_loss();
                    //cout<<"log loss: "<<loss<<endl;
                    if (loss<best_loss) {
                        best_loss = loss;
                        best_leaf = e_node;
                    }
                }
                best_path = best_leaf->get_lineage();
                
                string char_out = "";
                /* print the decoded sentence */
                for (unsigned i=1;i<best_path.size()-1;i++) {
                    if (i>1) {cout<<" ";out<<" ";char_out=char_out+" ";}
                    string w;
                    if (oov_candidates.count(i)) {
                        w=d.convert(best_path[i]);
                        if (w=="UNK") {

                            if (!include_charlm) {
                                //w=doc_str[k][i-1];
                                w=oov_candidates_str.at(i).back();
                            } else {
                                //cout<<"################"<< endl;
                                vector<string> char_outs = oov_candidates_str.at(i);
                                for (unsigned j=0;j<char_outs.size();++j) {
                                    char_outs[j] = char_out + char_outs[j];
                                }
                                // "selected" is a composed data structure here
                                cg.checkpoint();
 								auto selected = charlm.score_pick(cg, char_outs);
                                cg.revert();
                                cout << "<<selected index: "<<selected.best_idx<<">>";
                                w=(oov_candidates_str.at(i))[selected.best_idx];
                            }
                        }
                    } else {
                        w=doc_str[k][i-1];
                    }
                    cout<<w;
                    out<<w;
                    char_out=char_out+w;
                }
                cout<<endl;
                out<<endl;
                char_out=char_out+"\n";
            } // end of performing decoding for one sentence
            /* ---------------- */
            

        } // end of the loop through doc by k
        
        //Expression i_nerr = sum(errs);
        //return i_nerr;
        return 0;
    }

};




int main(int argc, char** argv) {
	// initialize dynet parameters
  dynet::initialize(argc, argv);

  unsigned num_layer = atoi(argv[1]);
  unsigned input_dim = atoi(argv[2]);
  unsigned hidden_dim = atoi(argv[3]);
  
  unsigned beam_size = atoi(argv[4]);
  
  string model_file = argv[5];
  string dict_file = argv[6];
  char* tra_file = (char *)argv[7];
  char* candidate_file = (char *)argv[8];
  char* res_file = (char *)argv[9];

  // beam search or context based search
  string decoder = argv[10];

  // char model parameters
  bool include_charlm = ((string)argv[11] == "true") || ((string)argv[11] == "True") ? true : false;
  string charmodel_file;
  string chardict_file;
  unsigned charlm_num_layer;
  unsigned charlm_input_dim;
  unsigned charlm_hidden_dim;
  if (include_charlm) {
      charmodel_file = argv[12];
      chardict_file = argv[13];
      charlm_num_layer = atoi(argv[14]);
      charlm_input_dim = atoi(argv[15]);
      charlm_hidden_dim = atoi(argv[16]);
  }


  // file stream to output translation results to
  ofstream out(res_file);
  
  // do the same as in test
  kSOS = d.convert("<s>");
  kEOS = d.convert("</s>");
  
  load_dict(dict_file, d);
  cout << "dict loaded from: "<<dict_file<<endl;
  d.freeze();
  unsigned vocab_size = d.size();
    
	// load the charlm 
	charlm_lstm charlm;
	Model charmodel;
    cout<<"load charlm? "<<include_charlm<<endl;
    if (include_charlm) {
    	if (boost::filesystem::exists(charmodel_file)) {
            cout << "Load char-level model from: " << charmodel_file << endl;
            charlm.initialize_with_dict(charmodel, chardict_file, charlm_num_layer, charlm_input_dim, charlm_hidden_dim);
            load_model(charmodel_file, charmodel);
        } else {
            cout << "Model not existed at: " << model_file << endl;
            throw std::invalid_argument("Model not existed");
        }
    }

	// load the dclm
   	Model amodel;

    RNNLanguageModel<LSTMBuilder> alm(
        amodel, 
        vocab_size, 
        num_layer,
        input_dim,
        hidden_dim);

    load_model(model_file, amodel);
    cout << "dclm loaded from: "<<model_file<<endl;

    Corpus tra_corpus = readData(tra_file, &d, false);
    //cout << "data read from: "<<tra_file<<endl;
    
    CorpusStr candidate_corpus = readDataStr(candidate_file);
    CorpusStr str_corpus = readDataStr(tra_file);
    
    unsigned doc_ctr = 0;
    for (auto& doc : tra_corpus) {
        ComputationGraph cg;
        
        //cout << "new doc!"<<endl;

        // cout<<"doc_ctr: "<<doc_ctr<<endl;
        // cout<<"size of candidate_corpus: "<<candidate_corpus.size()<<endl;
        // cout<<"size of str_corpus: "<<str_corpus.size()<<endl;

        DocStr doc_candidate = candidate_corpus[doc_ctr];
        DocStr doc_str = str_corpus[doc_ctr];

        //cout<<"decoder: "<<decoder<<endl;

        // "beam" or "context"
        if (decoder == "beam") {
            alm.decode_beam(
                doc,
                cg,
                &d,
                doc_candidate,
                doc_str,
                out,
                beam_size,
                include_charlm,
                charlm);
        } else if (decoder == "context") {
            alm.decode_context(
                doc, 
                cg, 
                &d, 
                doc_candidate, 
                doc_str, 
                out, 
                beam_size, 
                include_charlm, 
                charlm);
        } else if (decoder == "embed") {
            alm.decode_embed(
                doc, 
                cg, 
                &d, 
                doc_candidate, 
                doc_str, 
                out, 
                beam_size, 
                include_charlm, 
                charlm);
        }
        cout<<"="<<endl;
        out<<"="<<endl;
        doc_ctr++;
    }
    out.close();
    return 0;
}
