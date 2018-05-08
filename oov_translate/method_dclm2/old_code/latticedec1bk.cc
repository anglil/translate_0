#include <dynet/nodes.h>
#include <dynet/dynet.h>
#include <dynet/expr.h> 
    
#include "util.hpp"
#include "latticedec.h" 
#include "charlm.h"
#include "latticedec_algo.h"
    
using namespace dynet;
using namespace std;

dynet::Dict d;
int kEOS, kSOS;

float thres_dist = 0.5;


/* adclm */
template <class Builder> class DocumentAttentionalModel {
private:
    LookupParameter p_c;
    Parameter p_R;
    Parameter p_Q;
    Parameter p_P;
    Parameter p_bias;
    Parameter p_Wa;
    Parameter p_Ua;
    Parameter p_va;

    Builder builder;
    unsigned context_dim;

    Expression src;
    Expression i_R;
    Expression i_Q;
    Expression i_P;
    Expression i_bias;
    Expression i_Wa;
    Expression i_Ua;
    Expression i_va;
    Expression i_uax;
    Expression i_empty;

    std::vector<float> zeros;
    std::vector<Expression> context;

public:
	DocumentAttentionalModel();

	/* constructor */
	DocumentAttentionalModel(
		Model& model, 
		unsigned vocab_size,
		unsigned num_layer,
		unsigned input_dim,
		unsigned hidden_dim,
        unsigned align_dim
	) : builder(num_layer, input_dim+num_layer*hidden_dim, hidden_dim, model), context_dim(num_layer*hidden_dim) {
		p_c = model.add_lookup_parameters(vocab_size, {input_dim});
		p_R = model.add_parameters({vocab_size, hidden_dim});
		p_Q = model.add_parameters({hidden_dim, context_dim});
		p_P = model.add_parameters({hidden_dim, input_dim});
		p_bias = model.add_parameters({vocab_size});
		p_Wa = model.add_parameters({align_dim, num_layer*hidden_dim});
		p_Ua = model.add_parameters({align_dim, context_dim});
		p_va = model.add_parameters({align_dim});
	}


    
	/* build an adclm graph, which returns an Expression */
	unsigned decode_context(const Doc doc, 
        ComputationGraph& cg, Dict* dptr, 
        const DocStr doc_candidate, 
        const DocStr doc_str, 
        ofstream& out, 
        unsigned beam_size,

        bool include_charlm, 
        charlm_lstm& charlm) {

		// reset RNN builder for new graph
		builder.new_graph(cg);
        context.clear();
        
		i_R = parameter(cg, p_R);
		i_Q = parameter(cg, p_Q);
		i_P = parameter(cg, p_P);
		i_bias = parameter(cg, p_bias);
		i_Wa = parameter(cg, p_Wa);
		i_Ua = parameter(cg, p_Ua);
		i_va = parameter(cg, p_va);

		zeros.resize(context_dim, 0);
		i_empty = dynet::expr::input(cg, {context_dim}, &zeros);
        
        /* first sentence in the document */
        bool first = true;

        expanded_node* best_leaf=nullptr;
		/* iterate through all sentences in the doc */
		for (unsigned k = 0; k < doc.size(); k++) {
            

			// start a new sequence for each sentence
            //cout<<"before final_h 1"<<endl;
            
			if (!first && (best_leaf!=nullptr)) {
                //context.push_back(concatenate(builder.final_h()));
                auto tt = best_leaf->get_in_arc()->get_state();
                //cout << tt<<endl;
                auto ss = builder.get_h(tt);
                //cout<<ss.size()<<endl;
                context.push_back(concatenate(ss));
            }
            //cout<<"after final_h 1"<<endl;
    
            builder.start_new_sequence();
            
            if (context.size() > 1) {
                src = concatenate_cols(context);
                i_uax = i_Ua * src;
            }
            

            /* ---------------- */
			//cout << "new sequence!"<<endl;
            //cout<<"candidate size: "<<doc_candidate[k].size()<<endl;
            // for each sentence in this doc
            auto sent = doc[k];
            unsigned sent_len = sent.size();
            
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
                    for (auto arc_word:outbound_arc[t]) {
                        
                        RNNPointer state_cur = (RNNPointer)(-1);
                        double loss = 0.0;
                        vector<float> context_vec(p_R.dim().rows(), 0.0);
                        
                        //cout << "start expanding lattice"<<endl;
                        
                        if (t != 0) {
                            //cout<<"n_pre: "<<n_pre->get_name()<<endl;
                            
                            // get i_x_t
                            Expression i_x_t = lookup(cg, p_c, n_pre->get_in_arc()->get_word());
                            
                            // get i_c_t
                            Expression i_c_t;
                            if (context.size() > 1) {
                                Expression i_wah_rep;
                                auto tt = n_pre->get_in_arc()->get_state();
                                if ((t > 0) && (tt != -1)) {
                                    //cout<<"before final_h 2"<<endl;
                                    //auto ss = builder.final_h();
                                    //cout<<tt<<endl;
                                    auto ss = builder.get_h(tt);
                                    //cout<<ss.size()<<endl;
                                    auto i_h_tml = concatenate(ss);
                                    //cout<<"after final_h 2"<<endl;
                                    Expression i_wah = i_Wa * i_h_tml;
                                    i_wah_rep = concatenate_cols(vector<Expression>(context.size(), i_wah));
                                }

                                Expression i_e_t;
                                if ((t > 0) && (tt != -1)) i_e_t = transpose(tanh(i_wah_rep+i_uax))*i_va;
                                else i_e_t = transpose(tanh(i_uax))*i_va;

                                Expression i_alpha_t = softmax(i_e_t);
                                i_c_t = src * i_alpha_t;
                            } else if (context.size() == 1) {
                                i_c_t = context.back();
                            } else {
                                i_c_t = i_empty;
                            }

                            // concatenate i_x_t and i_c_t to form input
                            Expression input = concatenate(vector<Expression>({i_x_t, i_c_t}));
                            Expression i_y_t = builder.add_input(n_pre->get_in_arc()->get_state(), input);
                            Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
                            Expression i_r_t = affine_transform({i_bias, i_R, i_tildet_t});


                            Expression i_err = pickneglogsoftmax(i_r_t, arc_word);

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
                                            if ((out_arcs_num == 0) && (j == arc_num-1)) {
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
                    double best_prob = -1;
                    for (auto nn:expanded_nodes) {
                        auto e_node = nn.second;
                        double prob = e_node->get_in_arc()->get_prob();
                        if (prob>best_prob) {
                            best_prob = prob;
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
                
			} // end of the loop through sent
            
            /* ---------------- */
            //n00->dfs(0);
            vector<int> best_path = {};
            if (sent_len>0) {
                double best_loss = 9e+99;
                //expanded_node* best_leaf = nullptr;
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
            }
            /* ---------------- */
            
			// update context vector at the end of a sentence
			first = false;
		
        } // end of the loop through doc
        
		//Expression i_nerr = sum(errs);
		//return i_nerr;
        return 0;
	}


};




int main (int argc, char** argv) {
    dynet::initialize(argc, argv);

    unsigned num_layer = atoi(argv[1]);
    unsigned input_dim = atoi(argv[2]);
    unsigned hidden_dim = atoi(argv[3]);
    unsigned align_dim = atoi(argv[4]);
    
    unsigned beam_size = atoi(argv[5]);
    
    string model_file = argv[6];
    string dict_file = argv[7];
    char* tra_file = (char *)argv[8];
    char* candidate_file = (char *)argv[9];
    char* res_file = (char *)argv[10];

    string decoder = argv[11];

    // char model parameters
    bool include_charlm = ((string)argv[12] == "true") || ((string)argv[12] == "True") ? true : false;
    string charmodel_file;
    string chardict_file;
    unsigned charlm_num_layer; 
    unsigned charlm_input_dim; 
    unsigned charlm_hidden_dim;
    if (include_charlm) {
        charmodel_file = argv[13];
        chardict_file = argv[14];
        charlm_num_layer = atoi(argv[15]);
        charlm_input_dim = atoi(argv[16]);
        charlm_hidden_dim = atoi(argv[17]);
    }
    
    ofstream out(res_file);
    
    /* do the same as in test */
    kSOS = d.convert("<s>");
    kEOS = d.convert("</s>");
    
    load_dict(dict_file, d);
    cout << "dict loaded from: "<<dict_file<<endl;
    d.freeze();
    unsigned vocab_size = d.size();
    
    // load the charlm 
    charlm_lstm charlm;
    Model charmodel;
    if (include_charlm) {
        if (boost::filesystem::exists(charmodel_file)) {
            cout << "Load char-level model from: " << charmodel_file << endl;
            charlm.initialize_with_dict(charmodel, chardict_file, charlm_num_layer, charlm_input_dim, charlm_hidden_dim);
            load_model(charmodel_file, charmodel);
        } else {
            cout << "Model not existed at: " << model_file << endl;
            throw std::invalid_argument("Model not exists!");
        }
    }

    // load the dclm
    Model amodel;

    DocumentAttentionalModel<LSTMBuilder> alm(
        amodel, 
        vocab_size, 
        num_layer,
        input_dim,
        hidden_dim,
        align_dim);

    load_model(model_file, amodel);
    cout << "model loaded from: "<<model_file<<endl;

    Corpus tra_corpus = readData(tra_file, &d, false);
    cout<<"data read from: "<<tra_file<<endl;
    
    CorpusStr candidate_corpus = readDataStr(candidate_file);
    CorpusStr str_corpus = readDataStr(tra_file);
    
    unsigned doc_ctr = 0;
    for (auto& doc : tra_corpus) {
        ComputationGraph cg;
        //cout << "new doc!"<<endl;

        DocStr doc_candidate = candidate_corpus[doc_ctr];
        DocStr doc_str = str_corpus[doc_ctr];

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
