#include <dynet/nodes.h>
#include <dynet/dynet.h>
#include <dynet/expr.h>

#include "util.hpp"
#include "latticedec.h"
#include "charlm.h"

using namespace dynet;
using namespace std;

class decode_context {
  private:
    map<int, set<expanded_node*>> path_cache;
    map<string, expanded_node*> expanded_nodes;
    vector<set<int>> outbound_arc;
    int null_word;
    unsigned t;
    unsigned arc_num;

  public: 
    unsigned init(map<unsigned, set<int>> oov_candidates, vector<int> sent) {
      expanded_node* n00 = new expanded_node("n00", nullptr);
      expanded_nodes["n00"]=n00;
      null_word = -1;
      path_cache[null_word] = {n00};

      vector<set<int>> outbound_arc(sent_len, set<int>());
      for (unsigned t=0; t<sent_len; t++) {
        if (oov_candidates.count(t))
          outbound_arc[t] = oov_candidates[t];
        else
          outbound_arc[t] = {sent[t]};
      }
    }

    unsigned run(map<string, expanded_node*> &expanded_nodes_new, ) {
      string node_name = "n"+to_string(t+1)+to_string(i*arc_num+j);
      expanded_node* n_cur = new expanded_node(node_name, nullptr);
      
      // no path in cache represents the expanded node
      if (!path_cache.count(arc_word)) {
        // remove the previous expanded node
        if (t==0) {
          path_cache.erase(null_word);
        } else {
          auto word_pre = n_pre->get_in_arc()->get_word();
          if (path_cache.count(word_pre)) {
            path_cache.at(word_pre).erase(n_pre);
            if (path_cache.at(word_pre).size()==0)
              path_cache.erase(word_pre);
          }
        }

        // create a new expanded arc
        string arc_name = "a"+to_string(t+1)+to_string(i*arc_num+j);
        expanded_arc* e_arc = new expanded_arc(arc_name, arc_word, n_pre, n_cur);
        e_arc->set_state(state_cur);
        e_arc->set_loss(loss);
        e_arc->set_context_vec(context_vec);

        if (!path_cache.count(arc_word)) path_cache[arc_word]={n_cur};
        else path_cache[arc_word]={n_cur};
        expanded_nodes_new[node_name]=n_cur;
      // a path in cache can represent the expanded node
      } else {
        auto leaf_nodes = path_cache.at(arc_word);
        bool to_merge = false;
        for (auto leaf_node:leaf_nodes) {
          if (!expanded_nodes.count(leaf_node->get_name())) {
            vector<float> context_vec_competing = leaf_node->get_in_arc()->get_context_vec();
            float contexts_dist = dist(context_vec_competing, context_vec);

            if (contexts_dist<thres_dist) {
              double loss_competing = leaf_node->get_in_arc()->get_loss();
            
              // replace the path with the expanded node
              if (loss < loss_competing) {
                // remove the competing expanded node
                path_cache.at(arc_word).erase(leaf_node);
                expanded_nodes_new.erase(leaf_node->get_name());
                if (leaf_node->get_in_arc()->get_out_node()->get_name() != n_pre->get_name()) 
                  leaf_node->del_path();
                else
                  leaf_node->del_by_depth_1();

                // remove the previous expanded node
                if (t==0)
                  path_cache.erase(null_word);
                else {
                  auto word_pre = n_pre->get_in_arc()->get_word();
                  if (path_cache.count(word_pre)) {
                    path_cache.at(word_pre).erase(n_pre);
                    if (path_cache.at(word_pre).size()==0) path_cache.erase(word_pre);
                  }
                }

                // create a new expanded arc
                string arc_name = "a"+to_string(t+1)+to_string(i*arc_num+j);
                expanded_arc* e_arc = new expanded_arc(arc_name, arc_word, n_pre, n_cur);
                e_arc->set_state(state_cur);
                e_arc->set_loss(loss);
                e_arc->set_context_vec(context_vec);

                if (!path_cache.count(arc_word)) path_cache[arc_word]={n_cur};
                else path_cache.at(arc_word).insert(n_cur);
                expanded_nodes_new[node_name]=n_cur;

              // keep the existing path
              } else {
                auto out_arcs = n_pre->get_out_arcs();
                unsigned out_arcs_num = (*out_arcs).size();
                if ((out_arcs_num == 0) && (j == outbound_arc_num-1)) {
                  // remove the previous expanded node
                  if (t==0)
                    path_cache.erase(null_word);
                  else {
                    auto word_pre = n_pre->get_in_arc()->get_word();
                    if (path_cache.count(word_pre)) {
                      path_cache.at(word_pre).erase(n_pre);
                      if (path_cache.at(word_pre).size()==0) path_cache.erase(word_pre);
                    }
                  }

                  n_pre->del_path();
                }
              }
              to_merge = true;
              break;
            }
          }
        }
        if (!to_merge) {
          // remove the previous expanded node
          if (t==0)
            path_cache.erase(null_word);
          else {
            auto word_pre = n_pre->get_in_arc()->get_word();
            if (path_cache.count(word_pre)) {
              path_cache.at(word_pre).erase(n_pre);
              if (path_cache.at(word_pre).size()==0) path_cache.erase(word_pre);
            }
          }

          // create a new expanded arc
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
    }// end of run()
}
