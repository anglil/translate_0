#include <boost/algorithm/string/join.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <dynet/dynet.h>
#include <dynet/rnn.h>

using namespace std;
using namespace dynet;

class expanded_arc;

class expanded_node {
  private:
    string name;
    expanded_arc* in_arc;
    /* string: name of the arc */
    map<string, expanded_arc*> out_arcs;
  public:
    expanded_node();    

    expanded_node(string name, expanded_arc* in_arc) {
      this->name = name;
      this->in_arc = in_arc;
    }   
    void dfs(unsigned depth);
    void dfs();
    void del_path();
    void del_by_depth_1();
    vector<int> get_lineage();
    void set_in_arc(expanded_arc* in_arc) {
      this->in_arc = in_arc;
    }   
    void add_out_arc(string out_arc_id, expanded_arc* out_arc) {
      out_arcs[out_arc_id] = out_arc;
    }   
    /*void set_node(expanded_node* n2) {
 *       this->name = n2->name;
 *             this->in_arc = n2->in_arc;
 *                   this->out_arcs = n2->out_arcs;
 *                       }*/
    expanded_arc* get_in_arc() {
      return in_arc;
    }   
    map<string, expanded_arc*>* get_out_arcs() {
      return &out_arcs;
    } 
    string get_name() {
      return name;
    }
};

class expanded_arc {
  private:
    string name;
    int word;
    RNNPointer state;
    double loss;
    Expression i_e_t;
    vector<float> context_vec;
    expanded_node* out_node;
    expanded_node* in_node;
  public:
    expanded_arc();
    expanded_arc(string name, int word, expanded_node* out_node, expanded_node* in_node) {
      this->name = name;
      this->word = word;
      this->out_node = out_node;
      this->in_node = in_node;
      out_node->add_out_arc(this->name, this);
      in_node->set_in_arc(this);
    }
    void del_path();
    void set_out_node(expanded_node* out_node) {
      this->out_node = out_node;
      return;
    }
    void set_in_node(expanded_node* in_node) {
      this->in_node = in_node;
      return;
    }
    void set_embed(Expression i_e_t) {
      this->i_e_t = i_e_t;
      return;
    }
    void set_state(RNNPointer state) {
      this->state = state;
      return;
    }
    void set_loss(double loss) {
      this->loss = loss;
      return;
    }
    void set_context_vec(vector<float> hs) {
      this->context_vec.clear();
      for (auto elem:hs){
        this->context_vec.push_back(elem);
      }
      return;
    }
    expanded_node* get_out_node() {
      return out_node;
    }
    expanded_node* get_in_node() {
      return in_node;
    }
    string get_name() {
      return name;
    }
    int get_word() {
      return word;
    }
    Expression get_embed() {
      return i_e_t;
    }
    RNNPointer get_state() {
      return state;
    }
    double get_loss() {
      return loss;
    }
    vector<float> get_context_vec() {
      return context_vec;
    }
};

void expanded_node::dfs(unsigned depth) {
  if (this == nullptr) {
    return;
  }
  cout << this->get_name();
  auto out_arcs = this->get_out_arcs();
  unsigned ctr = (*out_arcs).size();
  unsigned i = 0;
  for (auto out_arc_pair : *out_arcs) {
    expanded_node* n = out_arc_pair.second->get_in_node();
    cout << "--" << out_arc_pair.second->get_name() << "--";
    n->dfs(depth+1);

    if ((ctr > 1) && (i!=ctr-1)) {
      cout << endl;
      cout << "   ";
      unsigned tmp = depth;
      while (tmp > 0) {
        cout << "          ";
        tmp--;
      }
    }
    i++;
  }
  return;
}

void expanded_node::dfs() {
    expanded_node::dfs(0);
    cout<<endl;
}

vector<int> expanded_node::get_lineage() {
    auto in_arc = this->get_in_arc();
    vector<int> lineage = {};
    while (in_arc != nullptr) {
        lineage.insert(lineage.begin(), in_arc->get_word());
        auto out_node = in_arc->get_out_node();
        in_arc = out_node->get_in_arc();
    }   
    return lineage;
}

void expanded_node::del_path() {
  auto out_arcs = this->get_out_arcs();
  unsigned ctr = (*out_arcs).size();
  if (ctr!=0) {
      //cout<<"can't remove node ["<<this->get_name()<<"]"<<endl;
      return;
  }

  expanded_arc* in_arc = this->get_in_arc();
  delete this;
  expanded_node* leaf = in_arc->get_out_node();
  out_arcs = leaf->get_out_arcs();
  out_arcs->erase(out_arcs->find(in_arc->get_name()));
  delete in_arc;

  ctr = (*out_arcs).size();

  while (ctr == 0) {
    in_arc = leaf->get_in_arc();
    delete leaf;
    leaf = in_arc->get_out_node();
    out_arcs = leaf->get_out_arcs();
    out_arcs->erase(out_arcs->find(in_arc->get_name()));
    delete in_arc;
    ctr = (*out_arcs).size();
  }
  return;
}


void expanded_node::del_by_depth_1() {
    auto out_arcs = this->get_out_arcs();
    unsigned ctr = (*out_arcs).size();
    if (ctr!=0) {
        //cout<<"can't remove node ["<<this->get_name()<<"]"<<endl;
        return;
    }
    
    expanded_arc* in_arc = this->get_in_arc();
    delete this;
    delete in_arc;
}

void expanded_arc::del_path() {
  expanded_node* leaf = this->get_out_node();
  auto out_arcs = leaf->get_out_arcs();
  out_arcs->erase(out_arcs->find(this->get_name()));
  delete this;
  expanded_arc* in_arc;

  unsigned ctr = (*out_arcs).size();

  while (ctr == 0) {
    in_arc = leaf->get_in_arc();
    delete leaf;
    leaf = in_arc->get_out_node();
    out_arcs = leaf->get_out_arcs();
    out_arcs->erase(out_arcs->find(in_arc->get_name()));
    delete in_arc;
    ctr = (*out_arcs).size();
  }
  return;
}

