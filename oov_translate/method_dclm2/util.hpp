#ifndef UTIL_HPP
#define UTIL_HPP

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/tensor.h"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <thread>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace dynet;

// ********************************************************
// Predefined information, used for the entire project
// ********************************************************
// Redefined types
typedef vector<int> Sent;
typedef vector<Sent> Doc;
typedef vector<Doc> Corpus;

typedef vector<string> SentStr;
typedef vector<SentStr> DocStr;
typedef vector<DocStr> CorpusStr;

typedef vector<string> DocChar;
typedef vector<DocChar> CorpusChar;

// *******************************************************
// split a string into a vector
// ******************************************************* 
vector<string> split(string str, char delimiter);

// *******************************************************
// get_eng_vocab
// *******************************************************  
set<string> get_eng_vocab(string eng_vocab_file);

// *******************************************************
// get_ug_dict
// ******************************************************* 
map<string,map<string,float>> get_ug_dict(string oov_candidates_file);

// *******************************************************
// get_aligned_oov
// ******************************************************* 
map<string,set<string>> get_aligned_oov(string oov_aligned_file);

// *******************************************************
// get_oov_candidates
// ******************************************************* 
map<unsigned,set<int>> get_oov_candidates(SentStr oc_sent, Dict* dptr);

// *******************************************************
// get_oov_candidates_str
// ******************************************************* 
map<unsigned,vector<string>> get_oov_candidates_str(SentStr oc_sent);
    
// *******************************************************
// compute normalized distance between two vectors
// *******************************************************   
float dist(const vector<float> &a, const vector<float> &b);

// *******************************************************
// load model from a archive file
// *******************************************************
int load_model(string fname, Model& model);

// *******************************************************
// save model from a archive file
// *******************************************************
int save_model(string fname, Model& model);

// *******************************************************
// save dict from a archive file
// *******************************************************
int save_dict(string fname, dynet::Dict d);

// *******************************************************
// load dict from a archive file
// *******************************************************
int load_dict(string fname, dynet::Dict& d);

// *******************************************************
// read sentences and convect tokens to indices
// *******************************************************
Sent MyReadSentence(const std::string& line, 
		    Dict* sd, 
		    bool update);

// *******************************************************
// read sentences and convect tokens to indices
// *******************************************************
Sent MyReadSentenceChar(const std::string& line, 
		    Dict* sd, 
		    bool update);

// *****************************************************
// 
// *****************************************************
Doc makeDoc();

// *****************************************************
// 
// *****************************************************
DocStr makeDocStr();

// *****************************************************
// 
// *****************************************************
DocChar makeDocChar();

// *****************************************************
// read training and dev data
// *****************************************************
CorpusChar readDataChar(char* filename);


// *****************************************************
// read training and dev data
// *****************************************************
Corpus readDataChar(char* filename,
	dynet::Dict* dptr,
	bool b_update = true);

// *****************************************************
// read training and dev data
// *****************************************************
CorpusStr readDataStr(char* filename);

// *****************************************************
// read training and dev data
// *****************************************************
Corpus readData(char* filename, 
		dynet::Dict* dptr,
		bool b_update = true);


// ******************************************************
// Convert 1-D tensor to vector<float>
// so we can create an expression for it
// ******************************************************
vector<float> convertT2V(const Tensor& t);

// ******************************************************
// Check the directory, if doesn't exist, create one
// ******************************************************
int check_dir(string path);

// ******************************************************
// Segment a long document into several short ones
// ******************************************************
Corpus segment_doc(Corpus doc, unsigned thresh);

#endif
