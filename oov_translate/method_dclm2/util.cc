#include "util.hpp"

// *******************************************************
// split a string into a vector
// ******************************************************* 
vector<string> split(string str, char delimiter) {
  vector<string> internal;
  stringstream ss(str); // Turn the string into a stream.
  string tok;
  
  while(getline(ss, tok, delimiter)) {
    internal.push_back(tok);
  }
  return internal;
}
    
    
// *******************************************************
// get_eng_vocab
// *******************************************************  
set<string> get_eng_vocab(string eng_vocab_file) {
  set<string> function_words = {"is", "was", "are", "were", "be", "the", "an", "a", "and", "or"};
  set<string> eng_vocab = {};
  ifstream in(eng_vocab_file.c_str());
  string w;
  while(getline(in, w)) {
    boost::trim(w);
    boost::algorithm::to_lower(w);
    if (!function_words.count(w)) eng_vocab.insert(w);
  }
  return eng_vocab;
}
    

// *******************************************************
// get_ug_dict
// ******************************************************* 
map<string,map<string,float>> get_ug_dict(string oov_candidates_file) {
  map<string,map<string,float>> ug_dict = {};
  ifstream in(oov_candidates_file.c_str());
  string line;
  while(getline(in, line)) {
    vector<string> l;
    boost::split(l, line, boost::is_any_of("\t"));
    if ((l.size() > 1) && (l[1] != "[NOHYPS]")) {
      string ug_word = l[0];
      boost::algorithm::to_lower(ug_word);
      map<string,float> en_hyp = {};
      
      vector<string> en_list;
      if (l.size() == 3) 
        boost::split(en_list, l[2], boost::is_any_of(";"));
      else if (l.size() == 2)
        boost::split(en_list, l[1], boost::is_any_of(";"));
      else continue;
      
      en_list.pop_back();
      for (string word_score_pair:en_list) {
        vector<string> word_score_list;
        boost::split(word_score_list, word_score_pair, boost::is_any_of(","));
        if (word_score_list.size() == 2) {
          string en_word = word_score_list[0];
          boost::algorithm::to_lower(en_word);
          float score = stof(word_score_list[1]);
          vector<string> tmp;
          boost::split(tmp, en_word, boost::is_any_of(" "));
          if (tmp.size() == 1) en_hyp[en_word] = score;
        }
      }
      if (!en_hyp.empty()) ug_dict[ug_word] = en_hyp;
    }
  }
  return ug_dict;
}

// *******************************************************
// get_aligned_oov
// ******************************************************* 
map<string,set<string>> get_aligned_oov(string oov_aligned_file) {
  map<string,set<string>> aligned_oov = {};
  ifstream in(oov_aligned_file.c_str());
  string line;
  while(getline(in, line)) {
    vector<string> l;
    boost::split(l, line, boost::is_any_of("\t"));
    string oov = l[0];
    aligned_oov[oov]={};
    for (unsigned i=1;i<l.size();i++) aligned_oov.at(oov).insert(l[i]);
  }
  return aligned_oov;
}

// *******************************************************
// get_oov_candidates
// *******************************************************  
map<unsigned,set<int>> get_oov_candidates(SentStr oc_sent, Dict* dptr) {
  // oc_sent: converted-to-int oov candidates (not actual sentences)
  map<unsigned,set<int>> oov_candidates;
  for (auto ele:oc_sent) {
    vector<string> oc_pair;
    boost::split(oc_pair, ele, boost::is_any_of(":"));
    //cout<<oc_pair[0]<<endl;
    unsigned oov_pos = stoul(oc_pair[0],nullptr,0)+1; // consider <s>, the start symbol artificially added to the start of a sentence
    vector<string> candidates;
    boost::split(candidates, oc_pair[1], boost::is_any_of(","));
    set<int> candidates_int;
    for (auto candidate:candidates) candidates_int.insert(dptr->convert(candidate));
    oov_candidates[oov_pos] = candidates_int;
  }
    return oov_candidates;
}

// *******************************************************
// get_oov_candidates_str
// *******************************************************  
map<unsigned,vector<string>> get_oov_candidates_str(SentStr oc_sent) {
  // oc_sentl unconverted, string-formatted, oov candidates (not actual sentences)
  map<unsigned,vector<string>> oov_candidates;
  for (auto ele:oc_sent) {
    vector<string> oc_pair;
    boost::split(oc_pair, ele, boost::is_any_of(":"));
    //cout<<oc_pair[0]<<endl;
    unsigned oov_pos = stoul(oc_pair[0],nullptr,0)+1; // consider <s>, the start symbol artificially added to the start of a sentence
    vector<string> candidates;
    boost::split(candidates, oc_pair[1], boost::is_any_of(","));
    oov_candidates[oov_pos] = candidates;
  }
  return oov_candidates;
}

// *******************************************************
// compute normalized distance between two vectors
// *******************************************************    
float dist(const vector<float> &a, const vector<float> &b) {
  unsigned card = a.size();
  vector<float> diff(card);
  float res = 0;
  for (unsigned i=0;i<card;i++) {
    diff[i] = pow(a[i]-b[i],2);
    res += diff[i];
  }
  res = sqrt(res)/card;
  return res;
}    

// *******************************************************
// load model from an archive file
// *******************************************************
int load_model(string fname, Model& model){
  //fname += ".model";
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> model; 
  in.close();
  return 0;
}

// *******************************************************
// save model from an archive file
// *******************************************************
int save_model(string fname, Model& model){
  //fname += ".model";
  ofstream out(fname);
  boost::archive::text_oarchive oa(out);
  oa << model; 
  out.close();
  return 0;
}

// *******************************************************
// save dict from an archive file
// *******************************************************
int save_dict(string fname, dynet::Dict d){
  //fname += ".dict";
  ofstream out(fname);
  boost::archive::text_oarchive odict(out);
  odict << d; 
  out.close();
  return 0;
}

// *******************************************************
// load dict from an archive file
// *******************************************************
int load_dict(string fname, dynet::Dict& d){
  //fname += ".dict";
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> d; 
  in.close();
  return 0;
}

// *******************************************************
// read sentences and convect chars to indices
// *******************************************************
Sent MyReadSentenceChar(const std::string& line,
  Dict* sd,
  bool update) {

  vector<string> strs;
  for (auto c:line) {
    strs.push_back(string(1, c));
  }

  Sent res;
  res.push_back(sd->convert("<s>"));
  for (auto& c:strs) {
    if (update) {
      res.push_back(sd->convert(c));
    } else {
      if (sd->contains(c)) {
        res.push_back(sd->convert(c));
      } else {
        res.push_back(sd->convert("UNK"));
      }
    }
  }
  res.push_back(sd->convert("</s>"));
  return res;
}

// *******************************************************
// read sentences and convect tokens to indices
// *******************************************************
Sent MyReadSentence(const std::string& line, 
                    Dict* sd, 
                    bool update) {
  vector<string> strs;
  boost::split(strs, line, boost::is_any_of(" "));
  //istringstream in(line);
  //string word;
  Sent res;
  res.push_back(sd->convert("<s>"));
  for (auto& word : strs){
  //while (in){
    //in >> word;
    //if (word.empty()) break;
    //cerr << "word = " << word << endl;
    if (update){
      res.push_back(sd->convert(word));
    } else {
      if (sd->contains(word)){
          res.push_back(sd->convert(word));
      } else{
          //sd->set_unk(word);
          //res.push_back(sd->convert(word));
          res.push_back(sd->convert("UNK"));
      }
    }
  }
  res.push_back(sd->convert("</s>"));
  //for (auto ss:res) cout<<ss<<" ";cout<<endl;
  return res;
}

// *****************************************************
// makeDoc
// *****************************************************
Doc makeDoc(){
  vector<vector<int>> doc;
  return doc;
}

// *****************************************************
// makeDocStr
// *****************************************************
DocStr makeDocStr(){
  vector<vector<string>> doc;
  return doc;
}

// *****************************************************
// makeDocChar
// *****************************************************
DocChar makeDocChar(){
  vector<string> doc;
  return doc;
}

// *****************************************************
// read training and dev data and test data for character level models
// *****************************************************
CorpusChar readDataChar(char* filename) {
  //cerr << "reading data from "<< filename << endl;
  CorpusChar corpus;
  DocChar doc;
  string line;
  int tlc = 0;
  ifstream in(filename);
  while (getline(in, line)) {
    ++tlc;
    if (line[0] != '=') {
      if (!line.empty()) {
          doc.push_back(line);
      }
    } else {
		  if (doc.size()>0) {
		  	corpus.push_back(doc);
		  	doc = makeDocChar();
		  }
	  }
  }
	if (doc.size()>0) 
    corpus.push_back(doc);
  cerr<<corpus.size() << " docs, "<<tlc<<" lines. "<<endl;
  return(corpus);
}

// *****************************************************
// read training and dev data and test data without a dictionary
// *****************************************************
CorpusStr readDataStr(char* filename) {
    //cerr << "reading data from "<< filename << endl;
    CorpusStr corpus;
    DocStr doc;
    SentStr sent;
    string line;
    int tlc = 0;
    int toks = 0;
    ifstream in(filename);
    while (getline(in, line)) {
        ++tlc;
        if (line[0] != '=') {
            string word;
            SentStr sent;
            if (!line.empty()) {
                vector<string> strs;
                boost::split(strs, line, boost::is_any_of(" "));
                for (auto& word : strs) {
                    sent.push_back(word);
                }
                if (sent.size()>0){
                    doc.push_back(sent);
                    toks += doc.back().size();
                } else {
                  /* impossible for empty sentence because every sentence has <s> and </s>*/
                  cerr<<"Empty sentence! (shouldn't be here)";
                  doc.push_back(sent);
                }
            } else {
                doc.push_back(sent);
            }
        } else {
            if (doc.size()>0) {
                corpus.push_back(doc);
                doc = makeDocStr();
            } else {
                cerr<<"Empty document: " << endl;
            }
        }
    }
    if (doc.size()>0) corpus.push_back(doc);
    cerr<<corpus.size() << " docs, " << tlc << " lines, " << toks << " tokens."<<endl;
    return(corpus);
}

// *****************************************************
// read training and dev data and test data
// *****************************************************
Corpus readDataChar(
  char* filename, 
  dynet::Dict* dptr, 
  bool b_update) {

  //cerr << "reading data from: "<<filename << endl;
  Corpus corpus;
  Doc doc;
  Sent sent;
  string line;
  int tlc = 0;
  int toks = 0;
  ifstream in(filename);
  while (getline(in, line)) {
    ++tlc;
    if (line[0] != '=') {
      sent = MyReadSentenceChar(line, dptr, b_update);
      if (sent.size() > 0) {
        doc.push_back(sent);
        toks += sent.size();
      } else {
        cerr << "Empty sentence: " << line << endl;
      }
    } else {
      if (doc.size() > 0) {
        corpus.push_back(doc);
        doc = makeDoc();
      } else {
        cerr << "Empty document " << endl;
      }
    }
  }
  if (doc.size() > 0) {
    corpus.push_back(doc);
  }
  cerr << corpus.size() << " docs, " << tlc << " lines, " 
       << toks << " tokens, " << dptr->size() 
       << " types." << endl;
  return corpus;
}

// *****************************************************
// read training and dev data and test data
// *****************************************************
Corpus readData(char* filename, 
                dynet::Dict* dptr,
                bool b_update) {
  //cerr << "reading data from: "<< filename << endl;
  Corpus corpus;
  Doc doc;
  Sent sent;
  string line;
  int tlc = 0;
  int toks = 0;
  ifstream in(filename);
  while (getline(in, line)) {
    ++tlc;
      /* document boundary */
    if (line[0] != '='){
      //cout<<"aaa: "<<line<<endl;
      sent = MyReadSentence(line, dptr, b_update);
      if (sent.size() > 0){
          doc.push_back(sent);
          toks += doc.back().size();
      } else {
          cerr << "Empty sentence: " << line << endl;
      }
    } else {
      if (doc.size() > 0){
          corpus.push_back(doc);
          doc = makeDoc();
      } else {
          cerr << "Empty document " << endl;
      }
    }
  }
  if (doc.size() > 0){
    corpus.push_back(doc);
  }
  cerr << corpus.size() << " docs, " << tlc << " lines, " 
       << toks << " tokens, " << dptr->size() 
       << " types." << endl;
  return(corpus);
}

// ******************************************************
// Convert 1-D tensor to vector<float>
// so we can create an expression for it
// ******************************************************
vector<float> convertT2V(const Tensor& t){
  vector<float> vf;
  int dim = t.d.d[0];
  for (int idx = 0; idx < dim; idx++){
    vf.push_back(t.v[idx]);
  }
  return vf;
}

// ******************************************************
// Check the directory, if it doesn't exist, create one
// ******************************************************
int check_dir(string path){
  boost::filesystem::path dir(path);
  if(!(boost::filesystem::exists(dir))){
    if (boost::filesystem::create_directory(dir)){
      std::cout << "....Successfully Created !" << "\n";
    }
  }
}

// ******************************************************
// Segment a long document into several short ones
// ******************************************************
Corpus segment_doc(Corpus corpus, unsigned thresh){
  Corpus newcorpus;
  for (auto& doc : corpus){
    if (doc.size() <= thresh){
      newcorpus.push_back(doc);
      continue;
    }
    Doc tmpdoc;
    unsigned counter = 0;
    for (auto& sent : doc){
      if (counter <= thresh){
				tmpdoc.push_back(sent);
				counter ++;
      } else {
				newcorpus.push_back(tmpdoc);
				tmpdoc.clear();
				tmpdoc.push_back(sent);
				counter = 1;
      }
    }
    if (tmpdoc.size() > 0){
      newcorpus.push_back(tmpdoc);
      tmpdoc.clear();
    }
  }
  return newcorpus;
}
