#include "util.hpp"
    
#include <dynet/globals.h>
#include <dynet/dynet.h>
#include <dynet/training.h>
#include <dynet/lstm.h>

#include <stdexcept>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/format.hpp>

using namespace std;
using namespace dynet;

#define ELPP_no_DEFAULT_LOG_FILE
#include <easylogging++.h>
INITIALIZE_EASYLOGGINGPP

class charlm_lstm {
private:
	LookupParameter p_c;
	Parameter p_R;
	Parameter p_bias;
	LSTMBuilder* builder;

	dynet::Dict d;
	string kSOS = "<s>";
	string kEOS = "</s>";
public:
	// initialize parameters, as a constructor
	void initialize(
		Model& model,
		unsigned vocab_size,
		unsigned num_layer,
		unsigned input_dim,
		unsigned hidden_dim) {

		builder = new LSTMBuilder(num_layer, input_dim, hidden_dim, model);
		p_c = model.add_lookup_parameters(vocab_size, {input_dim});
		p_R = model.add_parameters({vocab_size, hidden_dim});
		p_bias = model.add_parameters({vocab_size});
	}


	// initialize parameters and dictionary, used at test time when a dictionary is available
	void initialize_with_dict(
		Model& model,
		string dict_file,
		unsigned num_layer,
		unsigned input_dim,
		unsigned hidden_dim) {

		// load dictionary
	  if (!boost::filesystem::exists(dict_file)) {
	  	cerr << "Dict file doesn't exist at: " << dict_file;
	  	return;
	  } else {
	  	LOG(INFO) << "Loading dict from: " << dict_file;
	  	load_dict(dict_file, d);
	  	d.freeze();
	  }
	  
	  unsigned vocab_size = d.size();
	  LOG(INFO) << "Vocab size = " << vocab_size;

	  this->initialize(model, vocab_size, num_layer, input_dim, hidden_dim);
	}


	// the Selected data structure
	struct Selected {
		double best_loss = 9e+99;
		string best_sent = "";
		unsigned best_idx = 0;
	};


	// select the sentence with least loss, used to complement word-level LM
	Selected score_pick(ComputationGraph& cg, vector<string> sentences) {
		Selected selected;

		unsigned ctr = 0;
		for (string sentence:sentences) {
			Sent sent = MyReadSentenceChar(sentence, &d, false);

			builder->new_graph(cg);
			Expression i_R = parameter(cg, p_R);
			Expression i_bias = parameter(cg, p_bias);
			Expression i_x_t, i_y_t, i_r_t, i_err;

			builder->start_new_sequence();
			unsigned sent_len = sent.size();
			vector<Expression> errs;
			for (unsigned t=0;t<sent_len-1;t++) {
				i_x_t = lookup(cg, p_c, sent[t]);
				i_y_t = builder->add_input(i_x_t);
				i_r_t = i_bias + i_R * i_y_t;

				i_err = pickneglogsoftmax(i_r_t, sent[t+1]);
				errs.push_back(i_err);
			}
			double loss = as_scalar(cg.forward(sum(errs)));
			if (loss < selected.best_loss) {
				selected.best_loss = loss;
				selected.best_sent = sentence;
				selected.best_idx = ctr;
			}
			ctr++;
		}
		return selected;
	}

	// get loss at the sentence level
	Expression get_loss(const Sent sent, ComputationGraph& cg) {
		builder->new_graph(cg);
		Expression i_R = parameter(cg, p_R);
		Expression i_bias = parameter(cg, p_bias);
		Expression i_x_t, i_y_t, i_r_t, i_err;

		builder->start_new_sequence();
		unsigned sent_len = sent.size();

    vector<Expression> errs;
		for (unsigned t=0;t<sent_len-1;t++) {
			i_x_t = lookup(cg, p_c, sent[t]);
			i_y_t = builder->add_input(i_x_t);
			i_r_t = i_bias + i_R * i_y_t;

			i_err = pickneglogsoftmax(i_r_t, sent[t+1]);
			errs.push_back(i_err);
		}
		Expression i_nerr = sum(errs);
		return i_nerr;
	}

  // test
	int test(
    char* test_file,

		unsigned num_layer,
		unsigned input_dim,
		unsigned hidden_dim,

		string model_file,
		string dict_file,
    string log_file) {

		// load model and dict
		Model charmodel;
		this->initialize_with_dict(charmodel, dict_file, num_layer, input_dim, hidden_dim);
		load_model(model_file, charmodel);

		// load data
		Corpus test = readDataChar(test_file, &d, false);
		for (auto& doc:test) {
      for (auto& sent:doc) {
			  ComputationGraph cg;
  			auto loss_expr = get_loss(sent, cg);
        double loss = as_scalar(cg.forward(loss_expr));
      }
		}
	}

  // train
	int train(
		char* train_file,
		char* dev_file,

		unsigned num_layer,
		unsigned input_dim,
		unsigned hidden_dim,

		string model_file,
		string dict_file,
    string ppl_file,
    string log_file,

		float lr) {

		double best = 9e+99;
    if (boost::filesystem::exists(ppl_file)) {
      ifstream ppl_stream(ppl_file.c_str());
      if (ppl_stream.good()) {
        string ppl_line;
        getline(ppl_stream, ppl_line);
        best = atof(ppl_line.c_str());
      }
    }
    LOG(INFO) << "The previous best loss: " << best;

	  int improve_thres = 1000;
	  int improve_num = 0;

	  Corpus training, dev;

	  // load dictionary and data
	  if (!boost::filesystem::exists(dict_file)) {
	  	LOG(INFO) << "Reading training data from: " << train_file;
	  	training = readDataChar(train_file, &d, true);
	  	d.freeze();
	  	d.set_unk("UNK");

	  	LOG(INFO) << "Reading dev data from: " << dev_file;
	  	dev = readDataChar(dev_file, &d, false);
	  } else {
	  	LOG(INFO) << "Loading dict from: " << dict_file;
	  	load_dict(dict_file, d);
	  	d.freeze();

      LOG(INFO) << "Reading training data from: " << train_file;
	  	training = readDataChar(train_file, &d, false);

      LOG(INFO) << "Reading dev data from: " << dev_file;
	  	dev = readDataChar(dev_file, &d, false);
	  }
	  
	  unsigned vocab_size = d.size();
	  LOG(INFO) << "Vocab size = " << vocab_size;
	  save_dict(dict_file, d);
	  LOG(INFO) << "Save dict into: " << dict_file;

	  // load model / initialize model
	  Model model;
	  this->initialize(model, vocab_size, num_layer, input_dim, hidden_dim);

	  if (boost::filesystem::exists(model_file)) {
          LOG(INFO) << "Load model from: " << model_file;
          load_model(model_file, model);
      } else {
          LOG(INFO) << "Randomly initializing model parameters ...";
      }

    SimpleSGDTrainer trainer(model, lr);

    unsigned report_every_i = 50;
    unsigned report_every_i_dev = 20;
    unsigned ctr = 0; // document counter
    unsigned training_size = training.size();

    vector<unsigned> order(training_size);

    for (unsigned i=0; i<order.size(); ++i) order[i]=i;
    bool first = true;
    int report = 0;
    unsigned lines = 0;

    while (true) {
    	Timer iteration("complete in");

    	// train on the training data
    	double loss = 0;
    	unsigned chars = 0;
    	
    	for (unsigned i=0; i<report_every_i; ++i) {
    		if (ctr == training_size) {
    			ctr = 0;
    			if (first) {
    				first = false;
    			} else {
    				trainer.update_epoch();
    			}
    			shuffle(order.begin(), order.end(), *rndeng);
    		}

    		auto& doc = training[order[ctr]];
    		for (auto &sent:doc) {
    			chars += sent.size();
    		  ComputationGraph cg;
    		  Expression loss_expr = this->get_loss(sent, cg);
    		  loss += as_scalar(cg.forward(loss_expr));

    		  cg.backward(loss_expr);
    		  trainer.update(); // update model
        }
    		++ctr; 
        ++lines;
    	}
    	trainer.status();

    	LOG(INFO) << " E = " << boost::format("%1.4f")%(loss/chars)
    		<< " PPL = " << boost::format("%5.4f")%exp(loss/chars);

    	report++;

    	if (report % report_every_i_dev == 0) {

    		// score on the dev data
    		double dloss = 0;
    		unsigned dchars = 0;

    		for (unsigned i=0; i<dev.size(); ++i) {
    			const auto& doc = dev[i];
    			for (auto &sent:doc) {
    				dchars += sent.size();              
            ComputationGraph cg;
    			  Expression loss_expr = this->get_loss(sent, cg);
    			  dloss += as_scalar(cg.forward(loss_expr));
    			}
    		}
    		LOG(INFO) << "DEV [epoch = " << (lines / (double)training_size) << "] E = " 
              	<< boost::format("%1.4f") % (dloss/dchars) << " PPL = " 
              	<< boost::format("%5.4f") % exp(dloss/dchars) << " (" 
              	<< boost::format("%5.4f") % exp(best/dchars) << ") ";

			  if (dloss < best) {
			  	improve_num = 0;
			  	best = dloss;

			  	save_model(model_file, model);
			  	LOG(INFO) << "Save model into: " << model_file;

          ofstream ppl_stream;
          ppl_stream.open(ppl_file.c_str());
          ppl_stream << best;
          ppl_stream.close();
          LOG(INFO) << "Best perplexity "<<best<<" wrote to file.";
			  } else {
			  	improve_num++;
			  	if (improve_num >= improve_thres) {
			  		LOG(INFO) << "Training finished.";
			  		return 0;
			  	}
			  }
    	}
    }
    return 0;
	}
};
