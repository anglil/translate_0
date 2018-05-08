#include "dclm.h"
    
#include <dynet/training.h>
#include <dynet/timing.h>
#include <dynet/rnn.h>
#include <dynet/gru.h>
#include <dynet/lstm.h>
#include <dynet/dict.h>
    
#include "util.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <set>
#include <map>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/format.hpp>

using namespace std;
using namespace dynet;

#define ELPP_NO_DEFAULT_LOG_FILE
#include <easylogging++.h>
INITIALIZE_EASYLOGGINGPP

dynet::Dict d;
int kEOS, kSOS;

int train(
    char* train_file, 
    char* dev_file, 
    string model_type,
    unsigned num_layer,
    unsigned input_dim, 
    unsigned hidden_dim, 
    unsigned align_dim,
    unsigned len_thresh,
    string model_file, 
    string dict_file,
    string ppl_file,
    float lr
) {
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

    int improve_thres = 100;
    int improve_num = 0;

    Corpus training, dev;

    kSOS = d.convert("<s>");
    kEOS = d.convert("</s>");

    if (!boost::filesystem::exists(dict_file)) {
        LOG(INFO) << "Reading training data from: " << train_file;
        training = readData(train_file, &d, true);
        d.freeze();
        d.set_unk("UNK");	

        LOG(INFO) << "Reading dev data from: " << dev_file;
        dev = readData(dev_file, &d, false);
    } else {
        LOG(INFO) << "Load dict from: " << dict_file;
        load_dict(dict_file, d);
        d.freeze();

        LOG(INFO) << "Reading training data from: " << train_file;
        training = readData(train_file, &d, false);

        LOG(INFO) << "Reading dev data from: " << dev_file;
        dev = readData(dev_file, &d, false);
    }

    unsigned vocab_size = d.size();
    LOG(INFO) << "Vocab size = " << vocab_size;
    save_dict(dict_file, d);
    LOG(INFO) << "Save dict into: " << dict_file;
    training = segment_doc(training, len_thresh);
    LOG(INFO) << "Segmented training set size: " << training.size(); // segment corpus into smaller documents

    Model rmodel, amodel, omodel, hmodel;

    RNNLanguageModel<LSTMBuilder> rlm(
        rmodel,
        vocab_size,
        num_layer,
        input_dim,
        hidden_dim);
    
    DocumentAttentionalModel<LSTMBuilder> alm(
        amodel, 
        vocab_size, 
        num_layer, 
        input_dim, 
        hidden_dim, 
        align_dim);

    DCLMOutput<LSTMBuilder> olm(
        omodel, 
        vocab_size,
        num_layer, 
        input_dim, 
        hidden_dim);

    DCLMHidden<LSTMBuilder> hlm(
        hmodel, 
        vocab_size,
        num_layer, 
        input_dim, 
        hidden_dim);

    if (boost::filesystem::exists(model_file)) {
        LOG(INFO) << "Load model from: " << model_file;
        if (model_type == "rnnlm") {
            load_model(model_file, rmodel);
        } else if (model_type == "adclm") {
            load_model(model_file, amodel);
        } else if (model_type == "ccdclm") {
            load_model(model_file, hmodel);
        } else if (model_type == "codclm") {
            load_model(model_file, omodel);
        } else {
            LOG(INFO) << "Model "+model_type+" is not supported.";
            return -1;
        }
    } else {
        LOG(INFO) << "Randomly initializing model parameters ...";
    }

    Trainer* sgd = nullptr;
    if (model_type == "rnnlm") {
        sgd = new SimpleSGDTrainer(rmodel, lr);
    } else if (model_type == "adclm") {
        sgd = new SimpleSGDTrainer(amodel, lr);
    } else if (model_type == "ccdclm") {
        sgd = new SimpleSGDTrainer(hmodel, lr);
    } else if (model_type == "codclm") {
        sgd = new SimpleSGDTrainer(omodel, lr);
    } else {
        LOG(INFO) << "Model "+model_type+" is not supported.";
        return -1;
    }

    //cout << "optimization method specified"<<endl;

    unsigned report_every_i = 50;
    unsigned report_every_i_dev = 20;
    unsigned ctr = 0;
    unsigned training_size = training.size();

    vector<unsigned> order(training_size);

    for (unsigned i=0;i<order.size();++i) order[i]=i;
    bool first = true;
    int report = 0;
    unsigned lines = 0;

    //cout<<"start training"<<endl;

    while (true) {
        Timer iteration("complete in");
        double loss = 0;
        unsigned chars = 0;
        for (unsigned i=0; i<report_every_i; ++i) {
            if (ctr == training_size) {
                ctr = 0;
                if (first) {
                    first = false;
                } else {
                    sgd->update_epoch();
                }
                //LOG(INFO) << "*** SHUFFLE ***" << endl;
                shuffle(order.begin(), order.end(), *rndeng);
            }

            auto& doc = training[order[ctr]]; // random doc

            for (auto &sent: doc) {
                chars += sent.size()-1;
            }
            ++ctr;

            /*build a graph for this instance */
            ComputationGraph cg;

            Expression loss_expr;
            if (model_type=="rnnlm") {
                loss_expr = rlm.BuildGraph(doc, cg);
            } else if (model_type=="adclm") {
                loss_expr = alm.BuildGraph(doc, cg);
            } else if (model_type=="ccdclm") {
                loss_expr = hlm.BuildGraph(doc, cg);
            } else if (model_type=="codclm") {
                loss_expr = olm.BuildGraph(doc, cg);
            } else {
                LOG(INFO) << "Model "+model_type+" is not supported.";
                return -1;
            }

            loss += as_scalar(cg.forward(loss_expr));

            /* learning */
            cg.backward(loss_expr);
            sgd->update();
            ++lines;
        }
        sgd->status();

        LOG(INFO) << " E = " << boost::format("%1.4f")%(loss/chars)
            << " PPL = " << boost::format("%5.4f")%exp(loss/chars);

        /* show scores on the dev data */
        report++;
        if (report % report_every_i_dev == 0) {
            double dloss = 0;
            int dchars = 0;

            for (unsigned i=0;i<dev.size();++i) {
                const auto& doc = dev[i];
                ComputationGraph cg;

                Expression loss_expr;
                if (model_type=="rnnlm") {
                    loss_expr = rlm.BuildGraph(doc, cg);
                } else if (model_type=="adclm") {
                    loss_expr = alm.BuildGraph(doc, cg);
                } else if (model_type=="ccdclm") {
                    loss_expr = hlm.BuildGraph(doc, cg);
                } else if (model_type=="codclm") {
                    loss_expr = olm.BuildGraph(doc, cg);
                } else {
                    LOG(INFO)<<"Model "+model_type+" is not supported.";
                    return -1;
                }
                dloss += as_scalar(cg.forward(loss_expr));
                for (auto &sent: doc) {
                    dchars += sent.size() - 1;
                }
            }
            LOG(INFO) << "DEV [epoch = " << (lines / (double)training_size) << "] E = " 
                << boost::format("%1.4f") % (dloss/dchars) << " PPL = " 
                    << boost::format("%5.4f") % exp(dloss/dchars) << " (" 
                        << boost::format("%5.4f") % exp(best/dchars) << ") ";
            if (dloss < best) {
                improve_num = 0;
                best = dloss;
                LOG(INFO) << "Save model into: " << model_file;

                if (model_type == "rnnlm") {
                    save_model(model_file, rmodel);
                } else if (model_type == "adclm") {
                    save_model(model_file, amodel);
                } else if (model_type == "ccdclm") {
                    save_model(model_file, hmodel);
                } else if (model_type == "codclm") {
                    save_model(model_file, omodel);
                } else {
                    LOG(INFO) << "Model "+model_type+" is not supported.";
                    return -1;
                }
                ofstream ppl_stream;
                ppl_stream.open(ppl_file.c_str());
                ppl_stream << best;
                ppl_stream.close();
                LOG(INFO) << "Best perplexity "<<best<<" wrote to file.";
            } else {
                improve_num++;
                /* training ends if there is no improvement on dev set for a number of batches */
                if (improve_num >= improve_thres) {
                    return 0;
                }
            }
        }

    }//end of while(true)
    delete sgd;
    return 0;
}

/* test */
int test(
    char* test_file,
    string model_type,
    unsigned num_layer,
    unsigned input_dim, 
    unsigned hidden_dim, 
    unsigned align_dim,
    string model_file,
    string dict_file,
    string res_file
) {
    // load dict
    cerr << "Load dict from: " << dict_file << endl;
    load_dict(dict_file, d);
    d.freeze();
    unsigned vocab_size = d.size();
    cerr << "Vocab size = " << vocab_size << endl;


    // load model
    Model rmodel, amodel, omodel, hmodel;

    RNNLanguageModel<LSTMBuilder> rlm(
        rmodel,
        vocab_size,
        num_layer,
        input_dim,
        hidden_dim);

    DocumentAttentionalModel<LSTMBuilder> alm(
        amodel,
        vocab_size,
        num_layer,
        input_dim,
        hidden_dim,
        align_dim);

    DCLMOutput<LSTMBuilder> olm(
        omodel,
        vocab_size,
        num_layer,
        input_dim,
        hidden_dim);

    DCLMHidden<LSTMBuilder> hlm(
        hmodel,
        vocab_size,
        num_layer,
        input_dim,
        hidden_dim);

    cerr << "Load model from: " << model_file << endl;

    if (model_type == "rnnlm") {
        load_model(model_file, rmodel);
    } else if (model_type == "adclm") {
        load_model(model_file, amodel);
    } else if (model_type == "ccdclm") {
        load_model(model_file, hmodel);
    } else if (model_type == "codclm") {
        load_model(model_file, omodel);
    } else {
        cerr << "Model "+model_type+" is not supported." << endl;
        return -1;
    }

    /* load test data, recognizing doc boundaries "=" */
    /* with test data, there is no len_thresh that artificially segments the text into docs */
    cerr << "Read data from: " << test_file << endl;
    Corpus test = readData(test_file, &d, false);

    /* run test */
    ofstream fres; 
    fres.open(res_file);
    /* loss: loss over the whole text */
    /* dloss: loss over a doc */
    double loss(0), dloss(0);
    /* words: number of words in the whole text */
    /* dwords: number of words in a doc */
    int words(0), dwords(0);

    cerr << "Start computing ..." << endl;

    /* iterate over all the docs in the text */
    for (auto& doc : test) {
        ComputationGraph cg;

        Expression loss_expr;
        if (model_type == "rnnlm") {
            loss_expr = rlm.BuildGraph(doc, cg);
        } else if (model_type == "adclm") {
            loss_expr = alm.BuildGraph(doc, cg);
        } else if (model_type == "ccdclm") {
            loss_expr = hlm.BuildGraph(doc, cg);
        } else if (model_type == "codclm") {
            loss_expr = olm.BuildGraph(doc, cg);
        } else {
            cerr << "Model "+model_type+" is not supported." << endl;
            return -1;
        }

        dloss = as_scalar(cg.forward(loss_expr));
        loss += dloss;
        dwords = 0;
        for (auto& sent: doc) {
            dwords += (sent.size() - 1);
        }
        words += dwords;
        //cerr << boost::format("%5.4f") % exp(dloss/dwords) << endl;
        //fres << " PPL = " << boost::format("%5.4f") % exp(dloss/dwords) << endl;
    }
    cerr << " E = " 
        << boost::format("%1.4f") % (loss/words) 
        << " PPL = " 
        << boost::format("%5.4f") % exp(loss/words) << endl;
    fres << " PPL = "
        << boost::format("%5.4f") % exp(loss/words) << endl;
    fres.close();
    return 0;
}



/*
 * test on a set of files named through 0 to num-1 in a designated directory
 */
int test_group (string test_directory, 
                string model_type,
                unsigned num_layer,
                unsigned input_dim,
                unsigned hidden_dim,
                unsigned align_dim,
                string model_file,
                string dict_file,
                string res_directory,
                unsigned num) {

    check_dir(test_directory);
    check_dir(res_directory);

    /* load dict */
    cerr << "Load dict from: " << dict_file << endl;
    load_dict(dict_file, d);
    d.freeze();
    unsigned vocab_size = d.size();
    cerr << "Vocab size = " << vocab_size << endl;

    /* load model */
    Model rmodel, amodel, hmodel, omodel;

    RNNLanguageModel<LSTMBuilder> rlm(
        rmodel,
        vocab_size,
        num_layer,
        input_dim,
        hidden_dim);

    DocumentAttentionalModel<LSTMBuilder> alm(
        amodel,
        vocab_size,
        num_layer,
        input_dim,
        hidden_dim,
        align_dim);

    DCLMOutput<LSTMBuilder> olm(
        omodel,
        vocab_size,
        num_layer,
        input_dim,
        hidden_dim);

    DCLMHidden<LSTMBuilder> hlm(
        hmodel,
        vocab_size,
        num_layer,
        input_dim,
        hidden_dim);

    cerr << "Load model from: " << model_file << endl;

    if (model_type == "rnnlm") {
        load_model(model_file, rmodel);
    } else if (model_type == "adclm") {
        load_model(model_file, amodel);
    } else if (model_type == "ccdclm") {
        load_model(model_file, hmodel);
    } else if (model_type == "codclm") {
        load_model(model_file, omodel);
    } else {
        cerr << "Model "+model_type+" is not supported." << endl;
        return -1;
    }

    for (unsigned i=0; i<num; ++i) {
        /* load data from files in a directory */
        string test_file0 = test_directory + std::to_string(i);
        char* test_file = &test_file0[0u];
        cerr << "Read data from: " << test_file << endl;
        Corpus test = readData(test_file, &d, false);

        string res_file = res_directory + std::to_string(i);
        ofstream fres;
        fres.open(res_file);

        double loss(0), dloss(0);
        int words(0), dwords(0);
        cerr << "Start computing ..." << endl;
        for (auto& doc : test) {
            ComputationGraph cg;

            Expression loss_expr;
            if (model_type == "rnnlm") {
                loss_expr = rlm.BuildGraph(doc, cg);
            } else if (model_type == "adclm") {
                loss_expr = alm.BuildGraph(doc, cg);
            } else if (model_type == "ccdclm") {
                loss_expr = hlm.BuildGraph(doc, cg);
            } else if (model_type == "codclm") {
                loss_expr = olm.BuildGraph(doc, cg);
            } else {
                cerr << "Model "+model_type+" is not supported." << endl;
                return -1;		
            }

            //cg.print_graphviz();

            dloss = as_scalar(cg.forward(loss_expr));
            loss += dloss;
            dwords = 0;
            for (auto& sent: doc) {
                dwords += (sent.size() - 1);
            }
            words += dwords;
            //cerr << boost::format("%5.4f") % exp(dloss/dwords) << endl;
            //fres << " PPL = " << boost::format("%5.4f") % exp(dloss/dwords) << endl;
        }
        cerr << " E = " << boost::format("%1.4f") % (loss/words)
            << " PPL = " << boost::format("%5.4f") % exp(loss/words) << endl;
        fres << " PPL = " << boost::format("%5.4f") % exp(loss/words) << endl;
        fres.close();
    }
    return 0;
}

/*
 * test on a set of files named through 0 to num-1 in a designated directory
 */
int test_nbest (string sentences_file, 
                string results_file,
                string model_type,
                unsigned num_layer,
                unsigned input_dim,
                unsigned hidden_dim,
                unsigned align_dim,
                string model_file,
                string dict_file) {

    /* load dict */
    cerr << "Load dict from: " << dict_file << endl;
    load_dict(dict_file, d);
    d.freeze();
    unsigned vocab_size = d.size();
    cerr << "Vocab size = " << vocab_size << endl;

    /* load model */
    Model rmodel, amodel, hmodel, omodel;

    RNNLanguageModel<LSTMBuilder> rlm(
        rmodel,
        vocab_size,
        num_layer,
        input_dim,
        hidden_dim);

    DocumentAttentionalModel<LSTMBuilder> alm(
        amodel,
        vocab_size,
        num_layer,
        input_dim,
        hidden_dim,
        align_dim);

    DCLMOutput<LSTMBuilder> olm(
        omodel,
        vocab_size,
        num_layer,
        input_dim,
        hidden_dim);

    DCLMHidden<LSTMBuilder> hlm(
        hmodel,
        vocab_size,
        num_layer,
        input_dim,
        hidden_dim);

    cerr << "Load model from: " << model_file << endl;

    if (model_type == "rnnlm") {
        load_model(model_file, rmodel);
    } else if (model_type == "adclm") {
        load_model(model_file, amodel);
    } else if (model_type == "ccdclm") {
        load_model(model_file, hmodel);
    } else if (model_type == "codclm") {
        load_model(model_file, omodel);
    } else {
        cerr << "Model "+model_type+" is not supported." << endl;
        return -1;
    }

    ifstream in(sentences_file.c_str());
    ofstream out(results_file.c_str());
    
    unsigned ctr=0;
    unsigned ctr_res=0;
    string num_pre="0";
    string num_cur="0";
    double best_loss=9e+99;
    string best_sent="";
    
    string line;
    while(getline(in, line)) {
        //cout<<line<<endl;
        //vector<string> parse;
        //boost::split(parse, line, boost::is_any_of("|||"));
        vector<string> parse = split(line, '|');
        
        num_cur = parse[0];
        boost::trim(num_cur);
        string sent_str = parse[3];
        boost::trim(sent_str);
        //cout<<num_cur<<endl;
        //cout<<sent_str<<endl;
        boost::algorithm::to_lower(sent_str);
        //cout<<sent_str<<endl;
        //break;
        
        Sent sent = MyReadSentence(sent_str, &d, false);
        //cout<<sent_str<<endl;
        Doc doc = {sent};
        ComputationGraph cg;
        Expression loss_expr;
        if (model_type == "rnnlm") {
            loss_expr = rlm.BuildGraph(doc, cg);
        } else if (model_type == "adclm") {
            loss_expr = alm.BuildGraph(doc, cg);
        } else if (model_type == "ccdclm") {
            loss_expr = hlm.BuildGraph(doc, cg);
        } else if (model_type == "codclm") {
            loss_expr = olm.BuildGraph(doc, cg);
        } else {
            cerr << "Model "+model_type+" is not supported." << endl;
            return -1;
        }
        
        double loss = as_scalar(cg.forward(loss_expr));
        
        if (num_cur == num_pre) {
            if (loss<best_loss) {
                best_loss = loss;
                best_sent = sent_str;
            }
        } else {
            num_pre = num_cur;
            ctr_res++;
            
            cout << best_sent << endl;
            out << best_sent << endl;
            
            best_loss=9e+99;
            best_sent="";
            
            if (ctr_res%100==0) cout<<ctr_res<<" sentences extracted."<<endl;
        }  
        ctr++;
        
    }
    cout << best_sent << endl;
    out << best_sent << endl;
    ctr_res++;
    
    out.close();
    in.close();
    
    cout<<"--------"<<endl;
    cout<<ctr<<" sentences processed."<<endl;
    cout<<ctr_res<<" sentences selected."<<endl;
    
    return 0;
}

/* main */
int main(int argc, char** argv) {
    // argc: number of arguments
    // argv: list of arguments
    dynet::initialize(argc, argv);

    string cmd = argv[1];
    cout<<"Task: "<<cmd<<endl;

    if (cmd == "train") {
        char* train_file = argv[2];
        char* dev_file = argv[3];
        string model_type = argv[4];
        unsigned num_layer = atoi(argv[5]);
        unsigned input_dim = atoi(argv[6]);
        unsigned hidden_dim = atoi(argv[7]);
        unsigned align_dim = atoi(argv[8]);
        unsigned len_thresh = atoi(argv[9]);
        string model_file = argv[10];
        string dict_file = argv[11];
        string ppl_file = argv[12];
        string log_file = argv[13];		
        float lr = atof(argv[14]);

        el::Configurations defaultConf;
        defaultConf.set(el::Level::Info, el::ConfigurationType::Format, "%datetime{%h:%m:%s} %level %msg");
        defaultConf.set(el::Level::Info, el::ConfigurationType::Filename, log_file.c_str());
        el::Loggers::reconfigureLogger("default", defaultConf);

        train(
            train_file, 
            dev_file, 
            model_type,
            num_layer,
            input_dim, 
            hidden_dim, 
            align_dim, 
            len_thresh,
            model_file,
            dict_file,
            ppl_file,
            lr);

    }  else if (cmd == "test") {
        char* test_file = argv[2];	
        string model_type = argv[3];
        unsigned num_layer = atoi(argv[4]);
        unsigned input_dim = atoi(argv[5]);
        unsigned hidden_dim = atoi(argv[6]);
        unsigned align_dim = atoi(argv[7]);
        string model_file = argv[8];
        string dict_file = argv[9];
        string res_file = argv[10];    

        test(
            test_file, 
            model_type,
            num_layer,
            input_dim,
            hidden_dim,
            align_dim,
            model_file,
            dict_file,
            res_file);
        return -1;
    
    } else if (cmd == "test_group") {
        string test_directory = argv[2];
        string model_type = argv[3];
        unsigned num_layer = atoi(argv[4]);
        unsigned input_dim = atoi(argv[5]);
        unsigned hidden_dim = atoi(argv[6]);
        unsigned align_dim = atoi(argv[7]);
        string model_file = argv[8];
        string dict_file = argv[9];
        string res_directory = argv[10];
        unsigned num = atoi(argv[11]);

        test_group(
            test_directory, 
            model_type,
            num_layer,
            input_dim,
            hidden_dim,
            align_dim,
            model_file,
            dict_file,
            res_directory,
            num);
        return -1;

    } else if (cmd == "test_nbest") {
        string sentences_file = argv[2];
        string results_file = argv[3];
        string model_type = argv[4];
        unsigned num_layer = atoi(argv[5]);
        unsigned input_dim = atoi(argv[6]);
        unsigned hidden_dim = atoi(argv[7]);
        unsigned align_dim = atoi(argv[8]);
        string model_file = argv[9];
        string dict_file = argv[10];

        test_nbest(
            sentences_file,
            results_file,
            model_type,
            num_layer,
            input_dim,
            hidden_dim,
            align_dim,
            model_file,
            dict_file);
        return -1;
        
    } else {
        throw std::invalid_argument("Task "+cmd+" not supported.");
    }
    return 0;
}








