#pragma once

#include <dynet/globals.h>
#include <dynet/nodes.h>
#include <dynet/dynet.h>
#include <dynet/expr.h>

#include <iostream>
#include <map>
#include <set>
#include <string>
#include "util.hpp"


/* sentence level lstm */
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

        /* instantiator */
        Expression BuildGraph(const Doc doc, ComputationGraph& cg) {

            builder.new_graph(cg);

            /* hidden -> word rep parameter */
            Expression i_R = parameter(cg, p_R);
            /* word bias */
            Expression i_bias = parameter(cg, p_bias);

            Expression i_x_t, i_y_t, i_r_t, i_err;

            vector<Expression> errs;
            for (const auto &sent: doc ) {
                builder.start_new_sequence();
                unsigned sent_len = sent.size() - 1;
                for (unsigned t=0;t<sent_len;t++) {
                    i_x_t = lookup(cg, p_c, sent[t]);
                    i_y_t = builder.add_input(i_x_t);
                    i_r_t = i_bias + i_R * i_y_t;

                    i_err = pickneglogsoftmax(i_r_t, sent[t+1]);
                    errs.push_back(i_err);
                }
            }
            Expression i_nerr = sum(errs);
            return i_nerr;
        }
        
};


/* attentional dclm */
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

        /* instantiator */
        Expression BuildGraph(const Doc doc, ComputationGraph& cg) {
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

            vector<Expression> errs;
            bool first = true;

            for (const auto &sent: doc ) {

                /******** start_new_sentence ********/
                if (!first) context.push_back(concatenate(builder.final_h()));
        
                builder.start_new_sequence();
                if (context.size() > 1) {
                    src = concatenate_cols(context);
                    i_uax = i_Ua * src;
                }
                /****************/

                unsigned sent_len = sent.size() - 1;

                for (unsigned t=0;t<sent_len;t++) {
                    /********* add_input ********/
                    Expression i_x_t = lookup(cg, p_c, sent[t]);
                    Expression i_c_t;
                    if (context.size() > 1) {
                        Expression i_wah_rep;
                        if (t > 0) {
                            auto i_h_tml = concatenate(builder.final_h());
                            Expression i_wah = i_Wa * i_h_tml;
                            i_wah_rep = concatenate_cols(vector<Expression>(context.size(), i_wah));
                        }

                        Expression i_e_t;
                        if (t > 0) i_e_t = transpose(tanh(i_wah_rep+i_uax))*i_va;
                        else i_e_t = transpose(tanh(i_uax))*i_va;

                        Expression i_alpha_t = softmax(i_e_t);
                        i_c_t = src * i_alpha_t;
                    } else if (context.size() == 1) {
                        i_c_t = context.back();
                    } else {
                        i_c_t = i_empty;
                    }

                    Expression input = concatenate(vector<Expression>({i_x_t, i_c_t}));
                    Expression i_y_t = builder.add_input(input);
                    Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
                    Expression i_r_t = affine_transform({i_bias, i_R, i_tildet_t});
                    /****************/

                    Expression i_err = pickneglogsoftmax(i_r_t, sent[t+1]);

                    errs.push_back(i_err);
                }
                first = false;
            }
            Expression i_nerr = sum(errs);
            return i_nerr;
        }
    
};


/* ccdclm */
template <class Builder> class DCLMHidden {
    private:
        LookupParameter p_c; // word embeddings VxK1
        Parameter p_R; // output layer: VxK2
        Parameter p_bias; // bias Vx1
        Parameter p_context; // default context vector
        Parameter p_transform; // transformation matrix
        Builder builder;
    public:
        DCLMHidden();

        /* constructor */
        DCLMHidden(
            Model& model, 
            unsigned vocab_size,
            unsigned num_layer,
            unsigned input_dim,
            unsigned hidden_dim
        ) : builder(num_layer, input_dim+hidden_dim, hidden_dim, model) {
            p_c = model.add_lookup_parameters(vocab_size, {input_dim});
            // for hidden output
            p_R = model.add_parameters({vocab_size, hidden_dim});
            // for bias
            p_bias = model.add_parameters({vocab_size});
            // for default context vector
            p_context = model.add_parameters({hidden_dim});
        }

        /* build a ccdclm graph, which returns an Expression */
        Expression BuildGraph(const Doc doc, ComputationGraph& cg) {
            // reset RNN builder for new graph
            builder.new_graph(cg);
            // define expression
            Expression i_R = parameter(cg, p_R);
            Expression i_bias = parameter(cg, p_bias);
            Expression i_context = parameter(cg, p_context);
            Expression cvec, i_x_t, i_h_t, i_y_t, i_err;
            vector<Expression> vec_exp;

            /* build computation graph for the doc */
            vector<Expression> errs;

            /* iterate through all sentences in the doc */
            for (unsigned k = 0; k < doc.size(); k++) {
                // start a new sequence for each sentence
                builder.start_new_sequence();
                // for each sentence in this doc
                auto sent = doc[k];
                unsigned sent_len = sent.size() - 1;
                // get context vector if this is the first sent
                if (k == 0) {
                    cvec = i_context;
                }
                // build RNN for the current sentence
                /* iterate through all words in the sentence */
                for (unsigned t = 0; t < sent_len; t++) {
                    // get word representation
                    i_x_t = lookup(cg, p_c, sent[t]);
                    vec_exp.clear();
                    // add context vector
                    vec_exp.push_back(i_x_t);
                    vec_exp.push_back(cvec);
                    i_x_t = concatenate(vec_exp);
                    // compute hidden state
                    i_h_t = builder.add_input(i_x_t);
                    // compute prediction
                    i_y_t = (i_R * i_h_t) + i_bias;
                    // get prediction error
                    i_err = pickneglogsoftmax(i_y_t, sent[t+1]);
                    // add back error
                    errs.push_back(i_err);
                }
                // update context vector
                cvec = i_h_t;
            }
            Expression i_nerr = sum(errs);
            return i_nerr;
        }
    
};


/* codclm */
template <class Builder> class DCLMOutput { 
    private:
        LookupParameter p_c; // word embeddings VxK1
        Parameter p_R; // output layer: VxK2
        Parameter p_R2; // forward context vector: VxK2
        Parameter p_bias; // bias Vx1
        Parameter p_context; // default context vector for sent-level
        Builder builder;
    public:
        DCLMOutput();

        /* constructor */
        DCLMOutput(
            Model& model,
            unsigned vocab_size,
            unsigned num_layer,
            unsigned input_dim,
            unsigned hidden_dim
        ) : builder (num_layer, input_dim, hidden_dim, model) {
            p_c = model.add_lookup_parameters(vocab_size, {input_dim});
            // for hidden output
            p_R = model.add_parameters({vocab_size, hidden_dim});
            // for forward context vector
            p_R2 = model.add_parameters({vocab_size, hidden_dim});
            // for bias
            p_bias = model.add_parameters({vocab_size});
            // for default context vector
            p_context = model.add_parameters({hidden_dim});
        }

        /* build a codclm, which returns an Expression */
        Expression BuildGraph(const Doc doc, ComputationGraph& cg) {
            // reset RNN builder for new graph
            builder.new_graph(cg);
            // define expression
            Expression i_R = parameter(cg, p_R);
            Expression i_R2 = parameter(cg, p_R2);
            Expression i_bias = parameter(cg, p_bias);
            Expression i_context = parameter(cg, p_context);
            Expression cvec, i_x_t, i_h_t, i_y_t, i_err, ccpb;

            // build CG for the doc
            vector<Expression> errs;
            for (unsigned k = 0; k < doc.size(); k++) {
                builder.start_new_sequence();
                // for each sentence in this doc
                auto sent = doc[k];
                unsigned sent_len = sent.size() - 1;
                // start a new sequence for each sentence
                if (k == 0) cvec = i_context;
                // build RNN for the current sentence
                ccpb = (i_R2 * cvec) + i_bias;
                for (unsigned t = 0; t < sent_len; t++) {
                    // get word representation
                    i_x_t = lookup(cg, p_c, sent[t]);
                    // compute hidden state
                    i_h_t = builder.add_input(i_x_t);
                    // compute prediction
                    i_y_t = (i_R * i_h_t) + ccpb;
                    // get prediction error
                    // the reason for [t+1] is because sent[0] is "<s>", a symbol artificially added to the beginning of the sentence
                    i_err = pickneglogsoftmax(i_y_t, sent[t+1]);
                    // add error back
                    errs.push_back(i_err);
                }
                // update context vector
                cvec = i_h_t;
            }
            Expression i_nerr = sum(errs);
            return i_nerr;
        }

};


