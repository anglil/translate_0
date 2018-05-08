from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys 
import copy

from six.moves import xrange

from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
import tensorflow as tf

def flatten5d4d(x):
    xshape = tf.shape(x)
    result = tf.reshape(x, [xshape[0], xshape[1] * xshape[2], xshape[3], xshape[4]])
    xshape_static = x.get_shape()
    result.set_shape([xshape_static[0], None, xshape_static[3], xshape_static[4]])
    return result


@registry.register_model
class TransformerLex(transformer.Transformer):
    '''
    inherits Transformer which inherits t2t_model.T2TModel
    '''
    def model_fn_body(self, features):
        hparams = self._hparams
        
        encoder_input = features["inputs"]
        print(encoder_input.shape.as_list()) # ==> [None, None, None, 4, 300]
        encoder_input = flatten5d4d(encoder_input)
        print(encoder_input.shape.as_list()) # ==> [None, None, 4, 300]
        target_space = features["target_space_id"]
        print(target_space.shape.as_list()) # ==> []
        # encode_lex
        encoder_output, encoder_decoder_attention_bias = self.encode_lex(encoder_input, target_space, hparams)
        targets = features["targets"]
        print(targets.shape.as_list())
        targets = common_layers.flatten4d3d(targets)
        # decode_lex
        decoder_input, decoder_self_attention_bias = transformer.transformer_prepare_decoder(targets, hparams)
        decoder_output = self.decode(decoder_input, encoder_output, encoder_decoder_attention_bias, decoder_self_attention_bias, hparams)
        return decoder_output


    def encode_lex(self, encoder_input, target_space, hparams):
        '''
        encoder_input: [batch_size, input_len, hidden_dim]
        return: 
            encoder_output: [batch_size, input_len, hidden_dim]
            encoder_decoder_attention_bias: [batch_size, input_len]
        '''
        encoder_output_slices = []
        for i in range(encoder_input.get_shape()[2].value):
            encoder_input_slice = encoder_input[:,:,i,:]

            # bias
            encoder_padding = common_attention.embedding_to_padding(encoder_input_slice)
            print(encoder_padding.shape.as_list()) # ==> [None, None] (None, None, 4)
            ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
            encoder_self_attention_bias = ignore_padding
            encoder_decoder_attention_bias = ignore_padding
            print(ignore_padding.shape.as_list()) # ==> [None, 1, 1, None] (None, 1, 1, None, 4)

            # add target space to encoder input?
            ishape_static = encoder_input_slice.shape.as_list()
            print(ishape_static) # ==> [None, None, 300] (None, None, 4, 300)
            emb_target_space = common_layers.embedding(target_space, 32, ishape_static[-1], name="target_space_embedding")
            print(emb_target_space.shape.as_list()) # ==> [300]
            emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
            print(emb_target_space.shape.as_list()) # ==> [1, 1, 300]
            encoder_input_slice += emb_target_space
            print(encoder_input_slice.shape.as_list()) # ==> [None, None, 300] (None, None, 4, 300)

            # add timing signals to encoder input
            if hparams.pos == "timing":
                encoder_input_slice = common_attention.add_timing_signal_1d(encoder_input_slice)

            # dropout
            encoder_input_slice = tf.nn.dropout(encoder_input_slice, 1.0-hparams.layer_prepostprocess_dropout)
            encoder_output_slices.append(encoder_input_slice)
        encoder_output = tf.stack(encoder_output_slices, 2)
            
        # --------

        num_heads = int(hparams.lex_cap/2)
        hparams2 = tf.contrib.training.HParams(
            layer_preprocess_sequence=hparams.layer_preprocess_sequence,
            layer_postprocess_sequence=hparams.layer_postprocess_sequence,
            layer_prepostprocess_dropout=hparams.layer_prepostprocess_dropout,
            norm_type=hparams.norm_type,
            hidden_size=hparams.lex_cap,
            norm_epsilon=hparams.norm_epsilon,
            ffn_layer=hparams.ffn_layer,
            filter_size=hparams.filter_size,
            relu_dropout=hparams.relu_dropout,
            num_heads=num_heads,
            attention_dropout=hparams.attention_dropout,
            parameter_attention_key_channels=hparams.parameter_attention_key_channels,
            parameter_attention_value_channels=hparams.parameter_attention_value_channels)

        encoder_output_slices = []
        for i in range(encoder_output.get_shape()[3].value):
            encoder_input_slice = encoder_output[:,:,:,i]
            #print(encoder_input_slice.shape.as_list()) # ==> [None, None, 4]

            encoder_padding = common_attention.embedding_to_padding(encoder_input_slice)
            ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
            encoder_self_attention_bias = ignore_padding
            #print(encoder_self_attention_bias.shape.as_list()) # ==> [None, 1, 1, None]

            # encoder
            x = encoder_input_slice
            with tf.variable_scope("encoder_extra"+str(i)):
                # remove pad 
                pad_remover = None
                if hparams.use_pad_remover:
                    pad_remover = expert_utils.PadRemover(common_attention.attention_bias_to_padding(encoder_self_attention_bias))

                # self-attention along the lexicon dimension
                with tf.variable_scope("layer_extra"):
                    with tf.variable_scope("self_attention"):
                        #query_antecedent = layer_preprocess2(x, hparams, hparams.lex_cap)
                        query_antecedent = common_layers.layer_preprocess(x, hparams2)

                        y = common_attention.multihead_attention(
                            query_antecedent=query_antecedent,
                            memory_antecedent=None,
                            bias=encoder_self_attention_bias,
                            total_key_depth=hparams.attention_key_channels or hparams.lex_cap,
                            total_value_depth=hparams.attention_value_channels or hparams.lex_cap,
                            output_depth=hparams.lex_cap,
                            num_heads=num_heads,
                            dropout_rate=hparams.attention_dropout,
                            attention_type=hparams.self_attention_type,
                            max_relative_position=hparams.max_relative_position)
                        x = common_layers.layer_postprocess(x, y, hparams2)
                    with tf.variable_scope("ffn"):
                        y = transformer.transformer_ffn_layer(common_layers.layer_preprocess(x, hparams2), hparams2, pad_remover)
                        x = common_layers.layer_postprocess(x, y, hparams2)
                encoder_output_slice = common_layers.layer_preprocess(x, hparams2)

            encoder_output_slices.append(encoder_output_slice)
        encoder_output = tf.stack(encoder_output_slices, 3)
        print(encoder_output.shape.as_list()) # ==> [None, None, 4, 300]

        # --------

        lex_cap = encoder_output.get_shape()[2].value
        embed_len = encoder_output.get_shape()[3].value
        assert(lex_cap == hparams.lex_cap)
        aggregate_layer = tf.get_variable(
            name="Aggregate",
            shape=[embed_len, embed_len, lex_cap],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        encoder_output = tf.tensordot(encoder_output, aggregate_layer, axes=[[2,3],[1,2]])
        print(encoder_output.shape.as_list()) # ==> [None, None, 300]

        # --------

        # encoder
        x = encoder_output
        with tf.variable_scope("encoder"):
            # remove pad 
            pad_remover = None
            if hparams.use_pad_remover:
                pad_remover = expert_utils.PadRemover(common_attention.attention_bias_to_padding(encoder_self_attention_bias))

            # self-attention along the sentence dimension
            for layer in xrange(hparams.num_encoder_layers or hparams.num_hidden_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        query_antecedent = common_layers.layer_preprocess(x, hparams)
                        y = common_attention.multihead_attention(
                            query_antecedent=query_antecedent,
                            memory_antecedent=None,
                            bias=encoder_self_attention_bias,
                            total_key_depth=hparams.attention_key_channels or hparams.hidden_size,
                            total_value_depth=hparams.attention_value_channels or hparams.hidden_size,
                            output_depth=hparams.hidden_size,
                            num_heads=hparams.num_heads,
                            dropout_rate=hparams.attention_dropout,
                            attention_type=hparams.self_attention_type,
                            max_relative_position=hparams.max_relative_position)
                        x = common_layers.layer_postprocess(x, y, hparams)
                    with tf.variable_scope("ffn"):
                        y = transformer.transformer_ffn_layer(common_layers.layer_preprocess(x, hparams), hparams, pad_remover)
                        x = common_layers.layer_postprocess(x, y, hparams)
            encoder_output = common_layers.layer_preprocess(x, hparams)
            print(encoder_output.shape.as_list()) # ==> [None, None, 300] (None, None, 4, 300)
        return encoder_output, encoder_decoder_attention_bias


@registry.register_model
class TransformerLexafter(transformer.Transformer):
    '''
    inherits Transformer which inherits t2t_model.T2TModel
    '''
    def model_fn_body(self, features):
        hparams = self._hparams
        
        encoder_input = features["inputs"]
        print(encoder_input.shape.as_list()) # ==> [None, None, None, 4, 300]
        #encoder_input = common_layers.flatten4d3d(encoder_input)
        encoder_input = flatten5d4d(encoder_input)
        print(encoder_input.shape.as_list()) # ==> [None, None, 4, 300]
        target_space = features["target_space_id"]
        print(target_space.shape.as_list()) # ==> []
        # encode_lex
        encoder_output, encoder_decoder_attention_bias = self.encode_lex(encoder_input, target_space, hparams)
        targets = features["targets"]
        print(targets.shape.as_list())
        targets = common_layers.flatten4d3d(targets)
        # decode_lex
        decoder_input, decoder_self_attention_bias = transformer.transformer_prepare_decoder(targets, hparams)
        decoder_output = self.decode(decoder_input, encoder_output, encoder_decoder_attention_bias, decoder_self_attention_bias, hparams)
        return decoder_output

    def encode_lex(self, encoder_input, target_space, hparams):
        '''
        encoder_input: [batch_size, input_len, hidden_dim]
        return: 
            encoder_output: [batch_size, input_len, hidden_dim]
            encoder_decoder_attention_bias: [batch_size, input_len]
        '''
        encoder_output_slices = []
        for i in range(encoder_input.get_shape()[2].value):
            encoder_input_slice = encoder_input[:,:,i,:]

            # bias
            encoder_padding = common_attention.embedding_to_padding(encoder_input_slice)
            print(encoder_padding.shape.as_list()) # ==> [None, None] (None, None, 4)
            ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
            encoder_self_attention_bias = ignore_padding
            encoder_decoder_attention_bias = ignore_padding
            print(ignore_padding.shape.as_list()) # ==> [None, 1, 1, None] (None, 1, 1, None, 4)

            # add target space to encoder input?
            ishape_static = encoder_input_slice.shape.as_list()
            print(ishape_static) # ==> [None, None, 300] (None, None, 4, 300)
            emb_target_space = common_layers.embedding(target_space, 32, ishape_static[-1], name="target_space_embedding")
            print(emb_target_space.shape.as_list()) # ==> [300]
            emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
            print(emb_target_space.shape.as_list()) # ==> [1, 1, 300]
            encoder_input_slice += emb_target_space
            print(encoder_input_slice.shape.as_list()) # ==> [None, None, 300] (None, None, 4, 300)

            # add timing signals to encoder input
            if hparams.pos == "timing":
                encoder_input_slice = common_attention.add_timing_signal_1d(encoder_input_slice)

            # dropout
            encoder_input_slice = tf.nn.dropout(encoder_input_slice, 1.0-hparams.layer_prepostprocess_dropout)

            # encoder
            '''
            multihead_attention(
            query_antecedent: [batch, length_q, channels], -- x, x
            memory_antecedent: [batch, length_m, channels], -- None, encoder_output
            bias: bias tensor, -- encoder_self_attention_bias
            total_key_depth: int, -- hparams.attention_key_channels or hparams.hidden_size
            total_value_depth: int, -- hparams.attention_value_channels or hparams.hidden_size
            output_depth: integer, -- hparams.hidden_size
            num_heads: integer dividing total_key_depth and total_value_depth, -- hparams.num_heads (8)
            dropout_rate: float, -- hparams.attention_dropout
            ...
            cache=None: dict, containing tensors which are the results of previous attentions used for fast decoding, {'k': [batch_size, 0, key_channels], 'v': [batch_size, 0, value_channels], used in decoder self-attention)
            '''
            x = encoder_input_slice
            with tf.variable_scope("encoder"+str(i)):
                # remove pad 
                pad_remover = None
                if hparams.use_pad_remover:
                    pad_remover = expert_utils.PadRemover(common_attention.attention_bias_to_padding(encoder_self_attention_bias))

                # self-attention along the sentence dimension
                for layer in xrange(hparams.num_encoder_layers or hparams.num_hidden_layers):
                    with tf.variable_scope("layer_%d" % layer):
                        with tf.variable_scope("self_attention"):
                            query_antecedent = common_layers.layer_preprocess(x, hparams)
                            y = common_attention.multihead_attention(
                                query_antecedent=query_antecedent,
                                memory_antecedent=None,
                                bias=encoder_self_attention_bias,
                                total_key_depth=hparams.attention_key_channels or hparams.hidden_size,
                                total_value_depth=hparams.attention_value_channels or hparams.hidden_size,
                                output_depth=hparams.hidden_size,
                                num_heads=hparams.num_heads,
                                dropout_rate=hparams.attention_dropout,
                                attention_type=hparams.self_attention_type,
                                max_relative_position=hparams.max_relative_position)
                            x = common_layers.layer_postprocess(x, y, hparams)
                        with tf.variable_scope("ffn"):
                            y = transformer.transformer_ffn_layer(common_layers.layer_preprocess(x, hparams), hparams, pad_remover)
                            x = common_layers.layer_postprocess(x, y, hparams)
                encoder_output_slice = common_layers.layer_preprocess(x, hparams)
                print(encoder_output_slice.shape.as_list()) # ==> [None, None, 300] (None, None, 4, 300)

            encoder_output_slices.append(encoder_output_slice)
        encoder_output = tf.stack(encoder_output_slices, 2)
        print(encoder_output.shape.as_list()) # ==> [None, None, 4, 300]

        # --------

        encoder_output_slices = []
        #hparams2 = copy.deepcopy(hparams)
        #hparams2.hidden_size = hparams.lex_cap
        num_heads = int(hparams.lex_cap/2)
        hparams2 = tf.contrib.training.HParams(
            layer_preprocess_sequence=hparams.layer_preprocess_sequence,
            layer_postprocess_sequence=hparams.layer_postprocess_sequence,
            layer_prepostprocess_dropout=hparams.layer_prepostprocess_dropout,
            norm_type=hparams.norm_type,
            hidden_size=hparams.lex_cap,
            norm_epsilon=hparams.norm_epsilon,
            ffn_layer=hparams.ffn_layer,
            filter_size=hparams.filter_size,
            relu_dropout=hparams.relu_dropout,
            num_heads=num_heads,
            attention_dropout=hparams.attention_dropout,
            parameter_attention_key_channels=hparams.parameter_attention_key_channels,
            parameter_attention_value_channels=hparams.parameter_attention_value_channels)

        for i in range(encoder_output.get_shape()[3].value):
            encoder_input_slice = encoder_output[:,:,:,i]
            #print(encoder_input_slice.shape.as_list()) # ==> [None, None, 4]

            encoder_padding = common_attention.embedding_to_padding(encoder_input_slice)
            ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
            encoder_self_attention_bias = ignore_padding
            #print(encoder_self_attention_bias.shape.as_list()) # ==> [None, 1, 1, None]

            # encoder
            '''
            multihead_attention(
            query_antecedent: [batch, length_q, channels], -- x, x
            memory_antecedent: [batch, length_m, channels], -- None, encoder_output
            bias: bias tensor, -- encoder_self_attention_bias
            total_key_depth: int, -- hparams.attention_key_channels or hparams.hidden_size
            total_value_depth: int, -- hparams.attention_value_channels or hparams.hidden_size
            output_depth: integer, -- hparams.hidden_size
            num_heads: integer dividing total_key_depth and total_value_depth, -- hparams.num_heads (8)
            dropout_rate: float, -- hparams.attention_dropout
            ...
            cache=None: dict, containing tensors which are the results of previous attentions used for fast decoding, {'k': [batch_size, 0, key_channels], 'v': [batch_size, 0, value_channels], used in decoder self-attention)
            '''
            x = encoder_input_slice
            with tf.variable_scope("encoder_extra"+str(i)):
                # remove pad 
                pad_remover = None
                if hparams.use_pad_remover:
                    pad_remover = expert_utils.PadRemover(common_attention.attention_bias_to_padding(encoder_self_attention_bias))

                # self-attention along the lexicon dimension
                with tf.variable_scope("layer_extra"):
                    with tf.variable_scope("self_attention"):
                        #query_antecedent = layer_preprocess2(x, hparams, hparams.lex_cap)
                        query_antecedent = common_layers.layer_preprocess(x, hparams2)

                        y = common_attention.multihead_attention(
                            query_antecedent=query_antecedent,
                            memory_antecedent=None,
                            bias=encoder_self_attention_bias,
                            total_key_depth=hparams.attention_key_channels or hparams.lex_cap,
                            total_value_depth=hparams.attention_value_channels or hparams.lex_cap,
                            output_depth=hparams.lex_cap,
                            num_heads=num_heads,
                            dropout_rate=hparams.attention_dropout,
                            attention_type=hparams.self_attention_type,
                            max_relative_position=hparams.max_relative_position)
                        #x = layer_postprocess2(x, y, hparams, hparams.lex_cap)
                        x = common_layers.layer_postprocess(x, y, hparams2)
                    with tf.variable_scope("ffn"):
                        y = transformer.transformer_ffn_layer(common_layers.layer_preprocess(x, hparams2), hparams2, pad_remover)
                        #x = layer_postprocess2(x, y, hparams, hparams.lex_cap)
                        x = common_layers.layer_postprocess(x, y, hparams2)
                #encoder_output_slice = layer_preprocess2(x, hparams, hparams.lex_cap)
                encoder_output_slice = common_layers.layer_preprocess(x, hparams2)
                #print(encoder_output_slice.shape.as_list()) # ==> [None, None, 4] (None, None, 4, 300)

            encoder_output_slices.append(encoder_output_slice)
        encoder_output = tf.stack(encoder_output_slices, 3)
        print(encoder_output.shape.as_list()) # ==> [None, None, 4, 300]

        # --------

        lex_cap = encoder_output.get_shape()[2].value
        embed_len = encoder_output.get_shape()[3].value
        assert(lex_cap == hparams.lex_cap)
        aggregate_layer = tf.get_variable(
            name="Aggregate",
            shape=[embed_len, embed_len, lex_cap],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        encoder_output = tf.tensordot(encoder_output, aggregate_layer, axes=[[2,3],[1,2]])
        print(encoder_output.shape.as_list()) # ==> [None, None, 300]

        return encoder_output, encoder_decoder_attention_bias
    

