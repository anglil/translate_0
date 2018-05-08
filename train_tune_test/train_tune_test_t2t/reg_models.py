from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys 
import copy
import yaml

from six.moves import xrange

from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.data_generators import problem
import tensorflow as tf

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

def copy_hparams(hparams):
    hparams2 = tf.contrib.training.HParams()
    hparams_dict = hparams.values()
    for key, value in hparams_dict.items():
        hparams2.add_hparam(key, value)
    return hparams2

def flatten5d4d(x):
    '''joining width and height'''
    xshape = tf.shape(x)
    result = tf.reshape(x, [xshape[0], xshape[1] * xshape[2], xshape[3], xshape[4]])
    xshape_static = x.get_shape()
    result.set_shape([xshape_static[0], None, xshape_static[3], xshape_static[4]])
    return result

def lex_aggregate(t, hparams):
    t = common_layers.layer_preprocess(t, hparams) # normalization
    t = tf.transpose(t, perm=[0,1,3,2])
    t = tf.layers.dense(t, 1)#, activation=tf.tanh)
    t = tf.squeeze(t, [3])
    t = tf.layers.dropout(t, 1.0-hparams.layer_prepostprocess_dropout) # dropout/regularization
    return t

def reshape_2d(x_slices):
    shape_static = x_slices.get_shape()
    shape_dynamic = tf.shape(x_slices)
    seq_len = shape_dynamic[1]*shape_static[2].value
    x_slices = tf.reshape(x_slices, [shape_dynamic[0], seq_len, shape_static[3].value])
    return x_slices, shape_dynamic[0], shape_dynamic[1], shape_static[2].value, shape_static[3].value

# --------

def attn_over_sent_and_lex_1d_dec(x, encoder_output, decoder_self_attention_bias, encoder_decoder_attention_bias, hparams):
    '''
    decoder_input: [batch_size, decoder_length, hidden_dim]
    encoder_output: [batch_size, input_length, hidden_dim]
    encoder_decoder_attention_bias: [batch_size, input_length]
    decoder_self_attention_bias: [batch_size, decoder_length]
    '''
    with tf.variable_scope("self_attention"):
        query_antecedent = common_layers.layer_preprocess(x, hparams)
        y = common_attention.multihead_attention(
            query_antecedent=query_antecedent,
            memory_antecedent=None,
            bias=decoder_self_attention_bias,
            total_key_depth=hparams.attention_key_channels or hparams.hidden_size,
            total_value_depth=hparams.attention_value_channels or hparams.hidden_size,
            output_depth=hparams.hidden_size,
            num_heads=hparams.num_heads,
            dropout_rate=hparams.attention_dropout,
            attention_type=hparams.self_attention_type,
            max_relative_position=hparams.max_relative_position)
        x = common_layers.layer_postprocess(x, y, hparams)
    if encoder_output is not None:
        with tf.variable_scope("encdec_attention"):
            query_antecedent = common_layers.layer_preprocess(x, hparams)
            y = common_attention.multihead_attention(
                query_antecedent=query_antecedent,
                memory_antecedent=encoder_output,
                bias=encoder_decoder_attention_bias,
                total_key_depth=hparams.attention_key_channels or hparams.hidden_size,
                total_value_depth=hparams.attention_value_channels or hparams.hidden_size,
                output_depth=hparams.hidden_size,
                num_heads=hparams.num_heads,
                dropout_rate=hparams.attention_dropout)
            x = common_layers.layer_postprocess(x, y, hparams)
    with tf.variable_scope("ffn"):
        x0 = common_layers.layer_preprocess(x, hparams)
        y = transformer.transformer_ffn_layer(x0, hparams)
        x = common_layers.layer_postprocess(x, y, hparams)
    return x

def attn_over_sent_and_lex_2d_dec(x, encoder_output, decoder_self_attention_bias, hparams):
    with tf.variable_scope("self_attention"):
        query_antecedent = common_layers.layer_preprocess(x, hparams)
        y = common_attention.multihead_attention(
            query_antecedent=query_antecedent,
            memory_antecedent=None,
            bias=decoder_self_attention_bias,
            total_key_depth=hparams.attention_key_channels or hparams.hidden_size,
            total_value_depth=hparams.attention_value_channels or hparams.hidden_size,
            output_depth=hparams.hidden_size,
            num_heads=hparams.num_heads,
            dropout_rate=hparams.attention_dropout,
            attention_type=hparams.self_attention_type,
            max_relative_position=hparams.max_relative_position)
        x = common_layers.layer_postprocess(x, y, hparams)
    if encoder_output is not None:
        with tf.variable_scope("encdec_attention"):
            query_antecedent = common_layers.layer_preprocess(x, hparams)
            
            batch_size = tf.shape(encoder_output)[0]
            src_len = tf.shape(encoder_output)[1]
            tgt_len = tf.shape(query_antecedent)[1]
            lex_cap = encoder_output.shape.as_list()[2]
            hid_size = encoder_output.shape.as_list()[3]
    
            query_antecedent = tf.expand_dims(query_antecedent, 2)
            query_antecedent = tf.pad(query_antecedent, [[0,0],[0,0],[0,lex_cap-1],[0,0]])
            query_pad = tf.zeros([batch_size, src_len, lex_cap, hid_size])
            query_antecedent = tf.concat([query_antecedent, query_pad], 1)

            memory_antecedent = encoder_output
            memory_pad = tf.zeros([batch_size, tgt_len, lex_cap, hid_size])
            memory_antecedent = tf.concat([memory_antecedent, memory_pad], 1)

            tf.logging.info("dimension of decoder input at the enc-dec attention layer: {0}".format(query_antecedent.get_shape()))
            tf.logging.info("dimension of encoder output at the enc-dec attention layer: {0}".format(memory_antecedent.get_shape()))

            y = common_attention.multihead_attention_2d(
                query_antecedent=query_antecedent,
                memory_antecedent=memory_antecedent,
                total_key_depth=hparams.attention_key_channels or hparams.hidden_size,
                total_value_depth=hparams.attention_value_channels or hparams.hidden_size,
                output_depth=hparams.hidden_size,
                num_heads=hparams.num_heads,
                attention_type="masked_local_attention_2d",
                query_shape=(4,4),
                memory_flange=(4,4))

            tf.logging.info("dimension of enc-dec output: {0}".format(y.get_shape()))
            y = y[:,:,0,:]
            y = y[:,:tgt_len,:]

            x = common_layers.layer_postprocess(x, y, hparams)
    with tf.variable_scope("ffn"):
        x0 = common_layers.layer_preprocess(x, hparams)
        y = transformer.transformer_ffn_layer(x0, hparams)
        x = common_layers.layer_postprocess(x, y, hparams)
    return x

# --------

def attn_over_sent(x, pad_remover, encoder_self_attention_bias, hparams):
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
        x0 = common_layers.layer_preprocess(x, hparams)
        y = transformer.transformer_ffn_layer(x0, hparams, pad_remover)
        x = common_layers.layer_postprocess(x, y, hparams)
    return x

def attn_over_sent_and_lex_1d(x_slices, pad_remover_combined, encoder_self_attention_bias, hparams):
    x_slices, batch_size, sent_len, lex_cap, hid_dim = reshape_2d(x_slices)
    x_slices = attn_over_sent(x_slices, pad_remover_combined, encoder_self_attention_bias, hparams)
    x_slices = tf.reshape(x_slices, [batch_size, sent_len, lex_cap, hid_dim])
    return x_slices

def attn_over_sent_and_lex_2d(x_slices, pad_remover_combined, hparams):
    with tf.variable_scope("self_attention"):
        query_antecedent = common_layers.layer_preprocess(x_slices, hparams)
        y_slices = common_attention.multihead_attention_2d(
            query_antecedent=query_antecedent,
            memory_antecedent=None,
            total_key_depth=hparams.attention_key_channels or hparams.hidden_size,
            total_value_depth=hparams.attention_value_channels or hparams.hidden_size,
            output_depth=hparams.hidden_size,
            num_heads=hparams.num_heads,
            query_shape=(4,4),
            memory_flange=(4,4))
        x_slices = common_layers.layer_postprocess(x_slices, y_slices, hparams)
    with tf.variable_scope("ffn"):
        x0_slices = common_layers.layer_preprocess(x_slices, hparams)
        x0_slices, batch_size, sent_len, lex_cap, hid_dim = reshape_2d(x0_slices)
        y_slices = transformer.transformer_ffn_layer(x0_slices, hparams, pad_remover_combined)
        y_slices = tf.reshape(y_slices, [batch_size, sent_len, lex_cap, hid_dim])
        x_slices = common_layers.layer_postprocess(x_slices, y_slices, hparams)
    return x_slices

# --------

def transformer_prepare_encoder2(encoder_input, target_space, hparams, emb_name):
    '''the same as the existing module except for being able to name the embedding'''
    # compute bias
    ishape_static = encoder_input.shape.as_list()
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
    encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding
    if hparams.proximity_bias:
        encoder_self_attention_bias += common_attention.attention_bias_proximal(tf.shape(encoder_input)[1])

    # Append target_space_id embedding to encoder_input
    id_values = [value for attr, value in vars(problem.SpaceID).items() if not attr.startswith("__")]
    id_cur = int(max(id_values)+1)
    emb_target_space = common_layers.embedding(target_space, id_cur, ishape_static[-1], name=emb_name)
    emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
    encoder_input += emb_target_space

    # position embedding
    if hparams.pos == "timing":
        encoder_input = common_attention.add_timing_signal_1d(encoder_input)
    return encoder_input, encoder_self_attention_bias, encoder_decoder_attention_bias

def parallel_transformer_prepare_encoder(encoder_input, target_space, hparams):
    encoder_input_slices = []
    encoder_self_attention_bias_slices = []
    encoder_decoder_attention_bias_slices = []
    for i in range(encoder_input.get_shape()[2].value):
        encoder_input_slice = encoder_input[:,:,i,:]
        encoder_input_slice, encoder_self_attention_bias_slice, encoder_decoder_attention_bias_slice = transformer_prepare_encoder2(encoder_input_slice, target_space, hparams, "target_space_embedding"+str(i))
        encoder_input_slices.append(encoder_input_slice)
        encoder_self_attention_bias_slices.append(encoder_self_attention_bias_slice)
        encoder_decoder_attention_bias_slices.append(encoder_decoder_attention_bias_slice)
    encoder_input = tf.stack(encoder_input_slices, 2)
    #encoder_decoder_attention_bias = tf.stack(encoder_decoder_attention_bias_slices)
    #encoder_decoder_attention_bias = tf.reduce_mean(encoder_decoder_attention_bias, 0)
    return encoder_input, encoder_self_attention_bias_slices, encoder_decoder_attention_bias_slices

def get_pad_remover(hparams, encoder_self_attention_bias_slices, is_combined=False):
    '''
    is_combined: whether the multiple translation options are combined or not
    '''
    pad_remover = None
    if not is_combined:
        #encoder_self_attention_bias = tf.reduce_mean(tf.stack(encoder_self_attention_bias_slices), 0)
        encoder_self_attention_bias = encoder_self_attention_bias_slices[0]
        if hparams.use_pad_remover:
            padding = common_attention.attention_bias_to_padding(encoder_self_attention_bias)
            pad_remover = expert_utils.PadRemover(padding)
    else:
        encoder_self_attention_bias = tf.concat(encoder_self_attention_bias_slices, 3)
        if hparams.use_pad_remover:
            padding = common_attention.attention_bias_to_padding(encoder_self_attention_bias)
            pad_remover = expert_utils.PadRemover(padding)
    return (pad_remover, encoder_self_attention_bias)

class encode_fn2:
    def all1d(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        x = encoder_input
        with tf.variable_scope(name):
            pad_bias_combined = get_pad_remover(hparams, encoder_self_attention_bias_slices, is_combined=True)
            for layer in xrange(hparams.num_encoder_layers or hparams.num_hidden_layers):
                with tf.variable_scope("layer_%d" % layer):
                    x = attn_over_sent_and_lex_1d(x, pad_bias_combined[0], pad_bias_combined[1], hparams)
            return common_layers.layer_preprocess(x, hparams)
    def all1d2d(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        return encode_fn2.all1d(encoder_input, encoder_self_attention_bias_slices, hparams, name)

    def all2d(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        x = encoder_input
        with tf.variable_scope(name):
            pad_bias_combined = get_pad_remover(hparams, encoder_self_attention_bias_slices, is_combined=True)
            for layer in xrange(hparams.num_encoder_layers or hparams.num_hidden_layers):
                with tf.variable_scope("layer_%d" % layer):
                    x = attn_over_sent_and_lex_2d(x, pad_bias_combined[0], hparams)
            return common_layers.layer_preprocess(x, hparams)
    def all2d2d(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        return encode_fn2.all2d(encoder_input, encoder_self_attention_bias_slices, hparams, name)

    def allregular(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        x = encoder_input
        with tf.variable_scope(name):
            x_slices = []
            for i in range(x.get_shape()[2].value):
                with tf.variable_scope("encoder_lex"+str(i)):
                    x_slice = x[:,:,i,:]
                    pad_bias = get_pad_remover(hparams, [encoder_self_attention_bias_slices[i]])
                    for layer in xrange(hparams.num_encoder_layers or hparams.num_hidden_layers):
                        with tf.variable_scope("layer_%d" % layer):
                            x_slice = attn_over_sent(x_slice, pad_bias[0], pad_bias[1], hparams)
                    x_slices.append(x_slice)
            x = tf.stack(x_slices, 2)
            return common_layers.layer_preprocess(x, hparams)
    def allregular2d(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        return encode_fn2.allregular(encoder_input, encoder_self_attention_bias_slices, hparams, name)

class encode_fn:
    def all1daggregate(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        x = encoder_input
        with tf.variable_scope(name):
            pad_bias_combined = get_pad_remover(hparams, encoder_self_attention_bias_slices, is_combined=True)
            for layer in xrange(hparams.num_encoder_layers or hparams.num_hidden_layers):
                with tf.variable_scope("layer_%d" % layer):
                    x = attn_over_sent_and_lex_1d(x, pad_bias_combined[0], pad_bias_combined[1], hparams)
            x = lex_aggregate(x, hparams)
            return common_layers.layer_preprocess(x, hparams)

    def all2daggregate(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        x = encoder_input
        with tf.variable_scope(name):
            pad_bias_combined = get_pad_remover(hparams, encoder_self_attention_bias_slices, is_combined=True)
            for layer in xrange(hparams.num_encoder_layers or hparams.num_hidden_layers):
                with tf.variable_scope("layer_%d" % layer):
                    x = attn_over_sent_and_lex_2d(x, pad_bias_combined[0], hparams)
            x = lex_aggregate(x, hparams)
            return common_layers.layer_preprocess(x, hparams)

    def beforeaggregate(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        x = encoder_input
        with tf.variable_scope(name):
            pad_bias = get_pad_remover(hparams, encoder_self_attention_bias_slices)
            x = lex_aggregate(x, hparams)
            for layer in xrange(hparams.num_encoder_layers or hparams.num_hidden_layers):
                with tf.variable_scope("layer_%d" % layer):
                    x = attn_over_sent(x, pad_bias[0], pad_bias[1], hparams)
            return common_layers.layer_preprocess(x, hparams)

    def afteraggregate(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        x = encoder_input
        with tf.variable_scope(name):
            x_slices = []
            for i in range(x.get_shape()[2].value):
                with tf.variable_scope("encoder_lex"+str(i)):
                    x_slice = x[:,:,i,:]
                    pad_bias = get_pad_remover(hparams, [encoder_self_attention_bias_slices[i]])
                    for layer in xrange(hparams.num_encoder_layers or hparams.num_hidden_layers):
                        with tf.variable_scope("layer_%d" % layer):
                            x_slice = attn_over_sent(x_slice, pad_bias[0], pad_bias[1], hparams)
                    x_slices.append(x_slice)
            x = tf.stack(x_slices, 2)
            x = lex_aggregate(x, hparams)
            return common_layers.layer_preprocess(x, hparams)

    def before1daggregate(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        x = encoder_input
        with tf.variable_scope(name):
            pad_bias = get_pad_remover(hparams, encoder_self_attention_bias_slices)
            pad_bias_combined = get_pad_remover(hparams, encoder_self_attention_bias_slices, is_combined=True)
            x = attn_over_sent_and_lex_1d(x, pad_bias_combined[0], pad_bias_combined[1], hparams)
            x = lex_aggregate(x, hparams)
            for layer in xrange(hparams.num_encoder_layers or hparams.num_hidden_layers):
                with tf.variable_scope("layer_%d" % layer):
                    x = attn_over_sent(x, pad_bias[0], pad_bias[1], hparams)
            return common_layers.layer_preprocess(x, hparams)

    def after1daggregate(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        x = encoder_input
        with tf.variable_scope(name):
            pad_bias_combined = get_pad_remover(hparams, encoder_self_attention_bias_slices, is_combined=True)
            x_slices = []
            for i in range(x.get_shape()[2].value):
                with tf.variable_scope("encoder_lex"+str(i)):
                    x_slice = x[:,:,i,:]
                    pad_bias = get_pad_remover(hparams, [encoder_self_attention_bias_slices[i]])
                    for layer in xrange(hparams.num_encoder_layers or hparams.num_hidden_layers):
                        with tf.variable_scope("layer_%d" % layer):
                            x_slice = attn_over_sent(x_slice, pad_bias[0], pad_bias[1], hparams)
                    x_slices.append(x_slice)
            x = tf.stack(x_slices, 2)
            x = attn_over_sent_and_lex_1d(x, pad_bias_combined[0], pad_bias_combined[1], hparams)
            x = lex_aggregate(x, hparams)
            return common_layers.layer_preprocess(x, hparams)

    def before2daggregate(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        x = encoder_input
        with tf.variable_scope(name):
            pad_bias = get_pad_remover(hparams, encoder_self_attention_bias_slices)
            pad_bias_combined = get_pad_remover(hparams, encoder_self_attention_bias_slices, is_combined=True)
            x = attn_over_sent_and_lex_2d(x, pad_bias_combined[0], hparams)
            x = lex_aggregate(x, hparams)
            for layer in xrange(hparams.num_encoder_layers or hparams.num_hidden_layers):
                with tf.variable_scope("layer_%d" % layer):
                    x = attn_over_sent(x, pad_bias[0], pad_bias[1], hparams)
            return common_layers.layer_preprocess(x, hparams)

    def after2daggregate(encoder_input, encoder_self_attention_bias_slices, hparams, name):
        x = encoder_input
        with tf.variable_scope(name):
            pad_bias_combined = get_pad_remover(hparams, encoder_self_attention_bias_slices, is_combined=True)
            x_slices = []
            for i in range(x.get_shape()[2].value):
                with tf.variable_scope("encoder_lex"+str(i)):
                    x_slice = x[:,:,i,:]
                    pad_bias = get_pad_remover(hparams, [encoder_self_attention_bias_slices[i]])
                    for layer in xrange(hparams.num_encoder_layers or hparams.num_hidden_layers):
                        with tf.variable_scope("layer_%d" % layer):
                            x_slice = attn_over_sent(x_slice, pad_bias[0], pad_bias[1], hparams)
                    x_slices.append(x_slice)
            x = tf.stack(x_slices, 2)
            x = attn_over_sent_and_lex_2d(x, pad_bias_combined[0], hparams)
            x = lex_aggregate(x, hparams)
            return common_layers.layer_preprocess(x, hparams)

class decode_fn:
    def dec1d(decoder_input, encoder_output, decoder_self_attention_bias, encoder_decoder_attention_bias_slices, hparams, name):
        # flatten encoder output
        encoder_output, batch_size, sent_len, lex_cap, hid_dim = reshape_2d(encoder_output)
        encoder_decoder_attention_bias = tf.concat(encoder_decoder_attention_bias_slices, 3)
        x = decoder_input
        with tf.variable_scope(name):
            for layer in xrange(hparams.num_decoder_layers or hparams.num_hidden_layers):
                with tf.variable_scope("layer_%d" % layer):
                    x = attn_over_sent_and_lex_1d_dec(x, encoder_output, decoder_self_attention_bias, encoder_decoder_attention_bias, hparams)
            return common_layers.layer_preprocess(x, hparams)

    def dec2d(decoder_input, encoder_output, decoder_self_attention_bias, _, hparams, name):
        x = decoder_input
        with tf.variable_scope(name):
            for layer in xrange(hparams.num_decoder_layers or hparams.num_hidden_layers):
                with tf.variable_scope("layer_%d" % layer):
                    x = attn_over_sent_and_lex_2d_dec(x, encoder_output, decoder_self_attention_bias, hparams)
            return common_layers.layer_preprocess(x, hparams)

@registry.register_model
class TransformerLex2(transformer.Transformer):
    def encode(self, encoder_input, target_space, hparams):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(dir_path, "config.yml")
        config = yaml.load(open(config_file))
        enc_name = config["model_params"].split('_')[0][3:]

        encoder_input, encoder_self_attention_bias_slices, encoder_decoder_attention_bias_slices = parallel_transformer_prepare_encoder(encoder_input, target_space, hparams)
        encoder_input = tf.nn.dropout(encoder_input, 1.0-hparams.layer_prepostprocess_dropout)
        encoder_output = getattr(encode_fn2, enc_name)(encoder_input, encoder_self_attention_bias_slices, hparams, "encoder")
        return encoder_output, encoder_decoder_attention_bias_slices

    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias_slices, hparams):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(dir_path, "config.yml")
        config = yaml.load(open(config_file))
        enc_name = config["model_params"].split('_')[0][3:]
        dec_name = "dec1d"
        if enc_name.endswith("2d") and enc_name != "all2d":
            dec_name = "dec2d"

        decoder_input, decoder_self_attention_bias = transformer.transformer_prepare_decoder(decoder_input, hparams)
        decoder_input = tf.nn.dropout(decoder_input, 1.0-hparams.layer_prepostprocess_dropout)
        decoder_output = getattr(decode_fn, dec_name)(decoder_input, encoder_output, decoder_self_attention_bias, encoder_decoder_attention_bias_slices, hparams, "decoder")
        return decoder_output

    def model_fn_body(self, features):
        hparams = self._hparams

        # encode_lex
        encoder_input = features["inputs"]
        target_space = features["target_space_id"]
        encoder_input = flatten5d4d(encoder_input)
        encoder_output, encoder_decoder_attention_bias_slices = self.encode(encoder_input, target_space, hparams)

        # decode_lex
        targets = features["targets"]
        targets = common_layers.flatten4d3d(targets)
        decoder_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias_slices, hparams)

        return tf.expand_dims(decoder_output, axis=2)


@registry.register_model
class TransformerLex(transformer.Transformer):
    '''
    inherits Transformer which inherits t2t_model.T2TModel
    '''
    def encode(self, encoder_input, target_space, hparams):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(dir_path, "config.yml")
        config = yaml.load(open(config_file))
        enc_name = config["model_params"].split('_')[0][3:]

        if enc_name == "simple":
            encoder_input, encoder_self_attention_bias, encoder_decoder_attention_bias = transformer.transformer_prepare_encoder(encoder_input, target_space, hparams)
            encoder_input = tf.nn.dropout(encoder_input, 1.0-hparams.layer_prepostprocess_dropout)
            encoder_output = transformer.transformer_encoder(encoder_input, encoder_self_attention_bias, hparams)
        else:
            encoder_input, encoder_self_attention_bias_slices, encoder_decoder_attention_bias_slices = parallel_transformer_prepare_encoder(encoder_input, target_space, hparams)
            encoder_input = tf.nn.dropout(encoder_input, 1.0-hparams.layer_prepostprocess_dropout)
            encoder_output = getattr(encode_fn, enc_name)(encoder_input, encoder_self_attention_bias_slices, hparams, "encoder")
            encoder_decoder_attention_bias = tf.stack(encoder_decoder_attention_bias_slices)
            encoder_decoder_attention_bias = tf.reduce_mean(encoder_decoder_attention_bias, 0)
        return encoder_output, encoder_decoder_attention_bias

    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias, hparams):
        decoder_input, decoder_self_attention_bias = transformer.transformer_prepare_decoder(decoder_input, hparams)
        decoder_input = tf.nn.dropout(decoder_input, 1.0-hparams.layer_prepostprocess_dropout)
        decoder_output = transformer.transformer_decoder(decoder_input, encoder_output, decoder_self_attention_bias, encoder_decoder_attention_bias, hparams, cache=None)
        return decoder_output

    def model_fn_body(self, features):
        hparams = self._hparams

        # encode_lex
        encoder_input = features["inputs"]
        target_space = features["target_space_id"]
        if len(encoder_input.shape.as_list()) == 5:
            encoder_input = flatten5d4d(encoder_input)
        elif len(encoder_input.shape.as_list()) == 4:
            encoder_input = common_layers.flatten4d3d(encoder_input)
        else:
            raise ValueError("encoder_input rank error!")
        encoder_output, encoder_decoder_attention_bias = self.encode(encoder_input, target_space, hparams)

        # decode_lex
        targets = features["targets"]
        targets = common_layers.flatten4d3d(targets)
        decoder_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias, hparams)

        return tf.expand_dims(decoder_output, axis=2)


