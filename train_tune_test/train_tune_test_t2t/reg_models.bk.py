# after simple
@registry.register_model
class TransformerLexaftersimple(TransformerLexbeforesimple):
    def encode(self, encoder_input, target_space, hparams):
        encoder_output, _, encoder_decoder_attention_bias = parallel_enc(encoder_input, target_space, hparams)
        encoder_output = lex_aggregate(encoder_output)
        encoder_decoder_attention_bias = tf.reduce_mean(encoder_decoder_attention_bias, 0)
        return encoder_output, encoder_decoder_attention_bias
 
# before 1d
@registry.register_model
class TransformerLexbefore1d(TransformerLexbeforesimple):
    def encode(self, encoder_input, target_space, hparams):
        encoder_input, encoder_self_attention_bias_slices, encoder_decoder_attention_bia = parallel_transformer_prepare_encoder(encoder_input, target_space, hparams)
        encoder_input = attn_over_sent_and_lex(encoder_input, encoder_self_attention_bias_slices, hparams, "encoder_extra_1d")       
        encoder_output, encoder_decoder_attention_bias = TransformerLexbeforesimple.encode(self, encoder_input, target_space, hparams)
        return encoder_output, encoder_decoder_attention_bias

# after 1d
@registry.register_model
class TransformerLexafter1d(TransformerLexbeforesimple):
    def encode(self, encoder_input, target_space, hparams):
        encoder_output, _, encoder_decoder_attention_bias = parallel_enc(encoder_input, target_space, hparams)
        encoder_output = attn_over_sent_and_lex(encoder_output, target_space, hparams)
        encoder_output = lex_aggregate(encoder_output)
        encoder_decoder_attention_bias = tf.reduce_mean(encoder_decoder_attention_bias, 0)
        return encoder_output, encoder_decoder_attention_bias

# before simple tanh
@registry.register_model
class TransformerLexbeforesimpletanh(TransformerLexbeforesimple):
    def encode(self, encoder_input, target_space, hparams):
        encoder_input = lex_aggregate_tanh(encoder_input, hparams)
        encoder_input, encoder_self_attention_bias, encoder_decoder_attention_bias = transformer.transformer_prepare_encoder(encoder_input, target_space, hparams)
        encoder_input = tf.nn.dropout(encoder_input, 1.0-hparams.layer_prepostprocess_dropout)
        encoder_output = transformer.transformer_encoder(encoder_input, encoder_self_attention_bias, hparams)
        return encoder_output, encoder_decoder_attention_bias
   
# after simple tanh
@registry.register_model
class TransformerLexaftersimpletanh(TransformerLexbeforesimple):
    def encode(self, encoder_input, target_space, hparams):
        encoder_output, _, encoder_decoder_attention_bias = parallel_enc(encoder_input, target_space, hparams)
        encoder_output = lex_aggregate_tanh(encoder_output, hparams)
        encoder_decoder_attention_bias = tf.reduce_mean(encoder_decoder_attention_bias, 0)
        return encoder_output, encoder_decoder_attention_bias

# before 2d
@registry.register_model
class TransformerLexbefore2d(TransformerLexbeforesimple):
    def encode(self, encoder_input, target_space, hparams):
        encoder_input, encoder_self_attention_bias_slices, encoder_decoder_attention_bias = parallel_transformer_prepare_encoder(encoder_input, target_space, hparams)
        encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.layer_prepostprocess_dropout)
        encoder_input = attn_over_sent_and_lex_2d(encoder_input, encoder_self_attention_bias_slices, hparams)
        encoder_input = lex_aggregate(encoder_input)
        encoder_input = tf.nn.dropout(encoder_input, 1.0-hparams.layer_prepostprocess_dropout)
        encoder_self_attention_bias = tf.reduce_mean(tf.stack(encoder_self_attention_bias_slices), 0)
        encoder_output = transformer.transformer_encoder(encoder_input, encoder_self_attention_bias, hparams)
        return encoder_output, encoder_decoder_attention_bias

# after 2d
@registry.register_model
class TransformerLexafter2d(TransformerLexbeforesimple):
    def encode(self, encoder_input, target_space, hparams):
        encoder_output, encoder_self_attention_bias_slices, encoder_decoder_attention_bias = parallel_enc(encoder_input, target_space,hparams)
        encoder_output = attn_over_sent_and_lex_2d(encoder_output, encoder_self_attention_bias_slices, hparams)
        encoder_output = lex_aggregate(encoder_output)
        encoder_decoder_attention_bias = tf.reduce_mean(encoder_decoder_attention_bias, 0)
        return encoder_output, encoder_decoder_attention_bias

@registry.register_model
class TransformerLexbefore2d(TransformerLexbeforesimple):
    def encode(self, encoder_input, target_space, hparams):
        return encoder_output, encoder_decoder_attention_bias

    def encode_lex(self, encoder_input, target_space, hparams):
        '''
        encoder_input: [batch_size, input_len, hidden_size]
        return: 
            encoder_output: [batch_size, input_len, hidden_size]
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
            print(ishape_static) # ==> [None, None, 1024] (None, None, 4, 1024)
            emb_target_space = common_layers.embedding(target_space, 32, ishape_static[-1], name="target_space_embedding")
            print(emb_target_space.shape.as_list()) # ==> [1024]
            emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
            print(emb_target_space.shape.as_list()) # ==> [1, 1, 1024]
            encoder_input_slice += emb_target_space
            print(encoder_input_slice.shape.as_list()) # ==> [None, None, 1024] (None, None, 4, 1024)

            # add timing signals to encoder input
            if hparams.pos == "timing":
                encoder_input_slice = common_attention.add_timing_signal_1d(encoder_input_slice)

            # dropout
            encoder_input_slice = tf.nn.dropout(encoder_input_slice, 1.0-hparams.layer_prepostprocess_dropout)
            encoder_output_slices.append(encoder_input_slice)
        encoder_output = tf.stack(encoder_output_slices, 2)
        print(encoder_output.shape.as_list()) # ==> [None, None, 4, 1024]
            
        # --------
        print("--------")

        s_tmp = encoder_output.get_shape()
        s_tmp2 = tf.shape(encoder_output)
        d_tmp = s_tmp2[1]*s_tmp[2].value
        #num_heads = 8
        #hparams2 = tf.contrib.training.HParams(
        #    layer_preprocess_sequence=hparams.layer_preprocess_sequence,
        #    layer_postprocess_sequence=hparams.layer_postprocess_sequence,
        #    layer_prepostprocess_dropout=hparams.layer_prepostprocess_dropout,
        #    norm_type=hparams.norm_type,
        #    hidden_size=d_tmp,
        #    norm_epsilon=hparams.norm_epsilon,
        #    ffn_layer=hparams.ffn_layer,
        #    filter_size=hparams.filter_size,
        #    relu_dropout=hparams.relu_dropout,
        #    num_heads=num_heads,
        #    attention_dropout=hparams.attention_dropout,
        #    parameter_attention_key_channels=hparams.parameter_attention_key_channels,
        #    parameter_attention_value_channels=hparams.parameter_attention_value_channels)

        encoder_output = tf.reshape(encoder_output, [s_tmp2[0],d_tmp,s_tmp[3].value])
        tf.logging.info(encoder_output.shape.as_list())
        encoder_padding = tf.layers.dense(encoder_output, int(s_tmp[3].value/s_tmp[2].value))
        tf.logging.info(encoder_padding.shape.as_list())
        encoder_padding = common_attention.embedding_to_padding(encoder_padding)
        tf.logging.info(encoder_padding.shape.as_list())
        ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
        encoder_self_attention_bias = ignore_padding
        tf.logging.info(encoder_self_attention_bias.shape.as_list()) # ==> [None, 1, 1, None]

        # encoder
        x = encoder_output
        with tf.variable_scope("encoder_extra"):
            # remove pad 
            pad_remover = None
            if hparams.use_pad_remover:
                pad_remover = expert_utils.PadRemover(common_attention.attention_bias_to_padding(encoder_self_attention_bias))

            # self-attention along the lexicon dimension
            with tf.variable_scope("layer_extra"):
                with tf.variable_scope("self_attention"):
                    query_antecedent = common_layers.layer_preprocess(x, hparams)
                    print(query_antecedent.shape.as_list()) # ==> [batch, len_q, hid_dim]

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
            tf.logging.info(encoder_output.shape.as_list())
        encoder_output = tf.reshape(encoder_output, [s_tmp2[0], s_tmp2[1], s_tmp[2].value, s_tmp[3].value])
        tf.logging.info(encoder_output.shape.as_list())

        # --------
        print("--------")

        #lex_cap = encoder_output.get_shape()[2].value
        #embed_len = encoder_output.get_shape()[3].value
        #assert(lex_cap == hparams.lex_cap)

        #aggregate_layer = tf.get_variable(
        #    name="Aggregate",
        #    shape=[embed_len, embed_len, lex_cap],
        #    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        #encoder_output = tf.tensordot(encoder_output, aggregate_layer, axes=[[2,3],[1,2]])
        #aggregate_layer = tf.get_variable(
        #    name="Aggregate",
        #    shape=[lex_cap],
        #    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        #encoder_output = tf.tensordot(aggregate_layer, encoder_output, axes=[[0],[2]])

        encoder_output = tf.transpose(encoder_output, perm=[0,1,3,2])
        encoder_output = tf.layers.dense(encoder_output, 1)
        encoder_output = tf.squeeze(encoder_output, [3])

        print(encoder_output.shape.as_list()) # ==> [None, None, 1024]

        # --------
        print("--------")

        encoder_padding = common_attention.embedding_to_padding(encoder_output)
        ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
        encoder_self_attention_bias = ignore_padding

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
            print(encoder_output.shape.as_list()) # ==> [None, None, 1024] (None, None, 4, 1024)
        return encoder_output, encoder_decoder_attention_bias
    


@registry.register_model
class TransformerLexafter(transformer.Transformer):
    '''
    inherits Transformer which inherits t2t_model.T2TModel
    '''
    def model_fn_body(self, features):
        hparams = self._hparams
        
        encoder_input = features["inputs"]
        print(encoder_input.shape.as_list()) # ==> [None, None, None, 4, 1024]
        #encoder_input = common_layers.flatten4d3d(encoder_input)
        encoder_input = flatten5d4d(encoder_input)
        print(encoder_input.shape.as_list()) # ==> [None, None, 4, 1024]
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
        encoder_input: [batch_size, input_len, hidden_size]
        return: 
            encoder_output: [batch_size, input_len, hidden_size]
            encoder_decoder_attention_bias: [batch_size, input_len]
        '''
        encoder_output_slices = []
        encoder_decoder_attention_bias_slices = []
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
            print(ishape_static) # ==> [None, None, 1024] (None, None, 4, 1024)
            emb_target_space = common_layers.embedding(target_space, 32, ishape_static[-1], name="target_space_embedding")
            print(emb_target_space.shape.as_list()) # ==> [1024]
            emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
            print(emb_target_space.shape.as_list()) # ==> [1, 1, 1024]
            encoder_input_slice += emb_target_space
            print(encoder_input_slice.shape.as_list()) # ==> [None, None, 1024] (None, None, 4, 1024)

            # add timing signals to encoder input
            if hparams.pos == "timing":
                encoder_input_slice = common_attention.add_timing_signal_1d(encoder_input_slice)

            # dropout
            encoder_input_slice = tf.nn.dropout(encoder_input_slice, 1.0-hparams.layer_prepostprocess_dropout)

            # --------
            print("--------")

            # encoder

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
                print(encoder_output_slice.shape.as_list()) # ==> [None, None, 1024] (None, None, 4, 1024)

            encoder_output_slices.append(encoder_output_slice)
            encoder_decoder_attention_bias_slices.append(encoder_decoder_attention_bias)
        encoder_output = tf.stack(encoder_output_slices, 2)
        encoder_decoder_attention_bias = tf.stack(encoder_decoder_attention_bias_slices, 4)
        print(encoder_output.shape.as_list()) # ==> [None, None, 4, 1024]

        # --------
        encoder_output = tf.transpose(encoder_output, perm=[0,1,3,2])
        encoder_output = tf.layers.dense(encoder_output, 1)
        encoder_output = tf.squeeze(encoder_output, [3])
       
        encoder_decoder_attention_bias = tf.layers.dense(encoder_decoder_attention_bias, 1)
        encoder_decoder_attention_bias = tf.squeeze(encoder_decoder_attention_bias, [4])
        return encoder_output, encoder_decoder_attention_bias
  


@registry.register_model
class TransformerLexsimple(transformer.Transformer):
    '''
    inherits Transformer which inherits t2t_model.T2TModel
    '''
    def model_fn_body(self, features):
        hparams = self._hparams
        
        encoder_input = features["inputs"]
        print(encoder_input.shape.as_list()) # ==> [None, None, None, 4, 1024]
        encoder_input = flatten5d4d(encoder_input)
        print(encoder_input.shape.as_list()) # ==> [None, None, 4, 1024]
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
        encoder_input: [batch_size, input_len, hidden_size]
        return: 
            encoder_output: [batch_size, input_len, hidden_size]
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
            print(ishape_static) # ==> [None, None, 1024] (None, None, 4, 1024)
            emb_target_space = common_layers.embedding(target_space, 32, ishape_static[-1], name="target_space_embedding")
            print(emb_target_space.shape.as_list()) # ==> [1024]
            emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
            print(emb_target_space.shape.as_list()) # ==> [1, 1, 1024]
            encoder_input_slice += emb_target_space
            print(encoder_input_slice.shape.as_list()) # ==> [None, None, 1024] (None, None, 4, 1024)

            # add timing signals to encoder input
            if hparams.pos == "timing":
                encoder_input_slice = common_attention.add_timing_signal_1d(encoder_input_slice)

            # dropout
            encoder_input_slice = tf.nn.dropout(encoder_input_slice, 1.0-hparams.layer_prepostprocess_dropout)
            encoder_output_slices.append(encoder_input_slice)
        encoder_output = tf.stack(encoder_output_slices, 2)
            
        # --------

        encoder_output = tf.transpose(encoder_output, perm=[0,1,3,2])
        encoder_output = tf.layers.dense(encoder_output, 1)
        encoder_output = tf.squeeze(encoder_output, [3])

        # --------

        encoder_padding = common_attention.embedding_to_padding(encoder_output)
        print(encoder_padding.shape.as_list()) # ==> [None, None] (None, None, 4)
        ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
        encoder_self_attention_bias = ignore_padding
        encoder_decoder_attention_bias = ignore_padding

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
            print(encoder_output.shape.as_list()) # ==> [None, None, 1024] (None, None, 4, 1024)
        return encoder_output, encoder_decoder_attention_bias


@registry.register_model
class TransformerLexbefore2d(transformer.Transformer):
    '''
    inherits Transformer which inherits t2t_model.T2TModel
    '''
    def model_fn_body(self, features):
        hparams = self._hparams
        
        encoder_input = features["inputs"]
        print(encoder_input.shape.as_list()) # ==> [None, None, None, 4, 1024]
        encoder_input = flatten5d4d(encoder_input)
        print(encoder_input.shape.as_list()) # ==> [None, None, 4, 1024]
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
        encoder_input: [batch_size, input_len, hidden_size]
        return: 
            encoder_output: [batch_size, input_len, hidden_size]
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
            print(ishape_static) # ==> [None, None, 1024] (None, None, 4, 1024)
            emb_target_space = common_layers.embedding(target_space, 32, ishape_static[-1], name="target_space_embedding")
            print(emb_target_space.shape.as_list()) # ==> [1024]
            emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
            print(emb_target_space.shape.as_list()) # ==> [1, 1, 1024]
            encoder_input_slice += emb_target_space
            print(encoder_input_slice.shape.as_list()) # ==> [None, None, 1024] (None, None, 4, 1024)

            # add timing signals to encoder input (positional encoding)
            if hparams.pos == "timing":
                encoder_input_slice = common_attention.add_timing_signal_1d(encoder_input_slice)

            # dropout
            encoder_input_slice = tf.nn.dropout(encoder_input_slice, 1.0-hparams.layer_prepostprocess_dropout)
            encoder_output_slices.append(encoder_input_slice)
        encoder_output = tf.stack(encoder_output_slices, 2)
            
        # -------- a layer of 2d attention over lexicon

        encoder_input_slice = tf.transpose(encoder_output, perm=[0,2,1,3])
        print(encoder_input_slice.shape.as_list()) # ==> [None, None, 4, 1024]
        encoder_padding = common_attention.embedding_to_padding(encoder_input_slice)
        ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
        encoder_self_attention_bias = ignore_padding
        print(encoder_self_attention_bias.shape.as_list()) # ==> [None, 1, 1, None]

        # encoder
        x = encoder_input_slice
        with tf.variable_scope("encoder_extra"):
            # remove pad 
            pad_remover = None
            if hparams.use_pad_remover:
                pad_remover = expert_utils.PadRemover(common_attention.attention_bias_to_padding(encoder_self_attention_bias))

            # self-attention along the lexicon dimension
            with tf.variable_scope("layer_extra"):
                with tf.variable_scope("self_attention"):
                    #query_antecedent = layer_preprocess2(x, hparams, hparams.lex_cap)
                    query_antecedent = common_layers.layer_preprocess(x, hparams)
                    print(query_antecedent.shape.as_list())

                    y = common_attention.multihead_attention_2d(
                        query_antecedent=query_antecedent,
                        memory_antecedent=None,
                        total_key_depth=hparams.attention_key_channels or hparams.hidden_size,
                        total_value_depth=hparams.attention_value_channels or hparams.hidden_size,
                        output_depth=hparams.hidden_size,
                        num_heads=hparams.num_heads,
                        query_shape=(4,4),
                        memory_flange=(4,4))
                    print(y.shape.as_list())
                    x = common_layers.layer_postprocess(x, y, hparams)
                    print(x.shape.as_list())
                with tf.variable_scope("ffn"):
                    tmp = common_layers.layer_preprocess(x, hparams)
                    print(tmp.shape.as_list())
                    tmp_d = tf.shape(tmp)
                    tmp_s = tmp.get_shape()
                    lex_cap = tmp_s[1].value
                    hid_dim = tmp_s[3].value
                    tmp = tf.reshape(tmp, [tmp_d[0], tmp_d[2]*lex_cap, hid_dim])
                    print(tmp.shape.as_list())
                    y = transformer.transformer_ffn_layer(tmp, hparams, pad_remover)
                    print(y.shape.as_list())
                    y = tf.reshape(y, [tmp_d[0], lex_cap, tmp_d[2], hid_dim])
                    print(y.shape.as_list())
                    x = common_layers.layer_postprocess(x, y, hparams)
                    print(x.shape.as_list())
            encoder_output = common_layers.layer_preprocess(x, hparams)

        print(encoder_output.shape.as_list()) # ==> [None, None, 4, 1024]

        # --------

        encoder_output = tf.transpose(encoder_output, perm=[0,2,3,1])
        encoder_output = tf.layers.dense(encoder_output, 1)
        encoder_output = tf.squeeze(encoder_output, [3])
        print(encoder_output.shape.as_list())
        
        # --------
        encoder_padding = common_attention.embedding_to_padding(encoder_output)
        ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
        encoder_self_attention_bias = ignore_padding
        encoder_decoder_attention_bias = ignore_padding
        print(encoder_self_attention_bias.shape.as_list()) # ==> [None, 1, 1, None]

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
            print(encoder_output.shape.as_list()) # ==> [None, None, 1024] (None, None, 4, 1024)

        return encoder_output, encoder_decoder_attention_bias

@registry.register_model
class TransformerLexbefore1d(transformer.Transformer):
    '''
    inherits Transformer which inherits t2t_model.T2TModel
    '''
    def model_fn_body(self, features):
        hparams = self._hparams
        
        encoder_input = features["inputs"] # ==> [None, None, None, 4, 1024]
        targets = features["target_space_id"]
        print("hhh")
        print(targets)
        input("haha")

        encoder_input = flatten5d4d(encoder_input)
        print(encoder_input.shape.as_list()) # ==> [None, None, 4, 1024]
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
        encoder_input: [batch_size, input_len, hidden_size]
        return: 
            encoder_output: [batch_size, input_len, hidden_size]
            encoder_decoder_attention_bias: [batch_size, input_len]
        '''
        encoder_output_slices = []
        for i in range(encoder_input.get_shape()[2].value):
            encoder_input_slice = encoder_input[:,:,i,:]

            # bias used in self-attention and encoder-decoder attention
            encoder_padding = common_attention.embedding_to_padding(encoder_input_slice)
            ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
            encoder_self_attention_bias = ignore_padding
            encoder_decoder_attention_bias = ignore_padding

            # add target space to encoder input?
            ishape_static = encoder_input_slice.shape.as_list()
            emb_target_space = common_layers.embedding(target_space, 32, ishape_static[-1], name="target_space_embedding")
            emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
            encoder_input_slice += emb_target_space

            # add timing signals to encoder input
            if hparams.pos == "timing":
                encoder_input_slice = common_attention.add_timing_signal_1d(encoder_input_slice)

            # dropout
            encoder_input_slice = tf.nn.dropout(encoder_input_slice, 1.0-hparams.layer_prepostprocess_dropout)
            encoder_output_slices.append(encoder_input_slice)
        encoder_output = tf.stack(encoder_output_slices, 2)
        print(encoder_output.shape.as_list()) # ==> [None, None, 4, 1024]
            
        # --------
        print("--------")

        s_tmp = encoder_output.get_shape() # static shape
        s_tmp2 = tf.shape(encoder_output) # dynamic shape
        d_tmp = s_tmp2[1]*s_tmp[3].value

        hparams2 = copy_hparams(hparams)

        encoder_output = tf.reshape(encoder_output, [s_tmp2[0],s_tmp[2],d_tmp])
        print(encoder_output.shape.as_list())
        encoder_padding = common_attention.embedding_to_padding(encoder_output)
        ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
        encoder_self_attention_bias = ignore_padding
        print(encoder_self_attention_bias.shape.as_list()) # ==> [None, 1, 1, None]

        # encoder
        x = encoder_output
        with tf.variable_scope("encoder_extra"):
            # remove pad 
            pad_remover = None
            if hparams.use_pad_remover:
                pad_remover = expert_utils.PadRemover(common_attention.attention_bias_to_padding(encoder_self_attention_bias))

            # self-attention along the lexicon dimension
            with tf.variable_scope("layer_extra"):
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
            print(encoder_output.shape.as_list())
        encoder_output = tf.reshape(encoder_output, [s_tmp2[0], s_tmp2[1], s_tmp[2].value, s_tmp[3].value])
        print(encoder_output.shape.as_list())

        # --------
        print("--------")

        encoder_output = tf.transpose(encoder_output, perm=[0,1,3,2])
        encoder_output = tf.layers.dense(encoder_output, 1)
        encoder_output = tf.squeeze(encoder_output, [3])

        print(encoder_output.shape.as_list()) # ==> [None, None, 1024]

        # --------
        print("--------")

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
            print(encoder_output.shape.as_list()) # ==> [None, None, 1024] (None, None, 4, 1024)
        return encoder_output, encoder_decoder_attention_bias

