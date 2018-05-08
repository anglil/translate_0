from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

glove_mat = "/home/ec2-user/kklab/data/glove/glove.6B.300d.txt"
lexicon_dict_file = "/home/ec2-user/kklab/data/lorelei/LEXICONS/clean-merged/clean-merged/vie-eng.masterlex.txt"

@registry.register_hparams
def transformer_dim300_layer2():
    hparams = transformer.transformer_base()
    hparams.num_encoder_layers = 2
    hparams.num_decoder_layers = 2
    hparams.clip_grad_norm = 2.0
    hparams.batch_size = 512
    hparams.hidden_size = 300
    hparams.glove_dict_file = glove_mat
    hparams.lexicon_dict_file = lexicon_dict_file
    hparams.input_modalities = "inputs:symbol:lex_modality"
    hparams.target_modality = "symbol:default"
    hparams.shared_embedding_and_softmax_weights = int(False)
    hparams.symbol_modality_num_shards = 1
    hparams.num_heads = 6
    hparams.lex_cap = 4
    hparams.filter_size = 1024
    return hparams
