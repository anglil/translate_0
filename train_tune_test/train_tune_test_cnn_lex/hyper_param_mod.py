import sys
import yaml
import argparse

parser = argparse.ArgumentParser(description="this script sets hyperparameters for cnn-seq2seq training")
parser.add_argument('--config_yml_in', help='Input config yaml', required=True)
parser.add_argument('--config_yml_out', help='Output config yaml', required=True)
parser.add_argument('--dim', help='Dimension of hidden layer and embedding layer', required=True)
parser.add_argument('--num_layer', help='Number of layers in encoder and decoder', required=True)
parser.add_argument('--kernel_width', help='Kernel width in encoder and decoder', required=True)
parser.add_argument('--lexicon_injective', help='0: bijective, 1: aggregate, 2: max-pooling', required=True)
args = parser.parse_args()

config_yml_in = args.config_yml_in
config_yml_out = args.config_yml_out
dim = int(args.dim)
num_layer = int(args.num_layer)
kernel_width = int(args.kernel_width)
injective = int(args.lexicon_injective)

config = yaml.load(open(config_yml_in))

embedding_dim = dim
encoder_dim = dim
encoder_layer = num_layer
decoder_dim = dim
decoder_layer = num_layer
encoder_kernel_width = kernel_width
decoder_kernel_width = kernel_width

config["model_params"]["encoder.params"]["cnn.layers"] = encoder_layer
config["model_params"]["encoder.params"]["cnn.nhids"] = ",".join([str(encoder_dim)]*encoder_layer)
config["model_params"]["encoder.params"]["cnn.kwidths"] = ",".join([str(encoder_layer)]*encoder_layer)
config["model_params"]["decoder.params"]["cnn.layers"] = decoder_layer
config["model_params"]["decoder.params"]["cnn.nhids"] = ",".join([str(decoder_dim)]*decoder_layer)
config["model_params"]["decoder.params"]["cnn.kwidths"] = ",".join([str(encoder_layer)]*decoder_layer)
if injective == 0:
    config["model"] = "LexSeq2Seq"
else:
    config["model"] = "LexmultiSeq2Seq"
    if injective == 1:
        config["model_params"]["encoder.class"] = "seq2seq.encoders.Lex1EncoderFairseq"
    elif injective == 2:
        config["model_params"]["encoder.class"] = "seq2seq.encoders.Lex2EncoderFairseq"
   
with open(config_yml_out, 'w') as fw:
    yaml.dump(config, fw, default_flow_style=False)

