# utils
graph\_utils
vocab

# basic model definitions
graph\_module (
  \_build: not implemented
)

# Configurable prototype
Configurable (
  __init__(params, mode)
  mode: property
  params: property
  default\_params: abstractstaticmethod
)

# model prototype
# more details in model\_fn in <https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Estimator>
model\_base (
  \_build: not implemented
  \_build\_train\_op
  \_create\_optimizer
  \_clip\_gradients
  batch\_size: not implemented
)

# seq2seq prototype: model\_base
seq2seq\_model (
  \_build: 
    features, labels = _preprocess(features, labels) 
    encoder_output = encode(features, labels)
    decoder_output, _, = decode(features, labels)
    INFER:
      _create_predictions(decoder_output, features, labels)
    TRAIN:
      compute_loss(decoder_output, features, labels)
      _build_train_op(loss, gradient_multipliers)
  \_clip\_gradients(grads\_and\_vars)
  batch\_size(features, labels)
  \_create\_predictions(decoder\_output, features, labels)
  \_preprocess(features, labels)

  source\_embedding: templatemethod, random uniform initializer
  target\_embedding: templatemethod, random uniform initializer
  encode: not implemented, templatemethod
  decode: not implemented, templatemethod
  \_get\_beam\_search\_decoder(decoder)
  use\_beam\_search
  compute\_loss
)

# conv-seq2seq instantiation: seq2seq\_model
conv\_seq2seq (
  source\_embedding\_fairseq
  target\_embedding\_fairseq
  source\_pos\_embedding\_fairseq
  target\_pos\_embedding\_fairseq
  \_create\_decoder(encoder\_output, features, \_labels)
  \_decode\_train(decoder, \_encoder\_output, \_features, labels)
  \_decoder\_infer(decoder, \_encoder\_output, features, labels)
  encode(features, labels)
  decode(encoder\_output, features, labels)
)

# basic seq2seq instantiation (as a reference to conv-seq2seq): seq2seq\_model
basic\_seq2seq (
  \_create\_bridge(encoder\_outputs, decoder\_state\_size)
  \_create\_decoder(\_encoder\_output, \_features, \_labels)
  \_decode\_train(decoder, bridge, \_encoder\_output, \_features, labels)
  \_decode\_infer(decoder, bridge, \_encoder\_output, features, labels)
  encode(features, labels): use source\_embedding
  decode(encoder\_output, features, labels)
)



