Runs on tensor2tensor v1.2.5

# datagen
data_generators/problem.py
  (class Problem) def generate_data(data_dir, tmp_dir, task_id=-1)

# train
utils/trainer_utils.py
  def run(data_dir, model, output_dir, train_steps, eval_steps, schedule)
utils/trainer_utils.py
  def create_hparams(params_id, data_dir, passed_hparams=None) # register hparams
utils/trainer_utils.py
  def add_problem_hparams(hparams, problems) # register problem
utils/input_fn_builder.py
  def build_input_fn(mode, hparams, data_dir, num_datashards, ..., batch_size, dataset_split)
utils/model_builder.py
  def build_model_fn(model_name, problem_names, train_steps, ..., decode_hparams) # register model, register modality

#### the real deal, in detail, in order
# 1. register hparams, get tf.contrib.training.HParams object
utils/trainer_utils.py
  hparams = registry.hparams(params_id)()
utils/registry.py
  return _HPARAMS[name]
# 2. register problem, get Problem object (string -> id)
utils/trainer_utils.py
  problem = registry.problem(problem_name)
utils/registry.py
  return _PROBLEMS[base_name](was_reversed, was_copy)
# 3. register modality, get Modality object (id -> embedding)
utils/t2t_model.py
  input_modality[f] = registry.create_modality(modality_spec, hparams)
  target_modality = registry.create_modality(target_modality_spec, hparams)
utils/registry.py
  return retrieval_fns[modality_type](modality_name)(model_hparams, vocab_size)
# 4. register model, get T2TModel object
utils/model_builder.py
  model_class = registry.model(model)(hparams, model, hparams.problems[n], n, dp, devices.ps_devices(all_worders=True))
utils/registry.py
  return _MODELS[name]

# decode

# --------
# hparams
utils/registry.py
  _HPARAMS = {}
  def register_hparams(name=None)
  def hparams(name)
data_generators/problem.py
  def _default_hparams() # tf.contrib.training.HParams
  (class Problem) def get_hparams(model_hparams=None) # problem's hparams, not the general hparams
  (class Problem) def hparams(defaults, model_hparams) # problem's hparams, not the general hparams
utils/trainer_utils.py # register hparams
  def create_hparams(params_id, data_dir, passed_hparams=None)
layers/common_hparams.py
  @registry.register_hparams("basic_1")
  def basic_params1()

#problem
problems.py
  def problem(name)
utils/registry.py
  _PROBLEMS = {}
  def register_problem(name=None)
  def problem(name)
data_generators/problem.py
  class Problem(object)
  class Text2TextProblem(Problem)
data_generators/wmt.py
  class TranslateProblem(problem.Text2TextProblem)
  @registry.register_problem
  class TranslateEndeWmtBpe32l(TranslateProblem)

# model
utils/registry.py
  _MODELS = {}
  def register_model(name=None)
  def model(name)
utils/t2t_model.py
  class T2TModel(object)
models/transformer.py
  @registry.register_model
  class Transformer(t2t_model.T2TModel)

# modality
utils/registry.py
  class Modalities(object)
  _MODALITIES = {...} # pre-defined modalities
utils/modality.py
  class Modality(object)
  def create_modality(modality_spec, model_hparams)
utils/t2t_model.py
  (class T2TModel) def _create_modalities(problem_hparams, hparams) # register modality
layers/modalities.py
  class SymbolModality(modality.Modality)

####
# hparams seen at the registration of modality
[
  ('attention_dropout', 0.0), 
  ('attention_key_channels', 0), 
  ('attention_value_channels', 0), 
  ('batch_size', 2048), 
  ('clip_grad_norm', 2.0),
  ('compress_steps', 0),
  ('data_dir', '/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/t2t_inf'),
  ('dropout', 0.0),
  ('eval_drop_long_sequences', 0),
  ('factored_logits', 0),
  ('ffn_layer', 'conv_hidden_relu'),
  ('filter_size', 2048),
  ('grad_noise_scale', 0.0),
  ('hidden_size', 300),
  ('initializer', 'uniform_unit_scaling'),
  ('initializer_gain', 1.0),
  ('input_modalities', 'inputs:symbol:lex_modality'),
  ('kernel_height', 3),
  ('kernel_width', 1),
  ('label_smoothing', 0.1),
  ('layer_postprocess_sequence', 'dan'),
  ('layer_prepostprocess_dropout', 0.1),
  ('layer_preprocess_sequence', 'none'),
  ('learning_rate', 0.1),
  ('learning_rate_cosine_cycle_steps', 250000),
  ('learning_rate_decay_scheme', 'noam'),
  ('learning_rate_warmup_steps', 4000),
  ('length_bucket_step', 1.1),
  ('max_input_seq_length', 0),
  ('max_length', 256),
  ('max_relative_position', 0),
  ('max_target_seq_length', 0),
  ('min_length_bucket', 8),
  ('mode', 'train'),
  ('moe_hidden_sizes', '2048'),
  ('moe_k', 2),
  ('moe_loss_coef', 0.01),
  ('moe_num_experts', 64),
  ('multiply_embedding_mode', 'sqrt_depth'),
  ('nbr_decoder_problems', 1),
  ('norm_epsilon', 1e-06),
  ('norm_type', 'layer'),
  ('num_decoder_layers', 2),
  ('num_encoder_layers', 2),
  ('num_heads', 8),
  ('num_hidden_layers', 6),
  ('optimizer', 'Adam'),
  ('optimizer_adam_beta1', 0.9),
  ('optimizer_adam_beta2', 0.98), 
  ('optimizer_adam_epsilon', 1e-09),
  ('optimizer_momentum_momentum', 0.9),
  ('parameter_attention_key_channels', 0),
  ('parameter_attention_value_channels', 0),
  ('pos', 'timing'),
  ('prepend_mode', 'none'),
  ('problem_choice', 'adaptive'),
  ('proximity_bias', 0),
  ('relu_dropout', 0.0),
  ('sampling_method', 'argmax'),
  ('self_attention_type', 'dot_product'),
  ('shared_embedding_and_softmax_weights', 0),
  ('summarize_grads', 0),
  ('symbol_modality_num_shards', 16),
  ('target_modality', 'symbol:default'),
  ('use_fixed_batch_size', 0),
  ('use_pad_remover', 1),
  ('weight_decay', 0.0),
  ('weight_noise', 0.0)]

# hparams.problems
{
  'loss_multiplier': 1.0,
  'batch_size_multiplier': 1,
  'max_expected_batch_size_per_shard': 64,
  'input_modality': {'inputs': ('symbol', 8003)},
  'target_modality': ('symbol', 8003),
  'input_space_id': 32,
  'target_space_id': 3,
  'vocabulary': {
    'inputs': <tensor2tensor.data_generators.text_encoder.TokenTextEncoder object at 0x7fea6a08fd68>,
    'targets': <tensor2tensor.data_generators.text_encoder.TokenTextEncoder object at 0x7fea6a08fc88>},
  'was_reversed': False,
  'was_copy': False}

