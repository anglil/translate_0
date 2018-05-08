
The code does the following three things:
1. MT data preprocessing
2. machine translation
3. OOV translation

Preprcoessing and oov translation are done by running `run.sh`.
Run ```sh run.sh -h``` to see options for machine translation systems (e.g., phrase-based, self-attention-based or convolution-based system) and for OOV translation systems (e.g., DCLM, Pagerank, PMI, Ngram).

The code for the supported languages are **amh**, **il3**, **som**, **yor**, **hau**, **vie**, standing for Amharic, Uighur, Somali, Yoruba, Hausa and Vietnamese, respectively. The Vietnamese corpus comes from Linguistic Data Consortium, and other corpora come from the Lorelei project, requiring different preprocessing steps. The target language in the experiments has always been English, encoded as **eng**. Support for other target languages and source languages can be manually added in `config.sh`.

### Phrase-based machine translation
To run a phrase-based machine translation experiment while outputing OOV words and positions, go to `train_tune_test/train_tune_test_phrase` and run ```sh run.sh -h``` to see all the supported hyperparameters. For example, to train the phrase-based machine translation system on amh-eng, run
```shell
sh run.sh -s amh -t eng -k train
```

To train a phrase-based system on amh-eng with shared byte-pair encoded (BPE) training data as input, run
```shell
sh run.sh -s amh -t eng -k train -e -b -v 8000
```
where the shared BPE vocab size will be 8000.

To test the phrase-based system trained on the regular training data plus word pairs from external lexical sources, run
```shell
sh run.sh -s amh -t eng -k test -u
```

### Self-attention sequence to sequence model
To run the self-attention-based seq2seq model, go to `train_tune_test/train_tune_test_t2t` and run ```sh run.sh -h``` to see all the supported hyperparameters, including the gpu id to choose for training/testing, the task to run (prep/train/test),  hidden dimension, number of layers, whether to add word pairs from external lexical sources to the training data, learning rate, dropout rate at all layers, attention mechanism. Command line arguments also include whether the embedding layeris trainable, whether the embedding layer is randomly initialized, whether to cluster words based on synonyms, whether to use alignment or pretrained word embeddings for embedding layer initialization, whether to use byte-pair encoding/whether to share vocab between source and target/vocab size, etc.

For example, to train a self-attention based system on gpu 0 with 512 hidden dimension, 2 hidden layers, using byte-pair encoding shared between source and target into a 8000-sized vocab, run
```shell
sh run.sh -s amh -t eng -g 0 -k train -d 512 -n 2 -b -w -v 8000
```

### Convolution-based sequence to sequence model
To run the convolution-based sequence to sequence model, go to `train_tune_test/train_tune_test_cnn_tex` and run ```sh run.sh -h``` to see all the supported hypoparameters. The parameter that governs the way translation candidates are aggregated is the sixth parameter. If it's set to 0, the translation candidates for each word are aggregated right after the word embedding layer. If it's set to 1, the aggregation happens right before the encoder-decoder attention layer. If it's set to 2, the aggregation happens after the encoder-decoder attention layer, enabling the attention to be cast over all the translation candidates.

To see more command line arguments, vim into ```run.sh``` and try different hidden dimensions, numbers of layers, kernel widths of the convlution. There is also a parameter that dictates whether to attach word pairs from external lexicon sources to the training data.

### OOV translation
To run the dclm-based OOV translation experiment on phrased-based machine translation result for amh, for example, go back to the root directory and run
```shell
sh run.sh -s amh -t eng -m phrase -o dclm
```
To apply another OOV translation approach, simply change the `-o` flag.

There are a couple of parameters to use in DCLM-based OOV translation. 
1. To do OOV only alignment (as opposd to aligning every token from phrase-based output to Transformer-based output), change `only_oov' on line 1322 of `oov_translate/oov_candidates_preprocessing.py` to ```True```. 
2. To make it so that no adjacent alignments have the same aligned token, tune `no_repeat` on line 1324 to ```True```. 
3. To use the phase-based output as the base sequence, where every token in the phrase-based translation output has only one aligned position in the Transformer output, set `_1_2` to ```True``` on line 1321. Otherwise, if `_1_2` is set to ```False```, the Transformer output is used as the base sequence, and every token in the Transformer otuput has only one aligned position in the phrase-based translation output.
4. To use masterlexicon as the primary lexical source and use generated lists only when the masterlexicon doens't have a translation for a certain source word, put `underuse_projection` to ```True``` on line 1297. Otherwise, all lexical sources are treated equally, which may lead the OOV translation result to go down.
