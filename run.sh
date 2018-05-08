#!/bin/bash

verbose='false'
s=''
t=''
mt=phrase
oov_trans=dclm

usage() { echo -e "-s: source lanugage\n-t: target language\n-m: phrase, elisa, opennmt, tfnmt, t2t, cnn\n-o: dclm, pagerank, pmi, ngram, bound"; }

while getopts 'hs:t:m:o:v' flag; do
  case "${flag}" in
    s) s="${OPTARG}" ;;
    t) t="${OPTARG}" ;;
    m) mt="${OPTARG}" ;;
    o) oov_trans="${OPTARG}" ;;
    v) verbose='true' ;;
    h) usage; exit;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

if [ -z "$s" ]; then
  echo "no arg supplemented for s"; exit 1
fi
if [ -z "$t" ]; then
  echo "no arg supplemented for t"; exit 1
fi
if [ -z "$mt" ]; then
  echo "no arg supplemented for mt"; exit 1
fi
export mt
if [ "$mt" == "phrase" ] || [ "$mt" == "elisa" ]; then
  if [ -z "$oov_trans" ]; then
    echo "no arg supplemented for oov_trans"; exit 1
  fi
fi
export oov_trans

# isi, wmt, ldc
if [ "$s" == "de" ] || [ "$s" == "en" ]; then
  preproc=wmt
elif [ "$s" == "vie" ]; then
  preproc=ldc
else
  preproc=isi
fi


# preproc data
if [ "$preproc" == "wmt" ]; then
  sh preproc/preproc_wmt_data.sh $s $t
elif [ "$preproc" == "ldc" ]; then
  sh preproc/preproc_ldc_data.sh $s $t
elif [ "$preproc" == "isi" ]; then
  sh preproc/preproc_isi_data.sh $s $t
elif [ -z "$preproc" ]; then
  echo competition mode, no preparation needed
else
  return
fi

# train
if [ "$mt" == "phrase" ]; then
  sh train_tune_test/train_tune_test_phrase/train_tune_test_phrase.sh $s $t
elif [ "$mt" == "elisa" ]; then
  echo elisa system has already been trained
elif [ "$mt" == "opennmt" ]; then
  sh train_tune_test/train_tune_test_nmt/train_tune_test_nmt.sh $s $t
elif [ "$mt" == "tfnmt" ]; then
  sh train_tune_test/train_tune_test_tfnmt/train_tune_test_tfnmt.sh $s $t
elif [ "$mt" == "t2t" ]; then
  sh train_tune_test/train_tune_test_t2t/train_tune_test_t2t.sh $s $t
elif [ "$mt" == "cnn" ]; then
  sh train_tune_test/train_tune_test_cnn/train_tune_test_cnn.sh $s $t
else
  return
fi

# decode
if [ "$mt" == "phrase" ]; then
  sh train_tune_test/train_tune_test_phrase/decoder.sh $s $t
elif [ "$mt" == "elisa" ]; then
  sh train_tune_test/train_tune_test_isi/decoder.sh $s $t
elif [ "$mt" == "opennmt" ]; then
  sh train_tune_test/train_tune_test_nmt/decoder_nmt.sh $s $t
elif [ "$mt" == "tfnmt" ]; then
  sh train_tune_test/train_tune_test_tfnmt/decoder_tfnmt.sh $s $t
elif [ "$mt" == "t2t" ]; then
  sh train_tune_test/train_tune_test_t2t/decoder_t2t.sh $s $t
elif [ "$mt" == "cnn" ]; then
  sh train_tune_test/train_tune_test_cnn/decoder_cnn.sh $s $t
else
  return
fi

# oov_trans
source ./config.sh $s $t
if [ "$oov_trans" == "dclm" ]; then
  python oov_translate/method_dclm3/preproc/preproc.py
  python oov_translate/method_dclm3/train/train.py
  python oov_translate/method_dclm3/train/lattice_rescoring.py
elif [ "$oov_trans" == "ngram" ]; then
  python oov_translate/method_ngram/preproc/preproc.py
  python oov_translate/method_ngram/train/train.py
  python oov_translate/method_ngram/train/lattice_rescoring.py
elif [ "$oov_trans" == "pmi" ]; then
  python oov_translate/method_pmi2/preproc/preproc.py
  python oov_translate/method_pmi2/train/train.py
  python oov_translate/method_pmi2/train/lattice_rescoring.py
elif [ "$oov_trans" == "pagerank" ]; then
  python oov_translate/method_pagerank/lattice_rescoring.py
elif [ "$oov_trans" == "bound" ]; then
  python oov_translate/method_bound/lattice_rescoring.py
elif [ -z "$oov_trans" ]; then
  echo no OOV words
else
  return
fi


