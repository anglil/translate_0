#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../config.sh $1 $2
source $BASE_DIR/../utils.sh
s=$1
t=$2

# {bleu meteor sent_bleu}
for metric in bleu; do
	# {dev test syscomb eval}
	for dataset in dev test; do
		suffix=${t}.${dataset}.${yrv}
		#ref=${trans_dir}/$dataset/ref.$suffix
		#hyp=${trans_dir}/$dataset/onebest.$suffix
    ref=${trans_dir}/$dataset/${ref_label}.$suffix
    ref_raw=$trans_dir/$dataset/${ref_label}_raw.$suffix
		hyp=${trans_dir}/$dataset/onebest.$suffix

		# -------- baseline --------
		#echo baseline: $s $metric $dataset isi
		#get_$metric $ref $hyp
		#echo --------
    echo baseline: $s $metric $dataset phrase
    hyp_phrase=$trans_dir/$dataset/onebest_phrase.$suffix
    get_$metric $ref_raw $hyp_phrase
    #echo --------
    #echo baseline: $s $metric $dataset opennmt
		#get_$metric $ref_raw $hyp_opennmt
		#echo --------
    #echo baseline: $s $metric $dataset tfnmt
    #python remove_bpe_in_tfnmt.py $hyp_tfnmt $hyp_tfnmt.tmp
    #python format_bpe_text.py $hyp_tfnmt.tmp $hyp_tfnmt.tmp.tmp
		#get_$metric $ref_raw $hyp_tfnmt.tmp.tmp
		#echo --------

		# -------- topline --------
		#echo topline: $s $metric $dataset lattice $url_handler
		#get_$metric $ref ${trans_dir}/${dataset}/best_${metric}_lattice
		#echo --------
		#echo topline: $s $metric $dataset lattice-align $url_handler
		#get_$metric $ref ${trans_dir}/${dataset}/best_${metric}_lattice-align
		#echo --------
		#echo topline: $s $metric $dataset align $url_handler
		#get_$metric $ref ${trans_dir}/${dataset}/best_ref_align
		#echo --------
    
    # -------- lex-seq2seq models --------
    #for dim in 512 1024; do
    #  for layer in 2; do
    #    for kernel in 3; do
    #      for injective in 0; do
    #        method=lex${injective}_dim${dim}_layer${layer}_kernel${kernel}
    #        hyp_raw=$trans_dir/$dataset/onebest_$method.$suffix
    #        echo lex-seq2seq: $s $metric $dataset $method
    #        get_$metric $ref_raw $hyp_raw
    #        echo --------
    #      done
    #    done
    #  done
    #done

    # -------- conv-seq2seq models --------
    ## 128 256 512
    #for dim in 128; do
    #  # 2
    #  for layer in 2; do
    #    # 2 3 4
    #    for kernel in 3; do
    #      # 8000 32000 inf 8000-8000
    #      for bpe in 8000-8000; do
    #        method=cnn_dim${dim}_layer${layer}_kernel${kernel}_bpe${bpe}_lex
    #        hyp_raw=$trans_dir/$dataset/onebest_$method.$suffix
    #        echo conv-seq2seq: $s $metric $dataset $method
    #        get_$metric $ref_raw $hyp_raw
    #        echo --------
    #      done
    #    done
    #  done
    #done
    
    # -------- t2t models --------
    ## 128 256 512 1024
    #for dim in 1024; do
    #  # 2
    #  for layer in 2; do
    #    # 8000 32000 inf 8000-8000
    #    for bpe in 8000; do
    #      method=t2t_dim${dim}_layer${layer}_bpe${bpe}_lex
    #      hyp_raw=$trans_dir/$dataset/onebest_$method.$suffix
    #      echo t2t: $s $metric $dataset $method
    #      get_$metric $ref_raw $hyp_raw
    #      echo --------
    #    done
    #  done
    #done

    ## -------- new ngram model --------
    #for candidate_source in extracted_eng_vocab; do
    #  for ngram in 4; do
    #    method=${ngram}gram_$candidate_source
    #    hyp_ngram=${trans_dir}/$dataset/$method.$suffix
    #    echo ngram: $s $metric $dataset $method
    #    get_$metric $ref $hyp_ngram
    #    echo --------
    #  done
    #done

    ## -------- new dclm model --------
    ## extracted, eng_vocab, extracted_eng_vocab, aligned_extracted, aligned
    #for candidate_source in aligned; do
    #  # adclm ccdclm codclm rnnlm
    #  for model_type in adclm; do
    #    # context beam
    #    for decoder_type in context; do
    #      # False True
    #      for include_charlm in False; do
    #        method=${candidate_source}_${model_type}_${decoder_type}_${include_charlm}
    #        hyp_dclm=${trans_dir}/$dataset/$method.$suffix
    #        #hyp_dclm_postproc=${trans_dir}/$dataset/$method.$suffix.postproc
    #        #if [ ! -f $hyp_dclm_postproc ]; then
    #        #  python $BASE_DIR/post_processing.py $hyp_dclm $hyp_dclm_postproc
    #        #fi
    #        echo dclm: $s $metric $dataset $method
    #        #get_$metric $ref $hyp_dclm_postproc
    #        get_$metric $ref $hyp_dclm
    #        echo --------
    #      done
    #    done
    #  done
    #done

		# {ug_dict_withoutAlignedOov ug_dict_withAlignedOov eng_vocab}
		#for candidate_source in ug_dict_withAlignedOov; do
		#	# -------- pmi --------
		#	# {bs bp bd}
		#	for context_scale in bd; do
		#		# {boolean_windown sliding_window}
		#		for window_mechanism in boolean_window; do
		#			method=${window_mechanism}_${context_scale}
		#			hyp_pmi=${hyp}.oovtranslated.${candidate_source}.${method}
		#			echo pmi: $s $metric $dataset $candidate_source $method $url_handler
		#			get_$metric $ref $hyp_pmi
		#			echo --------
		#		done
		#	done

		#	# -------- pagerank --------
		#	# {pagerank pagerank_incomplete_graph}
		#	for method in pagerank_incomplete_graph; do
		#		hyp_pagerank=${hyp}.oovtranslated.${candidate_source}.${method}
		#		echo pagerank: $s $metric $dataset $candidate_source $method $url_handler
		#		get_$metric $ref $hyp_pagerank
		#		echo --------
		#	done

		## -------- lm nbest --------
		#for method in 4gram 4gram_restrict_vocab; do
		#	hyp_lm=${hyp}.nbest.${method}
		#	echo lm nbest: $s $metric $dataset $method $url_handler
		#	get_$metric $ref $hyp_lm
		#  echo --------
		#done

		## -------- dclm nbest --------
		#for method in rnnlm ccdclm codclm adclm; do
		#	hyp_lm=${hyp}.nbest.${method}
		#	echo dclm nbest: $s $metric $dataset $method $url_handler
		#	get_$metric $ref $hyp_lm
		#	echo --------
		#done
		echo ----------------

	done
done



