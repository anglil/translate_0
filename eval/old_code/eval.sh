#!/bin/bash

source /home/ec2-user/kklab/Projects/lrlp/scripts/config.sh $1
source /home/ec2-user/kklab/Projects/lrlp/scripts/utils.sh

# {il3 amh som yor}
#for lang in amh som yor; do
	# {bleu meteor}
	for metric in bleu; do
		# {dev test}
		for dataset in dev; do
			if [ $dataset == dev ]; then
				ref=${dev_data_dir}/${dev_id}.${m_rec}.${t}
				hyp=${trans_dir}/${dataset}/${dev_id}.${m_tra}.${t}
			else
				ref=${test_data_dir}/${test_id}.${m_rec}.${t}
				hyp=${trans_dir}/${dataset}/${test_id}.${m_tra}.${t}
			fi

			# -------- baseline --------
			#echo baseline: $s $metric $dataset 
			#get_$metric $ref $hyp
			#echo --------

			# -------- topline --------
			#echo topline: $s $metric $dataset lattice
			#get_$metric $ref ${trans_dir}/${dataset}/best_${metric}_lattice
			#echo --------
			#echo topline: $s $metric $dataset lattice-align
			#get_$metric $ref ${trans_dir}/${dataset}/best_${metric}_lattice-align
			#echo --------
			#echo topline: $s $metric $dataset align
			#get_$metric $ref ${trans_dir}/${dataset}/best_ref_align
			#echo --------

			# -------- seq2seq --------
			#if [ $dataset == dev ]; then
			#	echo seq2seq: $s $metric $dataset
			#	get_$metric $ref ${trans_dir}/${dataset}/${dev_id}.${m_nmt}.${t}
			#else
			#	echo seq2seq: $s $metric $dataset
			#	get_$metric $ref ${trans_dir}/${dataset}/${test_id}.${m_nmt}.${t}
			#fi

			# {ug_dict_withoutAlignedOov ug_dict_withAlignedOov eng_vocab}
			for candidate_source in ug_dict_withAlignedOov ug_dict_withoutAlignedOov eng_vocab; do

			#	# -------- pmi --------
			#	# {bs bp bd}
			#	for context_scale in bd; do
			#		# {boolean_windown sliding_window}
			#		for window_mechanism in boolean_window; do
			#			method=${window_mechanism}_${context_scale}
			#			hyp_pmi=${hyp}.oovtranslated.${candidate_source}.${method}
			#			echo pmi: $s $metric $dataset $candidate_source $method
			#			get_$metric $ref $hyp_pmi
			#			echo --------
			#		done
			#	done

			#	# -------- pagerank --------
			#	# {pagerank pagerank_incomplete_graph}
			#	for method in pagerank_incomplete_graph; do
			#		hyp_pagerank=${hyp}.oovtranslated.${candidate_source}.${method}
			#		echo pagerank: $s $metric $dataset $candidate_source $method
			#		get_$metric $ref $hyp_pagerank
			#		echo --------
			#	done

			#	# -------- lm --------
			#	# {lm_4gram lm_4gram_restrict_vocab}
			#	for method in lm_4gram lm_4gram_restrict_vocab; do
			#		hyp_lm=${hyp}.oovtranslated.${candidate_source}.${method}
			#		echo lm: $s $metric $dataset $candidate_source $method
			#		get_$metric $ref $hyp_lm
			#		echo --------
			#	done
	
				# -------- dclm --------
				# {rnnlm ccdclm codclm adclm 
				# rnnlm_beam ccdclm_beam codclm_beam adclm
				# rnnlm_charlm ccdclm_charlm codclm_charlm adclm_charlm
				# rnnlm_beam_charlm ccdclm_beam_charlm codclm_beam_charlm adclm_beam_charlm}
				for method in adclm adclm_embed; do
					hyp_lm=${hyp}.oovtranslated.${candidate_source}.${method}
					echo dclm: $s $metric $dataset $candidate_source $method
					if [ ! -f ${ref}.lowercase ]; then
						python lowercase.py ${ref} ${ref}.lowercase
					fi
					get_$metric ${ref}.lowercase $hyp_lm
					echo --------
				done

			done			

			## -------- lm nbest --------
			#for method in 4gram 4gram_restrict_vocab; do
			#	hyp_lm=${hyp}.nbest.${method}
			#	echo lm nbest: $s $metric $dataset $method
			#	get_$metric $ref $hyp_lm
			#  echo --------
			#done

			## -------- dclm nbest --------
			#for method in rnnlm ccdclm codclm adclm; do
			#	hyp_lm=${hyp}.nbest.${method}
			#	echo dclm nbest: $s $metric $dataset $method
			#	get_$metric ${ref}.lowercase $hyp_lm
			#	echo --------
			#done
			#echo ----------------

		done
		echo ----------------
	done
	echo ----------------
#done



