``Low Resource Language Processing``


1. pagerank
2. pointwise mutual information
3. n-gram backoff language model + lattice decoding
4. document context language model + lattice decoding
5. character level language model for UNK handling

``API of oov_candidates_preprocessing``
`get lexicon`
1. get_eng_vocab
get oov candidates from common english words
2. get_ug_dict2 
get oov candidates from Katrin's extracted candidates
3. get_oov_candidates_from_isi_xml
get oov candidates from an lexicon in xml
4. get_oov_candidates_from_extracted
get oov candidates from Katrin's extracted candidates and from an lexicon in xml
5. get_oov_candidates_from_multiple_sources
get oov candidates from a list of dictionaries
6. get_oov_candidates_from_googletranslate
get oov candidates from google translation by Leanne
7. get_oov_candidates_from_glosbe
get oov candidates from glosbe
8. get_oov_candidates_from_master_lexicon
get oov candidates from master lexicon

`helpers`
1. align_oov
fast align
2. parse_align_output
parse the fast align output
3. merge_candidate_lists
merge candidates dicts -- pos:candidates
4. parse_oov_candidate_line
parse one line in the oov candidiate file
5. get_lexicon_xml_path
get the path of the lexicon in xml
6. glosbe_translate
translate one word using glosbe

`write candidate list file using a lexicon`
1. write_candidate_list_file_from_aligned
write oov candidates fromalignment into candidate_list_file 
2. write_candidate_list_file_from_glosbe
wirte oov candidates from glosbe into candidate_list_file
3. write_candidate_list_file_from_extracted
write oov candidates from Katrin's extraction and the lexicon in xml into candidate_list_file
4. write_candidate_list_file_from_eng_vocab
write oov candidates from common english words into candidate_list_file
5. write_candidate_list_file_from_extracted_eng_vocab
write oov candidates from extraction+eng_vocab into candidate_list_file
6. write_candidate_list_file_from_two_sources
write oov candidates from any two sources into candidate_list_file
7. write_candidate_list_file_from_multiple_sources
write oov candidates from multiple sources into candidate_list_file

`init`
1. dataset
dev, test, etc.
2. candidate_source
glosbe, googletranslate, master_lexicon, extracted, eng_vocab, aligned, isi_xml
