import sys
import os
import re
import xml.etree.ElementTree as et
import json
import argparse

parser = argparse.ArgumentParser(description="This is a parser of the isi xml file")
parser.add_argument('--xml', help='Input xml file', required=True)
parser.add_argument('--source_lang', help="Source language", required=True)
parser.add_argument('--mt_source', help='Output source file for MT')
parser.add_argument('--mt_target', help='Output target file for MT')
parser.add_argument('--reference', help='Output reference file')
parser.add_argument('--onebest', help='Output onebest file')
parser.add_argument('--nbest', help='Output nbest file')
parser.add_argument('--oov', help='Output oov file')
#parser.add_argument('--oov_unique', help='Output unique, untranslated, oov file')
args = parser.parse_args()

xml_file = args.xml
source_language = args.source_lang
mt_source_file = args.mt_source
mt_target_file = args.mt_target # untokenized ref
reference_file = args.reference # tokenized ref
onebest_file = args.onebest # follow the format of pos1:candidate1,candidate2 pos2:candidate3
nbest_file = args.nbest # follow Moses nbest format
oov_file = args.oov
#oov_unique_file = args.oov_unique

roman_ab = {"som", "yor", "eng", "en", "de"}

########
# parse xml_file into separate files
########

tree = et.parse(xml_file) 
root = tree.getroot()

sent_ctr = 0
doc_ctr = 0

oov_word_list = list()
oov_word_dict = dict()
oov_word_set = set()

if mt_source_file != None:
    fs = open(mt_source_file, 'w')
if mt_target_file != None:
    ft = open(mt_target_file, 'w')
if reference_file != None:
    fr = open(reference_file, 'w')
if onebest_file != None:
    fo = open(onebest_file, 'w')
if nbest_file != None:
    fn = open(nbest_file, 'w')
if oov_file != None:
    fv = open(oov_file, 'w')

null_src = 0
null_ref = 0
null_hyp = 0

for doc in root.findall('DOCUMENT'):
    for seg in doc.findall('SEGMENT'):
        # 1. source
        src = seg.find('SOURCE')
        if src != None:
            mt_source_text = src.find('ORIG_RAW_SOURCE').text
        else:
            null_src += 1

        # 2. target (reference)
        target = seg.find('TARGET')
        if target != None:
            mt_target_text = target.find('ORIG_RAW_TARGET').text
            try:
                ref_text = target.find('AGILE_TOKENIZED_TARGET').text
            except:
                ref_text = target.find('ULF_TOKENIZED_TARGET').text
        else:
            null_ref += 1

        # 3. hypotheses
        onebest_cost = 1e9
        onebest_elem = {}         
        nbest_vec = []
        nbest = src.find('NBEST')
        if nbest != None:
            for hyp in nbest.findall('HYP'):
                # get each hypothesis
                align = hyp.find('ALIGNMENT')
                target_tok = align.find('TOKENIZED_TARGET')

                # get oov of this hypothesis
                tok_ctr = 0
                oov_pos = []
                oov_word = []
                hyp_text_tok = []
                for tok in target_tok.findall('TOKEN'):
                    if tok.attrib['rule-class'] == "unknown":
                        oov = tok.text
                        if source_language not in roman_ab:
                            if not re.match("^[a-zA-Z0-9_]*$", oov):
                                oov_word.append(oov)
                                oov_pos.append(tok_ctr)
                        else:
                            oov_word.append(oov)
                            oov_pos.append(tok_ctr)
                    hyp_text_tok.append(tok.text)

                    tok_ctr += 1

                # get text of this hypothesis
                hyp_text = ' '.join(hyp_text_tok)
                #hyp_text = hyp.find('TEXT').text
                #assert(tok_ctr==len(hyp_text.split(' ')))

                # get cost of this hypothesis
                cost = hyp.attrib["cost"]

                # aggregate this hypothesis
                hyp_elem = {"hyp":hyp_text, "cost":cost, "oov_pos":" ".join([str(pos) for pos in oov_pos]), "oov_word":oov_word}
                nbest_vec.append(hyp_elem)

                # check if this hypothesis is one best
                cost = float(cost)
                if cost < onebest_cost:
                    onebest_cost = cost
                    onebest_elem = dict(hyp_elem)
        else:
            null_hyp += 1

        ### mt_source
        if mt_source_file != None:
            fs.write(mt_source_text+"\n")           

        ### mt_target
        if mt_target_file != None:
            ft.write(mt_target_text+"\n")

        ### reference text
        if reference_file != None and null_ref == 0:
            fr.write(ref_text+"\n")

        ### 1best
        if onebest_file != None:
            oov_word_list+=onebest_elem["oov_word"]
            for oov in onebest_elem["oov_word"]:
                oov_word_dict[oov] = 1 if oov not in oov_word_dict else oov_word_dict[oov]+1
                oov_word_set.add(oov)
            fo.write(onebest_elem["hyp"]+"\n")

        ### nbest
        if nbest_file != None:
            for ele in nbest_vec:
                l = " ||| ".join([str(sent_ctr), ele["hyp"], ele["oov_pos"], ele["cost"]])
                fn.write(l+"\n")

        ### oov
        if oov_file != None:
            fv.write(onebest_elem["oov_pos"]+"\n")

        sent_ctr += 1

    doc_ctr += 1

    if mt_source_file != None and null_src == 0:
        fs.write("=\n")
    if mt_target_file != None and null_ref == 0:
        ft.write("=\n")
    if reference_file != None and null_ref == 0:
        fr.write("=\n")
    if onebest_file != None and null_hyp == 0:
        fo.write("=\n") 
    if nbest_file != None:
        fn.write("=\n")
    if oov_file != None:
        fv.write("=\n")

#print("number of null src: "+str(null_src))
#print("number of null ref: "+str(null_ref))
#print("number of null hyp: "+str(null_hyp))
print("number of OOV words: "+str(len(oov_word_list)))
print("number of unique OOV words: "+str(len(oov_word_set)))
print("number of sentences: "+str(sent_ctr))
print("number of documents: "+str(doc_ctr))

## get **unique oov words** that have not been translated
#if oov_unique_file != None:
#    dir_path = os.path.dirname(os.path.realpath(__file__))+"/"
#    sys.path.insert(0, dir_path+'../oov_translate/')
#    from oov_candidates_preprocessing import *
#    ocp = oov_candidates_preprocessing()
#    oov_candidates_all = ocp.get_oov_candidates_from_extracted(oov_candidates_dir, ocp.get_lexicon_xml_path())
#    
#    with open(oov_unique_file, 'w') as fu:
#        for oov_word in oov_word_set:
#            if oov_word not in oov_candidates_all:
#                fu.write(oov_word+"\n")
