import subprocess as sp
import os
import sys
import json
import requests
import string

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)
from config import *
from utils import *

# get lexicon xml path
def get_lexicon_xml_path():
    lexicon_xml = None
    for filename in os.listdir(corpus_dir):
        if "lexicon" in filename:
            lexicon_xml = corpus_dir+filename
    return lexicon_xml

# glosbe: a lexicon collected from glosbe, by angli
# googletranslate: a lexicon collected from google translate, by leanne+jonathan
# masterlexicon: master lexicon, by katrin+leanne
# isixml: a lexicon collected by isi
# extracted: a lexicon collected from related languages, by katrin
# engvocab: 10k common english words, maintained by angli
# aligned: a lexicon collected from alignment by angli

lexicon_files = {
    "glosbe":
        os.path.join(home_dir, "Projects", "lrlp", "scripts", "lexicon", "dict_glosbe", "dict_"+s+"_"+t), 
    "googletranslate":
        os.path.join(home_dir, "Projects", "lrlp", "scripts", "lexicon", "dict_googletranslate", "checked-"+s+".txt"),
    "masterlexicon":
        bilingual_lexicon, 
    "isixml":
        get_lexicon_xml_path(),
    "extracted":
        os.path.join(home_dir, "data", "lorelei", "for-angli", ".".join([s,"uniq","oov","hyps"])),
    "engvocab":
        eng_vocab_file,
    "aligned":
        None}

class oov_candidates_preprocessing:
    ###
    def get_engvocab(self, eng_list):
        '''
        load the english vocabulary except the function words into a lower cased word set: {candidate}
        params:
            eng_list: path to vocab file
        return:
            eng_vocab: {candidate}
        '''
        print('building common english word vocabulary...')
        eng_vocab = set()
        with open(eng_list) as f:
            for line in f:
                w = line.strip().lower() # lower c
                if w not in function_words:
                    eng_vocab.add(w)
        print('common english word vocabulary is built!')
        return eng_vocab
    
    
        
    def get_oov_candidates_from_extracted(self, ug_rank_file, take_phrase=False):
        '''
        retrieve a dictionary: {oov words: {english translations}}
        params:
            ug_rank_file: path to an oov vocab
            take_phrase: whether phrase candidates should be taken into account
        return:
            ug_dict: {oov words: {english translations}}
        '''
        ug_dict = dict()
        if os.path.exists(ug_rank_file):
            ctr = 0
            with open(ug_rank_file) as f:
                for line in f:
                    #l = line.strip().split('\t')
                    l = line.strip().split('  ')
                    if len(l) > 1 and l[1] != '[NOHYPS]':
                        ug_word = l[0].lower() # lower case
                        en_hyp = set()
                        
                        if len(l) == 3:
                            en_list = l[2].split(";")
                        elif len(l) == 2:
                            en_list = l[1].split(";")
                        else:
                            continue
                        
                        en_list.pop(-1)
                        for word_score_pair in en_list:
                            word_score_list = word_score_pair.split(',')
                            if len(word_score_list) == 2:
                                en_word = word_score_list[0].lower() # lower case
                                #score = float(word_score_list[1])
                                if take_phrase:
                                    if en_word:
                                        en_hyp.add(en_word)
                                else:
                                    en_word = en_word.strip()
                                    #if len(en_word.split(' ')) == 1 and en_word:
                                    #    en_hyp.add(en_word)
                                    en_words = en_word.split(' ')
                                    en_words = sorted(en_words, key=lambda k:len(k))
                                    en_hyp.add(en_words[-1])
                        if take_phrase:
                            ug_dict[ug_word] = en_hyp
                        else:
                            if en_hyp: 
                                ug_dict[ug_word] = en_hyp
            
                    ctr += 1
            for oov_word in ug_dict:
                assert(ug_dict[oov_word])
            print("extracted lexicon loaded from: "+ug_rank_file)
        else:
            print("extracted lexicon doesn't exist at: "+ug_rank_file)
        return ug_dict
    
    def get_oov_candidates_from_isixml(self, lexicon_xml):
        '''
        get oov candidates/hypotheses from an external lexicon in xml
        return {oov: {translations}}
        '''
        ug_dict = dict()

        if s == "il3" or s == "rus":
            tree = et.parse(lexicon_xml)
            root = tree.getroot()
            
            for entry in root.iter('ENTRY'):
                # uyghur lexicon format
                if entry.find('WORD') != None:
                    s_words = entry.find('WORD').text.split(' ')
                    for sense in entry.findall('SENSE'):
                        if sense.find('DEFINITION').text != None:
                            definition = sense.find('DEFINITION').text.strip()
                            definition = definition[:-1] if definition.endswith(".") else definition
                            for t_words in definition.split(','):
                                t_words = [item for item in t_words.split(' ') if item != '']
                                if len(t_words)==len(s_words):
                                    for i in range(len(t_words)):
                                        s_word = s_words[i]
                                        t_word = t_words[i]
                                        if s_word and t_word:
                                            if s_word not in ug_dict:
                                                ug_dict[s_word] = {t_word}
                                            else:
                                                ug_dict[s_word].add(t_word)
                # russian lexicon format
                else:
                    s_words = entry.find('LEMMA').text.split(' ')
                    for gloss in entry.findall('GLOSS'):
                        if gloss.text != None:
                            gloss0 = gloss.text.strip()
                            gloss0 = gloss0[:-1] if gloss0.endswith(".") else gloss0
                            for t_words in gloss0.split(','):
                                t_words = [item for item in t_words.split(' ') if item != '']
                                if len(t_words)==len(s_words):
                                    for i in range(len(t_words)):
                                        s_word = s_words[i]
                                        t_word = t_words[i]
                                        if s_word and t_word:
                                            if s_word not in ug_dict:
                                                ug_dict[s_word] = {t_word}
                                            else:
                                                ug_dict[s_word].add(t_word)
        # Tigrinya
        if s == "il5":
            with open(lexicon_xml) as f:
                for line in f:
                    l = line.strip().split('\t')
                    if len(l) == 2:
                        s_words = l[1].split(' ')
                        t_words = l[0].split(' ')
                        if len(t_words)==len(s_words):
                            for i in range(len(t_words)):
                                s_word = s_words[i]
                                t_word = t_words[i]
                                if s_word and t_word:
                                    if s_word not in ug_dict:
                                        ug_dict[s_word] = {t_word}
                                    else:
                                        ug_dict[s_word].add(t_word)
        # Oromo
        if s == "il6":
            with open(lexicon_xml) as f:
                for line in f:
                    l = line.strip().split('\t')
                    if len(l) == 3:
                        s_words = l[0].split(' ')
                        t_words = l[2].split(' ')
                        if len(t_words)==len(s_words):
                            for i in range(len(t_words)):
                                s_word = s_words[i]
                                t_word = t_words[i]
                                if s_word and t_word:
                                    if s_word not in ug_dict:
                                        ug_dict[s_word] = {t_word}
                                    else:
                                        ug_dict[s_word].add(t_word)
        for oov_word in ug_dict:
            assert(ug_dict[oov_word])
        return ug_dict
 

    ## get oov candidates from extracted (and any available local lexicon)
    #def get_oov_candidates_from_extracted(self, oov_candidates_dir, lexicon_xml=None):
    #    '''
    #    to get oov candidate map from the extracted files
    #    input: oov_candidates_dir, lexicon_xml (optional)
    #    return:
    #        oov_candidates_all: {oov:candidates}
    #    '''
    #    print("building oov candidate dictionary...")
    #    oov_candidates_all = dict()
    #    # from extracted files
    #    for filename in os.listdir(oov_candidates_dir):
    #        # select oov candidate files in the corresponding source language
    #        if s in filename:
    #            oov_candidates = self.get_ug_dict2(oov_candidates_dir+filename, False)
    #            for oov_word in oov_candidates:
    #                if oov_word not in oov_candidates_all:
    #                    oov_candidates_all[oov_word] = oov_candidates[oov_word]
    #                else:
    #                    oov_candidates_all[oov_word] |= oov_candidates[oov_word]
    #    # from an external lexicon in xml
    #    if os.path.exists(lexicon_xml):
    #        oov_candidates = self.get_oov_candidates_from_isixml(lexicon_xml)
    #        for oov_word in oov_candidates:
    #            if oov_word not in oov_candidates_all:
    #                oov_candidates_all[oov_word] = oov_candidates[oov_word]
    #            else:
    #                oov_candidates_all[oov_word] |= oov_candidates[oov_word]
    #    else:
    #        print("lexicon doesn't exist at ["+lexicon_xml+"]!")
    #                
    #    print("oov candidate dictionary is built!")
    #    return oov_candidates_all



    ## get oov candidates from google translation collected by leanne
    #def get_oov_candidates_from_googletranslate(self, infile):
    #    '''
    #    return {oov:{candidates}}
    #    '''
    #    oov_candidates_dict = dict()
    #    if os.path.exists(infile):
    #        with open(infile) as f:
    #            for l in f:
    #                line = l.strip().split('\t')
    #                assert(len(line)==3)
    #                if int(line[2]) > 0:
    #                    candidates = line[1].split(',')
    #                    candidates = [candidate.lower() for candidate in candidates if len(candidate.split(' '))==1] # lower c
    #                    if candidates != []:
    #                        oov_word = line[0]
    #                        oov_candidates_dict[oov_word]=set(candidates)
    #        print("googletranslate lexicon loaded from: "+infile)
    #    else:
    #        print("googletranslate lexicon doesn't exist at: "+infile)
    #    return oov_candidates_dict

    def get_oov_candidates_from_googletranslate(self, infile):
        '''
        return {oov:{candidates}}
        '''
        oov_candidates_dict = dict()
        if os.path.exists(infile):
            with open(infile) as f:
                for l in f:
                    line = l.strip().split('\t')
                    assert(len(line)==2)
                    oov_word = line[0].lower() # lower case
                    candidates = line[1].split(",")
                    for candidate in candidates:
                        c = candidate.split(".")[0]
                        c = c.split(" ")
                        if len(c) != 1:
                            c = sorted(c, key=lambda k:len(k), reverse=True)
                        c = c[0].lower() # lower case
                        if oov_word not in oov_candidates_dict:
                            oov_candidates_dict[oov_word] = {c}
                        else:
                            oov_candidates_dict[oov_word].add(c)
            print("googletranslate lexicon loaded from: "+infile)
        else:
            print("googletranslate lexicon doesn't exist at: "+infile)
        return oov_candidates_dict

    def get_oov_candidates_from_glosbe(self, local_glosbe_lexicon):
        '''
        return {oov:{candidates}}
        '''
        oov_candidates_dict = dict()
        if os.path.exists(local_glosbe_lexicon):
            with open(local_glosbe_lexicon) as f:
                for line in f:
                    line = line.strip()
                    src_word = line.split(":")[0].lower() # lower c
                    tgt_words = line.split(":")[1].split(",")
                    if tgt_words == ['']:
                        tgt_words = []
                    assert(src_word not in oov_candidates_dict)
                    if tgt_words != []:
                        oov_candidates_dict[src_word] = set([w.lower() for w in tgt_words]) # lower c
            print("glosbe lexicon loaded from: "+local_glosbe_lexicon)
        else:
            print("glosbe lexicon doesn't exist at: "+local_glosbe_lexicon)
        return oov_candidates_dict

    def get_oov_candidates_from_masterlexicon(self, infile):
        '''
        return {oov:{candidates}}
        '''
        oov_candidates_dict = dict()
        if os.path.exists(infile):
            ctr_line = 0
            with open(infile) as f:
                for l in f:
                    line = l.strip().split('\t')
                    words = line[0].lower() # lower case
                    translations = line[5].lower() # lower c
                    if len(words.split(' ')) == len(translations.split(' ')):
                        words = words.split(' ')
                        translations = translations.split(' ')
                        for i in range(len(words)):
                            word = words[i]
                            translation = translations[i]
                            if translation != "n/a":
                                if word in oov_candidates_dict:
                                    if translation not in oov_candidates_dict[word]:
                                        oov_candidates_dict[word].add(translation)
                                else:
                                    oov_candidates_dict[word] = {translation}
                    ctr_line += 1
            print("master lexicon loaded from: "+infile)
        else:
            print("master lexicon doesn't exist at: "+infile)
        return oov_candidates_dict

    # get oov candidates from a list of dictionaries
    def get_oov_candidates_from_multiple_sources(self, oov_candidates_dict_list):
        oov_candidates_dict = dict()
        for oov_candidates in oov_candidates_dict_list:
            for oov_word in oov_candidates:
                if oov_word not in oov_candidates_dict:
                    oov_candidates_dict[oov_word] = oov_candidates[oov_word]
                else:
                    oov_candidates_dict[oov_word] |= oov_candidates[oov_word]
        return oov_candidates_dict

    ################

    # obsolete
    def get_ug_dict(self, ug_rank_file, take_phrase=False):
        '''
        retrieve a dictionary: {oov words: {english translations: score}}, all lower cased
        params:
            ug_rank_file: path to an oov vocab
            take_phrase: whether phrase candidates should be taken into account
        return:
            ug_dict: {oov words: {english translations: score}}
        '''
        ug_dict = dict()
        ctr = 0 
        with open(ug_rank_file) as f:
            for line in f:
                l = line.strip().split('\t')
                if len(l) > 1 and l[1] != '[NOHYPS]':
                    ug_word = l[0].lower() # in case it is an english word, lower case the uyghur word
                    en_hyp = {}
                    
                    if len(l) == 3:
                        en_list = l[2].split(";")
                    elif len(l) == 2:
                        en_list = l[1].split(";")
                    else:
                        continue
                    
                    en_list.pop(-1)
                    for word_score_pair in en_list:
                        word_score_list = word_score_pair.split(',')
                        if len(word_score_list) == 2:
                            en_word = word_score_list[0].lower() # lower case the english word
                            score = float(word_score_list[1])
                            if take_phrase:
                                en_hyp[en_word] = score
                            else:
                                if len(en_word.split(' ')) == 1:
                                    en_hyp[en_word] = score
                    if take_phrase:
                        ug_dict[ug_word] = en_hyp
                    else:
                        if en_hyp != {}: 
                            ug_dict[ug_word] = en_hyp
        
                ctr += 1    
        return ug_dict
    

    # fast align
    # every token in source aligns with a token in target
    def align_oov(self, onebest_file, ref_file, align_output):
        '''
        params:
            onebest_file: source file for fast-align
            ref_file: target file for fast-align
            align_output: path to output generated by fast-align
        return:
            None
        '''
        ### parallel text fed to fast-align
        align_dir = os.path.join(tmp_dir, "align_dir")
        if not os.path.exists(align_dir):
            os.makedirs(align_dir)
        align_input = os.path.join(tmp_dir, os.path.join(align_dir, "align_input"))
        
        with open(onebest_file) as ft, open(ref_file) as fr, open(align_input, 'w') as fa:
            for l_tra in ft:
                l_tra = l_tra.strip()
                l_ref = fr.readline().strip()
                fa.write(l_tra) # source; left; 
                fa.write(" ||| ")
                fa.write(l_ref) # target; right
                fa.write('\n')
    
        ### source: translation with oov
        ### target: reference
        sh(fast_align+" -i "+align_input+" -d -o -v -r > "+align_output)
        print("alignment created from: "+align_input+" to "+align_output)
        
 
    # parse fast align output
    def parse_align_output(self, align_output_line, ref_line, output_tgt_pos=False):
        pos_to_word = {}
        pos_to_pos = {}
        if align_output_line.strip() == "":
            if output_tgt_pos:
                return pos_to_word, pos_to_pos
            return pos_to_word
        align_output_line = align_output_line.strip().split(' ')
        ref_line = ref_line.strip().split(' ')
        for pair in align_output_line:
            src_pos = int(pair.split('-')[0])
            tgt_pos = int(pair.split('-')[1])
            assert src_pos not in pos_to_word
            pos_to_word[src_pos] = ref_line[tgt_pos]
            pos_to_pos[src_pos] = tgt_pos
        if output_tgt_pos:
            return pos_to_word, pos_to_pos
        return pos_to_word
 
    
    # merge candidate lists -- pos:[candidates]
    def merge_candidate_lists(self, candidate_list1, candidate_list2, underuse_projection=False):
        candidate_map = dict(candidate_list1)
        for pos in candidate_list2:
            if pos not in candidate_map:
                candidate_map[pos] = candidate_list2[pos]
            else:
                if not underuse_projection:
                    candidate_map[pos] |= candidate_list2[pos]
        for pos in candidate_map:
            assert(candidate_map[pos])
        return candidate_map


    # parse a line in the oov candidate file
    def parse_oov_candidate_line(self, line, is_candidate_set=True):
        pos_to_candidate = dict()
        if line.strip() != "":
            for item in line.strip().split(' '):
                oov_pos = int(item.split(':')[0])
                if is_candidate_set:
                    oov_candidates = set(item.split(':')[1].split(','))
                else:
                    oov_candidates = item.split(':')[1].split(',')
                pos_to_candidate[oov_pos] = oov_candidates
        return pos_to_candidate




    # translate using glosbe
    def glosbe_translate(self, src_word):
        func = "translate"
        params = {
            "from":lang_code[s],
            "dest":lang_code[t],
            "format":"json",
            "phrase":src_word,
            "pretty":"true"}
        url = "https://glosbe.com/gapi/"+func+"?"+"&".join([key+"="+value for key,value in params.items()])
        r = requests.get(url)
        j = json.loads(r.content)

        oov_candidates_list = []
        if 'tuc' in j:
            translation = j['tuc']
            for item in translation:
                if "phrase" in item:
                    if item["phrase"]["language"] == lang_code[t]:
                        tgt_words = item["phrase"]["text"]
                        for tgt_word in tgt_words.split(", "):
                            if len(tgt_word.split(" "))==1 and tgt_word not in oov_candidates_list:
                                oov_candidates_list.append(tgt_word.lower())
            return oov_candidates_list
        else:
            # blocked for excess usage
            return "blocked"

    def is_oov_word(self, src_word, src_lang):
        for c in src_word:
            if c not in printables:
                return True
        if src_lang in roman_ab:
            for c in src_word:
                if c in digits or c in punctuations:
                    return False
            if src_word.strip().lower() in eng_vocab_set0:
                return False
        return True

        #src_w = "".join([i for i in src_word if i not in punctuations])
        #if src_lang in roman_ab:
        #    return (not src_w.isdigit())
        #else:
        #    #return (not src_w.isalnum())
        #    return re.match("^[a-zA-Z0-9_]*$", src_w)==None

    ############################

    # write candidate list file from aligned (with hyp)
    def write_candidate_list_file_from_alignedhyp(self, onebest_file_1, candidate_list_file_1, onebest_file_2, candidate_list_file_output, _1_2=True, only_oov=False, no_repeat=False, oov_sift=True):
        '''
        if _1_2: the base sequence is onebest_file_1 (pbmt)
        else: the base sequence is onebest_file_2 (transformer)

        input: 
            onebest_file_1 (with oov slots, e.g., pbmt output), 
            candidate_list_file_1 (oov positions and candidates),
            onebest_file_2 (without oov slots, e.g., transformer output),
        output: 
            candidate_list_file_output: based on onebest_file_1 if _1_2 is true, based on onebest_file_2 if _1_2 is false
        '''
        from collections import OrderedDict

        align_dir = os.path.join(tmp_dir, "align_dir2")
        if not os.path.exists(align_dir):
            os.makedirs(align_dir)
        align_output = os.path.join(align_dir, "align_output")
        if _1_2:
            self.align_oov(onebest_file_1, onebest_file_2, align_output)
        else:
            self.align_oov(onebest_file_2, onebest_file_1, align_output)

        with open(onebest_file_1) as f1, open(candidate_list_file_1) as fc1, open(onebest_file_2) as f2, open(align_output) as fa, open(candidate_list_file_output, "w") as fo:
            for l1 in f1:
                lc1 = fc1.readline()
                l2 = f2.readline()
                la = fa.readline()

                if l1.strip() == "=":
                    #assert(lc1.strip() == "=")
                    assert(l2.strip() == "=")
                    fo.write("=\n")
                    continue

                pos_to_candidate_1 = {int(pos_candidate.split(":")[0]):pos_candidate.split(":")[1].split(",") for pos_candidate in lc1.strip().split(" ") if pos_candidate != ""}
                pos_to_word = {}
                if _1_2:
                    # use 1 (pbmt) as the base sequence
                    pos_to_word_1_2, pos_to_pos_1_2 = self.parse_align_output(la, l2, output_tgt_pos=True)
                    l1 = l1.strip().split(" ")
                    for pos in range(len(l1)):
                        if pos in pos_to_word_1_2 and pos_to_word_1_2[pos] and pos_to_word_1_2[pos] not in punctuations and pos_to_word_1_2[pos] not in digits and l1[pos] and l1[pos] not in punctuations and l1[pos] not in digits:
                            if pos not in pos_to_candidate_1:
                                if not only_oov:
                                    if no_repeat:
                                        if pos-1 >= 0 and pos-1 in pos_to_pos_1_2:
                                            if pos_to_pos_1_2[pos] != pos_to_pos_1_2[pos-1]:
                                                pos_to_word = add_to_dict_set(pos, pos_to_word_1_2[pos], pos_to_word)
                                        else:
                                            pos_to_word = add_to_dict_set(pos, pos_to_word_1_2[pos], pos_to_word)
                                    else:
                                        pos_to_word = add_to_dict_set(pos, pos_to_word_1_2[pos], pos_to_word)
                                    pos_to_word = add_to_dict_set(pos, l1[pos], pos_to_word)
                            else:
                                cond = True if not oov_sift else self.is_oov_word(l1[pos], s)
                                if cond:
                                    pos_to_word = add_to_dict_set(pos, set(pos_to_candidate_1[pos]), pos_to_word)
                                    if no_repeat:
                                        if pos-1 >= 0 and pos-1 in pos_to_pos_1_2:
                                            if pos_to_pos_1_2[pos] != pos_to_pos_1_2[pos-1]:
                                                pos_to_word = add_to_dict_set(pos, pos_to_word_1_2[pos], pos_to_word)
                                        else:
                                            pos_to_word = add_to_dict_set(pos, pos_to_word_1_2[pos], pos_to_word)
                                    else:
                                        pos_to_word = add_to_dict_set(pos, pos_to_word_1_2[pos], pos_to_word)
                else:
                    # use 2 (transformer) as the base sequence
                    pos_to_word_2_1, pos_to_pos_2_1 = self.parse_align_output(la, l1, output_tgt_pos=True)
                    l2 = l2.strip().split(" ")
                    l1 = l1.strip().split(" ")
                    pos_pre = -1
                    for pos in range(len(l2)):
                        if pos in pos_to_word_2_1 and pos_to_word_2_1[pos] and pos_to_word_2_1[pos] not in punctuations and pos_to_word_2_1[pos] not in digits and l2[pos] and l2[pos] not in punctuations and l2[pos] not in digits:
                            pos_tgt = pos_to_pos_2_1[pos]
                            if pos_tgt in pos_to_candidate_1:
                                cond = True if not oov_sift else self.is_oov_word(l1[pos_tgt], s)
                                if cond:
                                    pos_to_word = add_to_dict_set(pos, set(pos_to_candidate_1[pos_tgt]), pos_to_word)
                            else:
                                if not only_oov:
                                    if no_repeat:
                                        if pos-1 >= 0 and pos-1 in pos_to_pos_2_1:
                                            if pos_to_pos_2_1[pos] != pos_to_pos_2_1[pos-1]:
                                                pos_to_word = add_to_dict_set(pos, pos_to_word_2_1[pos], pos_to_word)
                                        else:
                                            pos_to_word = add_to_dict_set(pos, pos_to_word_2_1[pos], pos_to_word)
                                    else:
                                        pos_to_word = add_to_dict_set(pos, pos_to_word_2_1[pos], pos_to_word)
                            pos_to_word = add_to_dict_set(pos, l2[pos], pos_to_word)
                pos_to_word = OrderedDict(sorted(pos_to_word.items(), key=lambda t:t[0]))
                oov_candidates_list = []
                for pos,words in pos_to_word.items():
                    oov_candidates_list.append(str(pos)+":"+",".join(list(words)))
                fo.write(" ".join(oov_candidates_list)+"\n")
        print("candidate_list_file created at: "+candidate_list_file_output)


    # write candidate list file from aligned (with ref)
    def write_candidate_list_file_from_aligned(self, onebest_file, oov_file, ref_file, candidate_list_file, oov_sift=True):
        '''
        input: onebest_file, oov_file, ref_file
        output: candidate_list_file
        '''
        align_dir = os.path.join(tmp_dir, "align_dir")
        if not os.path.exists(align_dir):
            os.makedirs(align_dir)
        align_output = os.path.join(tmp_dir, os.path.join(align_dir, "align_output"))
        self.align_oov(onebest_file, ref_file, align_output)

        with open(onebest_file) as fo, open(oov_file) as fv, open(align_output) as ff, open(ref_file) as fr, open(candidate_list_file, 'w') as fc:
            for line in fo:
                line_oov = fv.readline()
                line_ref = fr.readline()
                line_align = ff.readline()
                if line.strip() == "=":
                    assert(line_oov.strip() == "=")
                    assert(line_ref.strip() == "=")
                    fc.write("=\n")
                    continue
                oov_pos = [int(pos) for pos in line_oov.strip().split(" ") if pos != '']
                if oov_pos == []:
                    fc.write("\n")
                    continue
                hyp_tok = line.strip().split(" ")
                    
                pos_to_word = self.parse_align_output(line_align, line_ref)
                oov_candidates_list = []
                for pos in oov_pos:
                    oov_word = hyp_tok[pos]
                    cond = True if not oov_sift else self.is_oov_word(oov_word, s)
                    if not cond: continue
                    if pos in pos_to_word:
                        oov_candidates_list.append(str(pos)+":"+pos_to_word[pos])
                fc.write(" ".join(oov_candidates_list)+"\n")
        print("candidate_list_file created at: "+candidate_list_file)
        

    # write candidate list file from a web lexicon
    def write_candidate_list_file_from_glosbe(self, onebest_file, oov_file, local_glosbe_lexicon, candidate_list_file, oov_sift=True):
        '''
        input: onbest_file, oov_file, local_glosbe_lexicon
        output: candidate_list_file, local_glosbe_lexicon
        '''
        import requests
        import json

        # read local glosbe lexicon (oov:w1,w2,...)
        def read_glosbe_lexicon(local_glosbe_lexicon):
            oov_candidates_dict = {}
            if os.path.exists(local_glosbe_lexicon):
                with open(local_glosbe_lexicon) as f:
                    for line in f:
                        line = line.strip()
                        src_word = line.split(":")[0]
                        tgt_words = line.split(":")[1].split(",")
                        if tgt_words == ['']:
                            tgt_words = []
                        assert(src_word not in oov_candidates_dict)
                        oov_candidates_dict[src_word] = tgt_words
            return oov_candidates_dict

        # write local glosbe lexicon (oov:w1,w2,...)
        def write_glosbe_lexicon(oov_candidates_dict, local_glosbe_lexicon):
            with open(local_glosbe_lexicon, 'w') as fw:
                for src_word in oov_candidates_dict:
                    fw.write(src_word+":"+",".join(oov_candidates_dict[src_word])+"\n")

        # translate word
        def _glosbe_translate(src_word, oov_candidates_dict):
            if src_word not in oov_candidates_dict:
                oov_candidates_list = self.glosbe_translate(src_word)
                if oov_candidates_list != "blocked":
                    oov_candidates_dict[src_word] = oov_candidates_list
                else:
                    return "blocked", oov_candidates_dict
            return oov_candidates_dict[src_word], oov_candidates_dict

        oov_candidates_dict = read_glosbe_lexicon(local_glosbe_lexicon)

        oov_orig_num = 0
        oov_hit_num = 0
        oov_orig_set = set()
        oov_word_set = set()
        ctr = 0
        file_len = get_file_length(onebest_file)
        blocked = False
        with open(onebest_file) as fo, open(oov_file) as fv, open(candidate_list_file, 'w') as fc:
            for line in fo:
                line_oov = fv.readline()
                ctr += 1
                if line.strip() == "=":
                    assert(line_oov.strip() == "=")
                    fc.write("=\n")
                    if ctr%10 == 0:
                        print(str(ctr)+"/"+str(file_len)+" sentences processed.")
                    continue
                oov_pos = [int(pos) for pos in line_oov.strip().split(" ") if pos != '']
                if oov_pos == []:
                    fc.write("\n")
                    if ctr%10 == 0:
                        print(str(ctr)+"/"+str(file_len)+" sentences processed.")
                    continue
                hyp_tok = line.strip().split(" ")
                oov_candidates_list = [] # oov_pos:candidate1,candidate2
                for pos in oov_pos:
                    oov_word = hyp_tok[pos].lower() # lower case
                    cond = True if not oov_sift else self.is_oov_word(oov_word, s)
                    if not cond: continue
                    oov_trans_list, oov_candidates_dict = _glosbe_translate(oov_word, oov_candidates_dict)
                    oov_orig_num += 1
                    oov_orig_set.add(oov_word)
                    if oov_trans_list == "blocked":
                        blocked = True
                        break
                    # only oov words for which glosbe has a translation
                    if len(oov_trans_list) != 0:
                        oov_hit_num += 1
                        oov_word_set.add(oov_word)
                        oov_candidates_list.append(str(pos)+":"+",".join(oov_trans_list))
                if blocked:
                    print("blocked at sentence: "+str(ctr)+", oov: "+str(len(oov_word_set)))
                    break
                fc.write(" ".join(oov_candidates_list)+"\n")
                if ctr%10 == 0:
                    print(str(ctr)+"/"+str(file_len)+" sentences processed.")
        print(str(oov_hit_num)+"/"+str(oov_orig_num)+" oov words have hypotheses.")
        print(str(len(oov_word_set))+"/"+str(len(oov_orig_set))+" unique oov words have hypotheses")
        print("candidate_list_file created at: "+candidate_list_file)

        write_glosbe_lexicon(oov_candidates_dict, local_glosbe_lexicon)


    # write candidate list file from extracted
    def write_candidate_list_file_from_extracted(self, onebest_file, oov_file, oov_candidates_all, candidate_list_file, oov_sift=True):
        '''
        input: onebest_file, oov_file, oov_candidates_all
        output: candidate_list_file (oov_pos:candidate1,candidate2 oov_pos:candidate1,candidate2)
        '''
        #if os.path.exists(candidate_list_file):
        #    return
        oov_orig_num = 0
        oov_hit_num = 0
        oov_orig_set = set()
        oov_word_set = set()
        with open(onebest_file) as fo, open(oov_file) as fv, open(candidate_list_file, 'w') as fc:
            for line in fo:
                line_oov = fv.readline()
                if line_oov.strip() == "=":
                    assert(line.strip() == "=")
                    fc.write("=\n")
                    continue
                oov_pos = [int(pos) for pos in line_oov.strip().split(" ") if pos != '']
                if oov_pos == []:
                    fc.write("\n")
                    continue
                hyp_tok = line.strip().split(" ")
                
                oov_candidates_list = [] # oov_pos:candidate1,candidate2
                for pos in oov_pos:
                    oov_word = hyp_tok[pos].lower() # lower case
                    cond = True if not oov_sift else self.is_oov_word(oov_word, s)
                    if not cond: continue
                    oov_orig_num += 1
                    oov_orig_set.add(oov_word)
                    # only write down oov words for which there exist candidates
                    if oov_word in oov_candidates_all:
                        oov_hit_num += 1
                        oov_word_set.add(oov_word)
                        oov_candidates_list.append(str(pos)+":"+",".join(oov_candidates_all[oov_word]))
                fc.write(" ".join(oov_candidates_list)+"\n")
        print("{0}/{1}={2} oov words have hypotheses.".format(oov_hit_num, oov_orig_num, oov_hit_num*1.0/oov_orig_num))
        print("{0}/{1}={2} unique oov words have hypotheses".format(len(oov_word_set), len(oov_orig_set), len(oov_word_set)*1.0/len(oov_orig_set)))
        print("candidate_list_file created at: "+candidate_list_file)
    
    def write_candidate_list_file_from_googletranslate(self, onebest_file, oov_file, oov_candidates_all, candidate_list_file):
        return self.write_candidate_list_file_from_extracted(onebest_file, oov_file, oov_candidates_all, candidate_list_file)

    def write_candidate_list_file_from_masterlexicon(self, onebest_file, oov_file, oov_candidates_all, candidate_list_file):
        return self.write_candidate_list_file_from_extracted(onebest_file, oov_file, oov_candidates_all, candidate_list_file)

    def write_candidate_list_file_from_isixml(self, onebest_file, oov_file, oov_candidates_all, candidate_list_file):
        return self.write_candidate_list_file_from_extracted(onebest_file, oov_file, oov_candidates_all, candidate_list_file)

    # write candidate list file from eng_vocab   
    def write_candidate_list_file_from_engvocab(self, onebest_file, oov_file, eng_vocab, candidate_list_file):
        '''
        input: onebest_file, oov_file, eng_vocab
        output: candidate_list_file
        '''
        if not os.path.exists(candidate_list_file):
            candidate_list_file_tmp = candidate_list_file+".tmp"
            # write envocab candidates into file according to the domain-specific language
            if not os.path.exists(candidate_list_file_tmp):
                with open(oov_file) as fo, open(candidate_list_file_tmp, 'w') as fw:
                    for line in fo:
                        if line.strip() == "=":
                            fw.write("=\n")
                            continue
                        l = line.strip().split(' ')
                        if l != ['']:
                            l = [item+":"+",".join(eng_vocab) for item in l]
                            fw.write(" ".join(l)+"\n")
                        else:
                            fw.write("\n")
                print("candidate list file for eng_vocab created at: "+candidate_list_file_tmp)
            else:
                print("candidate list file for eng_vocab exists at: "+candidate_list_file_tmp)

            function_words_file = data_dir+"function_words_file"
            if not os.path.exists(function_words_file):
                with open(function_words_file, 'w') as f:
                    for word in function_words:
                        f.write(word+"\n")

            punctuations_file = data_dir+"punctuations_file"
            if not os.path.exists(punctuations_file):
                with open(punctuations_file, 'w') as f:
                    for punctuation in punctuations:
                        f.write(punctuation+"\n")

            print('--------')
            # compile
            print("cd "+os.path.join(dir_path,"method_pmi2")+"; "+\
                    "javac "+\
                    "-cp "+":".join([".",palmetto_jar,hppc_jar])+" "+\
                    "*.java")
            print('--------')
            # execute
            context_scale = "bp"
            window_mechanism = "boolean_window"
            index_path = index_dir+"wikipedia_"+context_scale
            num_candidates = 20#2   # number of top candidates to select from when rescoring

            batch_mode = 0
            if batch_mode != 0:
                print("cd "+os.path.join(dir_path,"method_pmi2")+"; "+\
                        "java "+\
                        "-cp "+":".join([".",palmetto_jar,hppc_jar])+" "+\
                        "write_candidate_list_from_eng_vocab_sequential "+\
                        " ".join([onebest_file, candidate_list_file_tmp, function_words_file, punctuations_file, index_path, context_scale, window_mechanism, str(num_candidates), candidate_list_file]))
            else:
                print("cd "+os.path.join(dir_path,"method_pmi2")+"; "+\
                        "java "+\
                        "-cp "+":".join([".",palmetto_jar,hppc_jar])+" "+\
                        "compute_pmi "+\
                        " ".join(["collect_pmi", onebest_file, candidate_list_file_tmp, function_words_file, punctuations_file, pmi_mat_dir, context_words_record_file, eng_vocab_file, index_path, context_scale, window_mechanism, str(pmi_mat_capacity)]))
                print("cd "+os.path.join(dir_path,"method_pmi2")+"; "+\
                        "java "+\
                        "-cp "+":".join([".",palmetto_jar,hppc_jar])+" "+\
                        "compute_pmi "+\
                        " ".join(["apply_pmi", onebest_file, candidate_list_file_tmp, function_words_file, punctuations_file, pmi_mat_dir, context_words_record_file, eng_vocab_file, candidate_list_file, str(num_candidates)]))
            print('--------')
        else:
            print("candidate_list_file exists at: "+candidate_list_file)


    ## write candidate list file from extracted and eng_vocab
    #def write_candidate_list_file_from_extracted_eng_vocab(self, candidate_list_file_extracted, candidate_list_file_eng_vocab, candidate_list_file):
    #    if not os.path.exists(candidate_list_file_extracted):
    #        raise Exception("candidate_list_file_extracted doesn't exist at: "+candidate_list_file_extracted)
    #    if not os.path.exists(candidate_list_file_eng_vocab):
    #        raise Exception("candidate_list_file_eng_vocab doesn't exist at: "+candidate_list_file_eng_vocab)
    #    with open(candidate_list_file_extracted) as fe, open(candidate_list_file_eng_vocab) as fv, open(candidate_list_file, 'w') as ff:
    #        for l_extracted in fe:
    #            l_eng_vocab = fv.readline()
    #            if l_eng_vocab == "\n":
    #                assert(l_extracted == "\n")
    #                ff.write("\n")
    #                continue
    #            if l_eng_vocab == "=\n":
    #                assert(l_extracted == "=\n")
    #                ff.write("=\n")
    #                continue
    #            candidate_map = dict()
    #            candidate_list_extracted = self.parse_oov_candidate_line(l_extracted) if l_extracted != "\n" else dict()
    #            candidate_list_eng_vocab = self.parse_oov_candidate_line(l_eng_vocab)
    #            for pos in candidate_list_extracted:
    #                assert(pos in candidate_list_eng_vocab)
    #                candidate_map[pos] = candidate_list_extracted[pos]
    #            for pos in candidate_list_eng_vocab:
    #                if pos not in candidate_map:
    #                    candidate_map[pos] = candidate_list_eng_vocab[pos]
    #                else:
    #                    candidate_map[pos] |= candidate_list_eng_vocab[pos]
    #            for pos in candidate_map:
    #                assert(candidate_map[pos])
    #            candidate_list = [pos+":"+",".join(candidate_map[pos]) for pos in candidate_map]
    #            ff.write(" ".join(candidate_list)+"\n")
    #    print("candidate_list_file_extracted_eng_vocab created at: "+candidate_list_file)


    # write candidate list file from two sources
    def write_candidate_list_file_from_two_sources(self, candidate_list_file1, candidate_list_file2, candidate_list_file, underuse_projection=False):
        with open(candidate_list_file1) as f1, open(candidate_list_file2) as f2, open(candidate_list_file, 'w') as fw:
            for l1 in f1:
                l2 = f2.readline()
                #if l1 == "\n":
                #    assert(l2 == "\n")
                #    fw.write("\n")
                #    continue
                if l1 == "\n" and l2 == "\n":
                    fw.write("\n")
                    continue
                if l1 == "=\n":
                    assert(l2 == "=\n")
                    fw.write("=\n")
                    continue
                candidate_list1 = self.parse_oov_candidate_line(l1)
                candidate_list2 = self.parse_oov_candidate_line(l2)
                candidate_map = self.merge_candidate_lists(candidate_list1, candidate_list2, underuse_projection)
                candidate_list = [pos+":"+",".join(candidate_map[pos]) for pos in candidate_map]
                fw.write(" ".join(candidate_list)+"\n")
        print("candidate_list_file_merged created at: "+candidate_list_file)


    # write oov candidates from multiple sources into candidate_list_file
    def write_candidate_list_file_from_multiple_sources(self, candidate_list_file_list, candidate_list_file, underuse_projection=False):
        #if os.path.exists(candidate_list_file):
        #    print("candidate_list_file exists at: "+candidate_list_file)
        #    return
        import shutil

        if underuse_projection:
            idx_extracted = -1 
            for i in range(len(candidate_list_file_list)):
                if "extracted" in candidate_list_file_list[i]:
                    idx_extracted = i
                    break
            if idx_extracted != -1:
                candidate_list_file_extracted = candidate_list_file_list[idx_extracted]
                candidate_list_file_list_2 = [i_name for i_idx,i_name in enumerate(candidate_list_file_list) if i_idx != idx_extracted]
                if len(candidate_list_file_list_2) == 0:
                    shutil.copy(candidate_list_file_list[0], candidate_list_file)
                else:
                    candidate_list_file_lexicon = "/tmp/"+os.path.basename(candidate_list_file)+"."+s+".lexicon"
                    self.write_candidate_list_file_from_multiple_sources(candidate_list_file_2, candidate_list_file_lexicon, underuse_projection=False)
                    self.write_candidate_list_file_from_two_sources(candidate_list_file_lexicon, candidate_list_file_extracted, candidate_list_file, underuse_projection=True)


        num_candidate_list_file = len(candidate_list_file_list)
        if num_candidate_list_file == 0:
            raise Exception("No candidate_list_file in the list!")
        if num_candidate_list_file == 1:
            if candidate_list_file_list[0] != candidate_list_file:
                shutil.copy(candidate_list_file_list[0], candidate_list_file)
        else:
            combined_file = candidate_list_file_list[0]
            combined_file_after = "/tmp/"+os.path.basename(candidate_list_file)+"."+s
            for i in range(1,num_candidate_list_file):
                self.write_candidate_list_file_from_two_sources(candidate_list_file_list[i], combined_file, combined_file_after+str(i))
                combined_file = combined_file_after+str(i)
            shutil.copy(combined_file_after+str(num_candidate_list_file-1), candidate_list_file)
        print("candidate_list_file created at: "+candidate_list_file)

    ############################
    # candidate_sources: list (glosbe, googletranslate, master_lexicon, ...)
    def append_lexicon_word_pairs_to_training(self, 
        candidate_sources, 
        train_src_file_lex=os.path.join(res_dir, "train", ".".join([src_label+raw+"_lex",s,"train",yrv])), 
        train_ref_file_lex=os.path.join(res_dir, "train", ".".join([ref_label+raw+"_lex",t,"train",yrv]))):
        oov_candidates_dict_list = []
        for candidate_source in candidate_sources:
            func = "get_oov_candidates_from_"+candidate_source
            lexicon_file = lexicon_files[candidate_source]
            oov_candidates_dict = getattr(oov_candidates_preprocessing, func)(self, lexicon_file)
            print("candidate_source: "+candidate_source)
            print("--------")
            oov_candidates_dict_list.append(oov_candidates_dict)
        oov_candidates_dict = self.get_oov_candidates_from_multiple_sources(oov_candidates_dict_list)

        lex_dict = dict()
        for dataset in ["dev", "test"]:
            onebest_file = os.path.join(res_dir,dataset,".".join(["onebest"+mt,t,dataset,yrv]))
            oov_pos_file = os.path.join(res_dir,dataset,"oov",".".join(["oov"+mt,t,dataset,yrv]))
            with open(onebest_file) as fo, open(oov_pos_file) as fv:
                for line in fo:
                    line_oov = fv.readline()
                    if line_oov.strip() == "=":
                        assert(line.strip() == "=")
                        continue
                    oov_pos = [int(pos) for pos in line_oov.strip().split(" ") if pos != '']
                    if oov_pos == []:
                        continue
                    hyp_tok = line.strip().split(" ")

                    for pos in oov_pos:
                        oov_word = hyp_tok[pos].lower() # lower case
                        if self.is_oov_word(oov_word, s) and oov_word in oov_candidates_dict:
                            if oov_word not in lex_dict:
                                lex_dict[oov_word] = oov_candidates_dict[oov_word]
                            else:
                                lex_dict[oov_word] |= oov_candidates_dict[oov_word]
        print("{} word pairs added to training data".format(len(lex_dict)))

        #train_src_file_lex = os.path.join(res_dir, "train", ".".join([src_label+raw+"_lex",s,"train",yrv]))
        #train_ref_file_lex = os.path.join(res_dir, "train", ".".join([ref_label+raw+"_lex",t,"train",yrv]))
        with open(train_src_file_lex, "w") as fsl, open(train_ref_file_lex, "w") as frl:
            with open(train_src_file) as fs, open(train_ref_file) as fr:
                for ls in fs:
                    fsl.write(ls)
                    lr = fr.readline()
                    frl.write(lr)
            for src_word,ref_words in lex_dict.items():
                for ref_word in ref_words:
                    fsl.write(src_word+"\n")
                    frl.write(ref_word+"\n")
        print("src_lex: "+train_src_file_lex)
        print("ref_lex: "+train_ref_file_lex)

    ############################
    # datasets: list (dev, test, ...)
    # candidate_sources: list (glosbe, googletranslate, master_lexicon, ...)
    def get_uniq_untranslated_oov(self, datasets, candidate_sources, mt=mt, oov_sift=True, write_to_file=True):
        
        oov_candidates_dict_list = []
        for candidate_source in candidate_sources:
            func = "get_oov_candidates_from_"+candidate_source
            lexicon_file = lexicon_files[candidate_source]
            oov_candidates_dict = getattr(oov_candidates_preprocessing, func)(self, lexicon_file)
            print("candidate_source: "+candidate_source)
            print("size: "+str(len(oov_candidates_dict)))
            print("--------")
            oov_candidates_dict_list.append(oov_candidates_dict)
        oov_candidates_dict = self.get_oov_candidates_from_multiple_sources(oov_candidates_dict_list)

        oov_untranslated_set = set()
        oov_abs_hit_num = 0
        oov_set_hit = set()
        oov_abs_num = 0
        oov_set = set()

        # aggregate oov words across datasets based on their positions
        for dataset in datasets:
            onebest_file = os.path.join(res_dir,dataset,".".join(["onebest"+mt,t,dataset,yrv]))
            oov_pos_file = os.path.join(res_dir,dataset,"oov",".".join(["oov"+mt,t,dataset,yrv]))

            with open(onebest_file) as fo, open(oov_pos_file) as fv:
                for line in fo:
                    line_oov = fv.readline()
                    if line_oov.strip() == "=":
                        assert(line.strip() == "=")
                        continue
                    oov_pos = [int(pos) for pos in line_oov.strip().split(" ") if pos != '']
                    if oov_pos == []:
                        continue
                    hyp_tok = line.strip().split(" ")
                    
                    for pos in oov_pos:
                        oov_word = hyp_tok[pos].lower() # lower case
                        cond = True if not oov_sift else self.is_oov_word(oov_word, s)
                        if cond:
                            oov_abs_num += 1
                            oov_set.add(oov_word)
                            if oov_word.lower() in oov_candidates_dict or oov_word in oov_candidates_dict or (oov_sift is False and not self.is_oov_word(oov_word, s)):
                                oov_abs_hit_num += 1
                                oov_set_hit.add(oov_word)
                            else:
                                oov_untranslated_set.add(oov_word)
                        
        print("{0}/{1}={2} oov words have hypotheses.".format(oov_abs_hit_num, oov_abs_num, oov_abs_hit_num*1.0/oov_abs_num))
        print("{0}/{1}={2} unique oov words have hypotheses".format(len(oov_set_hit),len(oov_set),len(oov_set_hit)*1.0/len(oov_set)))
        print(str(len(oov_untranslated_set))+" unique oov words don't have hypotheses")

        if write_to_file:
            uniq_oov_dir = os.path.join(res_dir, "oov")
            if not os.path.exists(uniq_oov_dir):
                os.makedirs(uniq_oov_dir)
            suffix = "" if oov_sift else "_false"
            oov_untranslated_file = os.path.join(uniq_oov_dir,".".join(["-".join(sorted(datasets))+"_"+"-".join(sorted(candidate_sources))+mt+suffix,t,yrv]))

            with open(oov_untranslated_file, 'w') as fw:
                for w in oov_untranslated_set:
                    fw.write(w+"\n")
            print("oov_untranslated_file created at: "+oov_untranslated_file)

    ############################
    def get_statistics(self):
        ## number of tokens 
        #assert(os.path.exists(train_ref_file))
        #num_doc = 0
        #num_sent = 0
        #num_tok = 0
        #with open(train_ref_file) as f:
        #    for l in f:
        #        if l.strip() != "=":
        #            num_sent += 1
        #            num_tok += len(l.strip().split(" "))
        #        else:
        #            num_doc += 1
        #print("number of documents in train target: {}".format(num_doc))
        #print("number of sentences in train target: {}".format(num_sent))
        #print("number of tokens in train target: {}".format(num_tok))
        #assert(os.path.exists(dev_ref_file))
        #num_doc = 0
        #num_sent = 0
        #num_tok = 0
        #with open(dev_ref_file) as f:
        #    for l in f:
        #        if l.strip() != "=":
        #            num_sent += 1
        #            num_tok += len(l.strip().split(" "))
        #        else:
        #            num_doc += 1
        #print("number of documents in dev target: {}".format(num_doc))
        #print("number of sentences in dev target: {}".format(num_sent))
        #print("number of tokens in dev target: {}".format(num_tok))
        #assert(os.path.exists(test_ref_file))
        #num_doc = 0
        #num_sent = 0
        #num_tok = 0
        #with open(test_ref_file) as f:
        #    for l in f:
        #        if l.strip() != "=":
        #            num_sent += 1
        #            num_tok += len(l.strip().split(" "))
        #        else:
        #            num_doc += 1
        #print("number of documents in test target: {}".format(num_doc))
        #print("number of sentences in test target: {}".format(num_sent))
        #print("number of tokens in test target: {}".format(num_tok))

        # vocab number in training data
        assert(os.path.exists(train_src_file))
        train_src_set = set()
        ctr = 0
        with open(train_src_file) as f:
            for l in f:
                l = l.strip().split(" ")
                for tok in l:
                    train_src_set.add(tok)
                    ctr += 1
        print("(type-wise) vocabulary size of "+s+" in "+st+" is: {}".format(len(train_src_set)))
        print("(token-wise) vocabulary size of "+s+" in "+st+" is: {}".format(ctr))

        # type and token coverage of oov
        assert(os.path.exists(test_src_file))
        test_src_set = set()
        test_src_hit_set = set()
        ctr = 0
        ctr_hit = 0
        with open(test_src_file) as f:
            for l in f:
                l = l.strip().split(" ")
                for tok in l:
                    if tok in train_src_set or tok.lower() in train_src_set:
                        test_src_hit_set.add(tok)
                        ctr_hit += 1
                    test_src_set.add(tok)
                    ctr += 1
        print("(type-wise) {0}/{1}={2} oov words are not covered by the training set.".format(len(test_src_set)-len(test_src_hit_set), len(test_src_set), (len(test_src_set)-len(test_src_hit_set))*1.0/len(test_src_set)))
        print("(token-wise) {0}/{1}={2} oov words are not covered by the training set.".format(ctr-ctr_hit, ctr, (ctr-ctr_hit)*1.0/ctr))

        # lexicon coverage
        dataset = "test"
        candidate_source = ["googletranslate", "masterlexicon", "extracted"]
        candidate_list_file = os.path.join(res_dir,dataset,"oov",".".join(["-".join(sorted(candidate_source)),t,dataset,yrv]))
        assert(os.path.exists(candidate_list_file))
        

        # accuracy of lexicons against aligned reference
        dataset = "test"
        candidate_source = ["googletranslate", "masterlexicon", "extracted"]
        candidate_list_file = os.path.join(res_dir,dataset,"oov",".".join(["-".join(sorted(candidate_source)),t,dataset,yrv]))
        assert(os.path.exists(candidate_list_file))
        candidate_list_file_aligned = os.path.join(res_dir,dataset,"oov",".".join(["aligned",t,dataset,yrv]))
        assert(os.path.exists(candidate_list_file_aligned))
        ctr = 0
        ctr_hit = 0
        ctr_set = set()
        ctr_slot = 0
        ctr_candidate = 0
        with open(candidate_list_file) as fhyp, open(candidate_list_file_aligned) as fref:
            for lref in fref:
                lhyp = fhyp.readline()
                if lref == "=\n":
                    assert(lhyp == "=\n")
                    continue
                #pos_to_word_ref = self.parse_oov_candidate_line(lref)
                pos_to_word_ref = {int(item.split(":")[0]):item.split(":")[1] for item in lref.strip().split(" ") if item != ""}
                pos_to_word_hyp = self.parse_oov_candidate_line(lhyp)
                for pos_hyp in pos_to_word_hyp:
                    if len(pos_to_word_hyp[pos_hyp]) >= 1 and pos_to_word_hyp[pos_hyp] != {""}:
                        ctr_slot += 1
                        ctr_candidate += len(pos_to_word_hyp[pos_hyp])
                for pos_ref in pos_to_word_ref:
                    if pos_ref in pos_to_word_hyp:
                        if pos_to_word_ref[pos_ref].lower() in pos_to_word_hyp[pos_ref] or pos_to_word_ref[pos_ref] in pos_to_word_hyp[pos_ref]:
                            ctr_hit += 1
                        ctr += 1
                        ctr_set.add(pos_to_word_ref[pos_ref].lower())
        print("the accuracy of "+str(candidate_source)+" is: {0}/{1}={2}".format(ctr_hit, len(ctr_set), ctr_hit*1.0/len(ctr_set)))
        print("average number of candidates per slot: {0}/{1}={2}".format(ctr_candidate, ctr_slot, ctr_candidate*1.0/ctr_slot))
    
    ############################
    # dataset: dev, test, ...
    # candidate_source: string, list
    def init(self, dataset, candidate_source, mt=mt):
        # input
        ref_file = os.path.join(res_dir,dataset,".".join([ref_label+raw,t,dataset,yrv]))
        onebest_file = os.path.join(res_dir,dataset,".".join(["onebest"+mt,t,dataset,yrv]))
        oov_pos_file = os.path.join(res_dir,dataset,"oov",".".join(["oov"+mt,t,dataset,yrv]))

        # output: pos:c1,c2 pos:c1,c2
        if isinstance(candidate_source, str):
            candidate_list_file = os.path.join(res_dir,dataset,"oov",".".join([candidate_source,t,dataset,yrv]))
            if os.path.exists(candidate_list_file):
                return None, ref_file, onebest_file, None, oov_pos_file, candidate_list_file
        
            if candidate_source == "glosbe":
                local_glosbe_lexicon = lexicon_files[candidate_source]
                self.write_candidate_list_file_from_glosbe(onebest_file, oov_pos_file, local_glosbe_lexicon, candidate_list_file)
            elif candidate_source == "googletranslate":
                local_google_lexicon = lexicon_files[candidate_source]
                oov_candidates_dict = self.get_oov_candidates_from_googletranslate(local_google_lexicon)
                self.write_candidate_list_file_from_googletranslate(onebest_file, oov_pos_file, oov_candidates_dict, candidate_list_file)
            elif candidate_source == "masterlexicon":
                lex = lexicon_files[candidate_source]
                oov_candidates_dict = self.get_oov_candidates_from_masterlexicon(lex)
                self.write_candidate_list_file_from_masterlexicon(onebest_file, oov_pos_file, oov_candidates_dict, candidate_list_file)
            elif candidate_source == "extracted":
                lex = lexicon_files[candidate_source]
                oov_candidates_dict = self.get_oov_candidates_from_extracted(lex)
                self.write_candidate_list_file_from_extracted(onebest_file, oov_pos_file, oov_candidates_dict, candidate_list_file)
            elif candidate_source == "isixml":
                lexicon_xml = lexicon_files[candidate_source]
                oov_candidates_dict = self.get_oov_candidates_from_isixml(lexicon_xml)
                self.write_candidate_list_file_from_isixml(onebest_file, oov_pos_file, oov_candidates_dict, candidate_list_file)
            elif candidate_source == "engvocab":
                eng_vocab_f = lexicon_files[candidate_source]
                eng_vocab = self.get_engvocab(eng_vocab_f)
                self.write_candidate_list_file_from_engvocab(onebest_file, oov_pos_file, eng_vocab, candidate_list_file)
            #elif candidate_source == "extracted_eng_vocab":
            #    # extracted
            #    candidate_list_file_extracted = res_dir+dataset+"/"+"oov/"+".".join(["extracted",t,dataset,yrv])
            #    self.init(dataset, "extracted")
            #    # eng_vocab
            #    candidate_list_file_eng_vocab = res_dir+dataset+"/"+"oov/"+".".join(["eng_vocab",t,dataset,yrv])
            #    if not os.path.exists(candidate_list_file_eng_vocab):
            #        self.init(dataset, "engvocab")
            #    else:
            #        print("candidate_list_file_eng_vocab exists at: "+candidate_list_file_eng_vocab)
            #    self.write_candidate_list_file_from_extracted_eng_vocab(candidate_list_file_extracted, candidate_list_file_eng_vocab, candidate_list_file)
            elif candidate_source == "aligned":
                self.write_candidate_list_file_from_aligned(onebest_file, oov_pos_file, ref_file, candidate_list_file)
            #elif candidate_source == "aligned_extracted":
            #    candidate_list_file_aligned = os.path.join(res_dir, dataset, "oov", ".".join(["aligned",t,dataset,yrv]))
            #    self.init(dataset, "aligned")
            #    candidate_list_file_extracted = os.path.join(res_dir, dataset, "oov", ".".join(["extracted",t,dataset,yrv]))
            #    self.init(dataset, "extracted")
            #    self.write_candidate_list_file_from_two_sources(candidate_list_file_aligned, candidate_list_file_extracted, candidate_list_file)
            elif candidate_source == "alignedhyp":
                raise Exception("\"alignedhyp\" has to be used with other candidate sources")
            else:
                raise Exception("invalid candidate source: "+candidate_source)

        elif isinstance(candidate_source, list):
            underuse_projection = True
            u_p = "-up" if underuse_projection else ""
            if "extracted" in candidate_source and underuse_projection:
                candidate_list_file = os.path.join(res_dir,dataset,"oov",".".join(["-".join(sorted(candidate_source))+u_p,t,dataset,yrv]))
            else:
                candidate_list_file = os.path.join(res_dir,dataset,"oov",".".join(["-".join(sorted(candidate_source)),t,dataset,yrv]))
            if os.path.exists(candidate_list_file):
                return None, ref_file, onebest_file, None, oov_pos_file, candidate_list_file

            if "alignedhyp" in candidate_source:
                candidate_source.remove("alignedhyp")
                _, _, onebest_file_1, _, _, candidate_list_file_1 = self.init(dataset, candidate_source, mt=mt)
                # a transformer output
                method_2 = "t2t_dim512_layer2_lr0.2_dropout0.1_bpe8000"
                onebest_file_2 = os.path.join(res_dir,dataset,".".join(["onebest_"+method_2,t,dataset,yrv]))
                print("onebest_file_1: \n"+onebest_file_1)
                assert(os.path.exists(onebest_file_1))
                print("----")
                print("candidate_list_file_1: \n"+candidate_list_file_1)
                assert(os.path.exists(candidate_list_file_1))
                print("----")
                print("onebest_file_2: \n"+onebest_file_2)
                assert(os.path.exists(onebest_file_2))
                print("----")
                _1_2 = True if s in {"hau","vie"} else False
                only_oov = False
                o_o = "-oo" if only_oov else ""
                no_repeat = False
                n_r = "-nr" if no_repeat else ""
                candidate_source.append("alignedhyp")
        
                if "extracted" in candidate_source and underuse_projection:
                    candidate_list_file_output = os.path.join(res_dir,dataset,"oov",".".join(["-".join(sorted(candidate_source))+"-"+str(_1_2)+o_o+n_r+u_p,t,dataset,yrv]))
                else:
                    candidate_list_file_output = os.path.join(res_dir,dataset,"oov",".".join(["-".join(sorted(candidate_source))+"-"+str(_1_2)+o_o+n_r,t,dataset,yrv]))
                self.write_candidate_list_file_from_alignedhyp(onebest_file_1, candidate_list_file_1, onebest_file_2, candidate_list_file_output, _1_2=_1_2, only_oov=only_oov, no_repeat=no_repeat)
                onebest_file_output = onebest_file_1 if _1_2 else onebest_file_2
                return _, _, onebest_file_output, _, _, candidate_list_file_output

            candidate_list_file_list = []
            for c in candidate_source:
                _, _, _, _, _, candidate_list_file_0 = self.init(dataset, c)
                candidate_list_file_list.append(candidate_list_file_0)
                print("--------")
            self.write_candidate_list_file_from_multiple_sources(candidate_list_file_list, candidate_list_file)
            print("--------")

        #data_in_domain_xml = corpus_dir+".".join([dataset_name,st,dataset,yrv,"xml"])

        return None, ref_file, onebest_file, None, oov_pos_file, candidate_list_file



