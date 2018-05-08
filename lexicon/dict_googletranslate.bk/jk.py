# get oov candidates from google translation collected by leanne
import os
def get_oov_candidates_from_googletranslate(infile):
    oov_candidates_dict = dict()
    if os.path.exists(infile):
        with open(infile) as f:
            for l in f:
                line = l.strip().split('\t')
                if int(line[2]) > 0:
                    candidates = line[1].split(',')
                    candidates = [candidate.lower() for candidate in candidates if len(candidate.split(' '))==1] # lower c
                    if candidates != []: 
                        oov_word = line[0]
                        oov_candidates_dict[oov_word] = set(candidates)
        print("googletranslate lexicon loaded from: "+infile)
    else:
        print("googletranslate lexicon doesn't exist at: "+infile)
    return oov_candidates_dict

print(get_oov_candidates_from_googletranslate("ben_eng_translations.txt"))
