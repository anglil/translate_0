import sys
import os

infile = sys.argv[1]
outfile = sys.argv[2]
bilingual_lexicon_file = sys.argv[3]

# each entry word has a *list* of translations
def get_lexicon_dict(bilingual_lexicon_file):
    '''
    returns a lexicon dict
    '''
    if not os.path.exists(bilingual_lexicon_file):
        raise ValueError("bilingual_lexicon_file doesn't exist at: "+bilingual_lexicon_file)
    ctr_line = 0
    lexicon_dict = dict()
    with open(bilingual_lexicon_file) as f:
        for l in f:
            line = l.strip().split('\t')
            words = line[0].lower() # lower case
            translations = line[5].lower() # lower case
            if len(words.split(' ')) == len(translations.split(' ')):
                words = words.split(' ')
                translations = translations.split(' ')
                for i in range(len(words)):
                    word = words[i]
                    translation = translations[i]
                    if translation != "n/a":
                        if word in lexicon_dict:
                            if translation not in lexicon_dict[word]:
                                lexicon_dict[word].append(translation)
                        else:
                            lexicon_dict[word] = [translation]
            ctr_line += 1
    return lexicon_dict

lexicon_dict = get_lexicon_dict(bilingual_lexicon_file)
with open(infile) as f, open(outfile, 'w') as fw:
    for line in f:
        l = line.strip().split(' ')
        l_new = ["["+"/".join(lexicon_dict[tok])+"]" if tok in lexicon_dict else tok for tok in l]
        fw.write(" ".join(l_new)+"\n")
