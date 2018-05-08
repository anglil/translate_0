import sys
import matplotlib.pyplot as plt

# get vocab from training data
def get_vocab_set(source_file):
    vocab_set = set()
    with open(source_file) as f:
        for l in f:
            line = l.strip().split(' ')
            for w in line:
                vocab_set.add(w.lower()) # lower case
    return vocab_set

# get lexicon from a lexicon file
def get_lexicon_dict(bilingual_lexicon_file):
    ctr_line = 0
    lexicon_dict = dict()
    with open(bilingual_lexicon_file) as f:
        for l in f:
            line = l.strip().split('\t')
            words = line[0].lower() # lower case
            translations = line[5].lower() # lower case
            words = words.split(' ')
            translations = translations.split(' ')
            if len(words) == len(translations):
                for i in range(len(words)):
                    word = words[i]
                    translation = translations[i]
                    if translation != "n/a":
                        if word in lexicon_dict:
                            lexicon_dict[word].add(translation)
                        else:
                            lexicon_dict[word] = {translation}
            ctr_line += 1
    return lexicon_dict

def get_lexicon_dict_hist(lexicon_dict, image_file="tmp.png"):
    lexicon_dict_hist = dict()
    lexicon_dict_hist2 = dict()
    for key, value in lexicon_dict.items():
        lexicon_dict_hist[key] = len(value)
    for key, value in lexicon_dict_hist.items():
        if value not in lexicon_dict_hist2:
            lexicon_dict_hist2[value] = 1
        else:
            lexicon_dict_hist2[value] += 1
    print(lexicon_dict_hist2)
    plt.bar(list(lexicon_dict_hist2.keys()), lexicon_dict_hist2.values())
    plt.xlim(0,20)
    plt.xlabel("Number of translations")
    plt.ylabel("Number of source words")
    plt.savefig(image_file)

def get_test_coverage_type_based(test_src_vocab_set, training_src_vocab_set, lexicon_dict):
    overlap_vocab_ctr = 0
    overlap_lexicon_ctr = 0
    overlap_vocab_lexicon_ctr = 0
    for w in test_src_vocab_set:
        if w in lexicon_dict:
            overlap_lexicon_ctr += 1
        if w in training_src_vocab_set:
            overlap_vocab_ctr += 1
        elif w in lexicon_dict:
            overlap_vocab_lexicon_ctr += 1

    training_src_vocab_size = len(training_src_vocab_set)
    test_src_vocab_size = len(test_src_vocab_set)

    print("lexicon size: "+str(len(lexicon_dict)))
    print("training src vocab size: "+str(training_src_vocab_size))
    print("test src vocab set: "+str(test_src_vocab_size))
    print("overlap_vocab_ctr: "+str(overlap_vocab_ctr))
    print("overlap_lexicon_ctr: "+str(overlap_lexicon_ctr))
    print("overlap_vocab_lexicon_ctr: "+str(overlap_vocab_lexicon_ctr))
    print("lexicon coverage: "+str(overlap_lexicon_ctr*1.0/test_src_vocab_size))
    print("vocab coverage: "+str(overlap_vocab_ctr*1.0/test_src_vocab_size))
    print("vocab and lexicon coverage: "+str((overlap_vocab_lexicon_ctr+overlap_vocab_ctr)*1.0/test_src_vocab_size))

def write_dict_to_parallel_files(lexicon_dict, src_file, tgt_file):
    ctr = 0
    with open(src_file, 'w') as fs, open(tgt_file, 'w') as ft:
        for src_word in lexicon_dict:
            tgt_words = lexicon_dict[src_word]
            for tgt_word in tgt_words:
                fs.write(src_word+"\n")
                ft.write(tgt_word+"\n")
                ctr += 1
    print("src_file created at: "+src_file)
    print("tgt_file created at: "+tgt_file)
    print(str(ctr)+" words written to file.")


if __name__ == "__main__":
    train_src = sys.argv[1]
    test_src = sys.argv[2]
    bilingual_lexicon = sys.argv[3]
    src_file_from_lexicon = sys.argv[4]
    tgt_file_from_lexicon = sys.argv[5]

    print("train_src: "+train_src)
    print("test_src: "+test_src)
    print("bilingual_lexicon: "+bilingual_lexicon)

    training_src_vocab_set = get_vocab_set(train_src)
    test_src_vocab_set = get_vocab_set(test_src)
    lexicon_dict = get_lexicon_dict(bilingual_lexicon)
    get_test_coverage_type_based(test_src_vocab_set, training_src_vocab_set, lexicon_dict)
    #write_dict_to_parallel_files(lexicon_dict, src_file_from_lexicon, tgt_file_from_lexicon)

    #get_lexicon_dict_hist(lexicon_dict)
