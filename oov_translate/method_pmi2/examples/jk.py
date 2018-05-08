#with open("candidate_list_file_tmp.tmp2") as f:
#    for l in f:
#        candidates = l.strip().split(' ')[0].split(":")[1].split(",")
#        print(len(candidates))
#        break

function_words = set()
with open('function_words_file') as f:
    for l in f:
        function_word = l.strip()
        function_words.add(function_word)
print(len(function_words))
candidates = []
with open("/home/ec2-user/kklab/data/google-10000-english/google-10000-english.txt") as f, open("../pmi_dir/candidate_words_record_file2", "w") as fw:
    for l in f:
        candidate = l.strip()
        if candidate not in function_words:
            candidates.append(candidate)
            fw.write(candidate+"\n")
print(len(candidates))
