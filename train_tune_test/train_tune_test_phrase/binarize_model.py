import sys

model_path = sys.argv[1]
phrase_table_path = sys.argv[2]
lexical_table_path = sys.argv[3]
model_bin_path = sys.argv[4]

with open(model_path) as f, open(model_bin_path, 'w') as fw:
    for line in f:
        l = line.strip().split(' ')
        if l[0] == "PhraseDictionaryMemory":
            l[0] = "PhraseDictionaryCompact"
            if len(l) > 1:
                for i in range(1,len(l)):
                    item = l[i].split('=')
                    if item[0] == "path":
                        l[i] = "path="+phrase_table_path
                        break
        elif l[0] == "LexicalReordering":
            if len(l) > 1:
                for i in range(1, len(l)):
                    item = l[i].split('=')
                    if item[0] == "path":
                        l[i] = "path="+".".join(lexical_table_path.split(".")[:-1])
                        break
        fw.write(" ".join(l)+"\n")
                
