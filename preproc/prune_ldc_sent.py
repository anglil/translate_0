import sys
import os

infile_src = sys.argv[1]
infile_tgt = sys.argv[2]
outfile_src = sys.argv[3]
outfile_tgt = sys.argv[4]

ctr = 0
idx_untranslated = []

#if not os.path.exists(outfile_src) or not os.path.exists(outfile_tgt):
nonsense_ctr = 0
pre_doc_empty = True
with open(infile_src) as f_in_src, open(infile_tgt) as f_in_tgt:
    with open(outfile_src, 'w') as f_out_src, open(outfile_tgt, 'w') as f_out_tgt:
        for l_in_src in f_in_src:
            l_in_tgt = f_in_tgt.readline()
            if '_ _' in l_in_src or '_ _' in l_in_tgt:
                nonsense_ctr += 1
            elif '# untranslated' in l_in_src or '# untranslated' in l_in_tgt:
                nonsense_ctr += 1
                ctr += 1
                idx_untranslated.append(ctr)
            else:
                if l_in_src.strip() == "=":
                    assert(l_in_tgt.strip() == "=")
                    if not pre_doc_empty:
                        f_out_src.write(l_in_src)
                        f_out_tgt.write(l_in_tgt)
                        pre_doc_empty = True
                else:
                    f_out_src.write(l_in_src)
                    f_out_tgt.write(l_in_tgt)
                    pre_doc_empty = False
                ctr += 1
print("src input: "+infile_src)
print("tgt input: "+infile_tgt)
print("src output: "+outfile_src)
print("tgt output: "+outfile_tgt)
print(str(nonsense_ctr)+" lines of nonsense removed.")
#else:
#    print("src output exists at: "+outfile_src)
#    print("tgt output exists at: "+outfile_tgt)

print(ctr)
print(idx_untranslated)
print(len(idx_untranslated))
