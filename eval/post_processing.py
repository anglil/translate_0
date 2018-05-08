import sys

in_file = sys.argv[1]
out_file = sys.argv[2]

with open(in_file) as fi, open(out_file, 'w') as fo:
    for line in fi:
        l = line.strip().split(' ')
        l_new = [tok for tok in l if tok != '...' and tok != ':::']
        fo.write(' '.join(l_new)+"\n")
