import sys

in_file = sys.argv[1]
out_file = sys.argv[2]

with open(in_file) as f, open(out_file, 'w') as fw:
    for line in f:
        fw.write(line.lower())

