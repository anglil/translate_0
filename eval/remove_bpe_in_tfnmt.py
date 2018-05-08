import sys

infile = sys.argv[1]
outfile = sys.argv[2]

with open(infile) as f, open(outfile, 'w') as fw:
    for line in f:
        l = line.strip().split(' ')
        if '</s>' in l:
            l = l[:l.index('</s>')]
        fw.write(' '.join(l)+'\n')
