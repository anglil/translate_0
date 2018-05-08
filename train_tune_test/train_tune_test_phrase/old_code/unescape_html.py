import html
import sys

'''
to unescape html
'''

esc = sys.argv[1]
unesc = sys.argv[2]

with open(esc) as fr, open(unesc, "w") as fw:
    for line in fr:
        l = line.strip().split(' ')
        l_new = [html.unescape(item) for item in l]
        fw.write(' '.join(l_new)+'\n')
