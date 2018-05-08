import sys
import argparse

parser = argparse.ArgumentParser(description="this script extracts oov positions and unique oov words from the Moses onebest output and oov word output.")
parser.add_argument('--onebest_file', help='Input: Moses-output onebest file', required=True)
parser.add_argument('--oov_word_file', help='Input: Moses-output oov file', required=True)
parser.add_argument('--oov_pos_file', help='Output: oov position (space delimited) file', required=True)
parser.add_argument('--oov_unique_file', help='Output: unique oov file', required=True)
args = parser.parse_args()

onebest_file = args.onebest_file
oov_word_file = args.oov_word_file
oov_pos_file = args.oov_pos_file
oov_unique_file = args.oov_unique_file

oov_set = set()
with open(onebest_file) as f_onebest, open(oov_word_file) as f_oov, open(oov_pos_file, 'w') as f_pos:
    for line_onebest in f_onebest:
        line_oov = f_oov.readline()
        if line_onebest.strip() == "=":
            f_pos.write("=\n")
            continue
        if line_oov == "\n":
            f_pos.write("\n")
            continue
        l_onebest = line_onebest.strip().split(' ')
        l_oov = line_oov.strip().split(' ')
        oov_pos = []
        for oov in l_oov:
            if oov in l_onebest:
                oov_set.add(oov)
                oov_pos.append(str(l_onebest.index(oov)))
        f_pos.write(' '.join(oov_pos)+"\n")

with open(oov_unique_file, 'w') as f_unique:
    for oov in oov_set:
        f_unique.write(oov+"\n")
