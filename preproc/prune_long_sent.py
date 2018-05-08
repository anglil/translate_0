import argparse
import os 

parser = argparse.ArgumentParser(description="This script throws away long sentences in source and target files for MT")
parser.add_argument('--source_input', help="Input source file", required=True)
parser.add_argument('--target_input', help="Input target file", required=True)
parser.add_argument('--max_len', help="Input max sentence length")
parser.add_argument('--source_output', help="Output source file", required=True)
parser.add_argument('--target_output', help="Output source file", required=True)
args = parser.parse_args()

source_input_file = args.source_input
target_input_file = args.target_input
max_sent_len = int(args.max_len) if args.max_len != None else 100
source_output_file = args.source_output
target_output_file = args.target_output

if not os.path.exists(source_output_file) or not os.path.exists(target_output_file):
    ctr = 0
    with open(source_input_file) as fsi, open(target_input_file) as fti, open(source_output_file, 'w') as fso, open(target_output_file, 'w') as fto:
        for lsi in fsi:
            lti = fti.readline()
            if len(lsi.split(' ')) <= max_sent_len and len(lti.split(' ')) <= max_sent_len:
                fso.write(lsi)
                fto.write(lti)
            ctr += 1
            if ctr%100000 == 0:
                print(str(ctr)+" sentences pruned.")
    print("pruned files created at "+source_output_file+" and "+target_output_file)
else:
    print("pruned files exist at "+source_output_file+" and "+target_output_file)

