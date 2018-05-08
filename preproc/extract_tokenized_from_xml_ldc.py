import sys 
import os
import re
import xml.etree.ElementTree as et
import json
import argparse
import random

# always eng lang2 and non-eng lang1

parser = argparse.ArgumentParser(description="This is a parser of the ldc xml file")
parser.add_argument('--lang1_dir', help='Input directory of lang1 files', required=True)
parser.add_argument('--lang2_dir', help='Input directory of lang2 files', required=True)
parser.add_argument('--training_docs', help='Output indices and length of datasets selected as training', required=True)
parser.add_argument('--training_lang1', help='Output training set on lang1 side', required=True)
parser.add_argument('--training_lang2', help='Output training set on lang2 side', required=True)
parser.add_argument('--dev_docs', help='Output indices and length of datasets selected as dev', required=True)
parser.add_argument('--dev_lang1', help='Output dev set on lang1 side', required=True)
parser.add_argument('--dev_lang2', help='Output dev set on lang2 side', required=True)
parser.add_argument('--test_docs', help='Output indices and length of datasets selected as test', required=True)
parser.add_argument('--test_lang1', help='Output test set on lang1 side', required=True)
parser.add_argument('--test_lang2', help='Output test set on lang2 side', required=True)
#parser.add_argument('--xml', help='Input xml file', required=True)
#parser.add_argument('--lang', help="Source language", required=True)
#parser.add_argument('--output', help='Output file', required=True)
args = parser.parse_args()

lang1_dir=args.lang1_dir
if not lang1_dir.endswith('/'):
    lang1_dir += '/'
lang2_dir=args.lang2_dir
if not lang2_dir.endswith('/'):
    lang2_dir += '/'

training_docs=args.training_docs
training_lang1=args.training_lang1
training_lang2=args.training_lang2

dev_docs=args.dev_docs
dev_lang1=args.dev_lang1
dev_lang2=args.dev_lang2

test_docs=args.test_docs
test_lang1=args.test_lang1
test_lang2=args.test_lang2

#xml_file = args.xml
#lang = args.lang
#output_file = args.output

roman_ab = {"som", "yor", "eng", "en", "de"}

def parse_xml(xml_file, output_file=None):
    tree = et.parse(xml_file)
    root = tree.getroot()
    
    if output_file != None:
        fw = open(output_file, 'w')

    text_output = ""
    sent_ctr = 0
    for seg in root.iter('SEG'):
        line = seg.find('ORIGINAL_TEXT').text
        if output_file != None:
            fw.write(line.strip()+"\n")
        text_output += (line.strip()+"\n")
        sent_ctr += 1
    return text_output, sent_ctr


def assign_train_dev_test(lang1_dir, lang2_dir, training_docs, training_lang1, training_lang2, dev_docs, dev_lang1, dev_lang2, test_docs, test_lang1, test_lang2):
    '''
    input: 
        lang1_dir, lang2_dir
    output:
        training_docs, training_lang1, training_lang2,
        dev_docs, dev_lang1, dev_lang2,
        test_docs, test_lang1, test_lang2,
    return:
    '''
    if os.path.exists(training_docs) and os.path.exists(training_lang1) and os.path.exists(training_lang2) \
            and os.path.exists(dev_docs) and os.path.exists(dev_lang1) and os.path.exists(dev_lang2) \
            and os.path.exists(test_docs) and os.path.exists(test_lang1) and os.path.exists(test_lang2):
        print("training_docs exists at: "+training_docs)
        print("training_lang1 exists at: "+training_lang1)
        print("training_lang2 exists at: "+training_lang2)       
        print("dev_docs exists at: "+dev_docs)
        print("dev_lang1 exists at: "+dev_lang1)
        print("dev_lang2 exists at: "+dev_lang2) 
        print("test_docs exists at: "+test_docs)
        print("test_lang1 exists at: "+test_lang1)
        print("test_lang2 exists at: "+test_lang2) 
    else:
        # get file counts
        files_lang1 = os.listdir(lang1_dir)
        files_lang2 = os.listdir(lang2_dir)
        assert(len(files_lang1)==len(files_lang2))

        # check file name alignment between two languages
        files_lang1_tmp = [f.split('.')[0] for f in files_lang1]
        for f in files_lang2:
            assert(f.split('.')[0] in files_lang1_tmp)

        # generate train, dev, test counts
        ctr = len(files_lang1)
        num_test = int(ctr * 0.02)
        num_dev = int(ctr * 0.04)
        num_train = ctr - num_test - num_dev
        
        # assign train, dev, test files
        random.shuffle(files_lang1)
        train_files1 = files_lang1[:num_train]
        dev_files1 = files_lang1[num_train:num_train+num_dev]
        test_files1 = files_lang1[num_train+num_dev:]

        train_files2 = []
        for f in train_files1:
            f_tmp = '.'.join(f.split('.')[:-2])+".eng."+'.'.join(f.split('.')[-2:])
            assert(f_tmp in files_lang2)
            train_files2.append(f_tmp)
        dev_files2 = []
        for f in dev_files1:
            f_tmp = '.'.join(f.split('.')[:-2])+".eng."+'.'.join(f.split('.')[-2:])
            assert(f_tmp in files_lang2)
            dev_files2.append(f_tmp)
        test_files2 = []
        for f in test_files1:
            f_tmp = '.'.join(f.split('.')[:-2])+".eng."+'.'.join(f.split('.')[-2:])
            assert(f_tmp in files_lang2)
            test_files2.append(f_tmp)

        # merge train files and document its statistics
        ctr = 0
        with open(training_docs, 'w') as f, open(training_lang1, 'w') as f_train1, open(training_lang2, 'w') as f_train2:
            for f1 in train_files1:
                f2 = train_files2[ctr]
                train_text1, text_len1 = parse_xml(lang1_dir+f1)
                train_text2, text_len2 = parse_xml(lang2_dir+f2)
                assert(text_len1==text_len2)
                f.write(f1+'\t'+f2+'\t'+str(text_len1)+'\n')
                f_train1.write(train_text1+"=\n")
                f_train2.write(train_text2+"=\n")
                ctr += 1
                if (ctr%500==0):
                    print(str(ctr)+" training docs processed.")
        print("training_docs created at: "+training_docs)
        print("training_lang1 created at: "+training_lang1)
        print("training_lang2 created at: "+training_lang2)

        # merge dev files and document its statistics
        ctr = 0
        with open(dev_docs, 'w') as f, open(dev_lang1, 'w') as f_dev1, open(dev_lang2, 'w') as f_dev2:
            for f1 in dev_files1:
                f2 = dev_files2[ctr]
                dev_text1, text_len1 = parse_xml(lang1_dir+f1)
                dev_text2, text_len2 = parse_xml(lang2_dir+f2)
                assert(text_len1==text_len2)
                f.write(f1+'\t'+f2+'\t'+str(text_len1)+"\n")
                f_dev1.write(dev_text1+"=\n")
                f_dev2.write(dev_text2+"=\n")
                ctr += 1
                if (ctr%500==0):
                    print(str(ctr)+" dev docs processed.")
        print("dev_docs created at: "+dev_docs)
        print("dev_lang1 created at: "+dev_lang1)
        print("dev_lang2 created at: "+dev_lang2)

        # merge test files and document its statistics
        ctr = 0
        with open(test_docs, 'w') as f, open(test_lang1, 'w') as f_test1, open(test_lang2, 'w') as f_test2:
            for f1 in test_files1:
                f2 = test_files2[ctr]
                test_text1, text_len1 = parse_xml(lang1_dir+f1)
                test_text2, text_len2 = parse_xml(lang2_dir+f2)
                assert(text_len1==text_len2)
                f.write(f1+'\t'+f2+'\t'+str(text_len1)+"\n")
                f_test1.write(test_text1+"=\n")
                f_test2.write(test_text2+"=\n")
                ctr += 1
                if (ctr%500==0):
                    print(str(ctr)+" test docs processed.")
        print("test_docs created at: "+test_docs)
        print("test_lang1 created at: "+test_lang1)
        print("test_lang2 created at: "+test_lang2)


assign_train_dev_test(lang1_dir, lang2_dir, \
        training_docs, training_lang1, training_lang2, \
        dev_docs, dev_lang1, dev_lang2, \
        test_docs, test_lang1, test_lang2)
        
