import xml.etree.ElementTree as et
import sys
import re

########
# extract untokenized source and target data (train, dev, test, syscomb, etc.) from xml
########

### test, dev, etc.
#dataset = sys.argv[1]
prefix = sys.argv[1]

#y='y2'
#r='r1'
#v='v1'
#s='amh'
#t='eng'
s = sys.argv[2]
t = sys.argv[3]

### remain, replace, remove
special_char = sys.argv[4]
print("special_char: "+special_char)

#data_dir = '/home/ec2-user/kklab/data/ELISA/evals/'+y+'/elisa.'+s+'.package.'+y+r+'.'+v+'/'
#prefix = data_dir + '.'.join(['elisa', s+'-'+t, dataset, y+r, v])
xml_file = prefix+'.xml'

### email validator
email_validator = re.compile(r"[^@]+@[^@]+\.[^@]+")
### url validator
url_validator = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

tree = et.parse(xml_file)
root = tree.getroot()

sent_ctr = 0
doc_ctr = 0

with open(prefix+'.'+s, 'w') as fs, open(prefix+'.'+t, 'w') as ft:
    for doc in root.findall('DOCUMENT'):
    
        for seg in doc.findall('SEGMENT'):
            null_field = 0

            orig_raw_source = seg.find('SOURCE').find('ORIG_RAW_SOURCE')
            if orig_raw_source != None:
                orig_raw_source = orig_raw_source.text
            else:
                null_field = 1

            orig_raw_target = seg.find('TARGET').find('ORIG_RAW_TARGET')
            if orig_raw_target != None:
                orig_raw_target = orig_raw_target.text
            else:
                null_field = 1

            if null_field == 0:

                if special_char == "remain":
                    fs.write(orig_raw_source.strip()+"\n")
                    ft.write(orig_raw_target.strip()+"\n")
                    sent_ctr += 1
                    continue

                l_orig_raw_source = orig_raw_source.strip().split(' ')
                l_orig_raw_target = orig_raw_target.strip().split(' ')

                if special_char == "replace":
                    for i in range(len(l_orig_raw_source)):
                        tok = l_orig_raw_source[i]
                        if tok.startswith('#'):
                            l_orig_raw_source[i] = 'hashtag'
                        elif tok.startswith('@'):
                            l_orig_raw_source[i] = 'twitter_account'
                        elif email_validator.match(tok):
                            l_orig_raw_source[i] = 'email_address'
                        elif url_validator.match(tok):
                            l_orig_raw_source[i] = 'url'


                    
                    for i in range(len(l_orig_raw_target)):
                        tok = l_orig_raw_target[i]
                        if tok.startswith('#'):
                            l_orig_raw_target[i] = 'hashtag'
                        elif tok.startswith('@'):
                            l_orig_raw_target[i] = 'twitter_account'
                        elif email_validator.match(tok):
                            l_orig_raw_target[i] = 'email_address'
                        elif url_validator.match(tok):
                            l_orig_raw_target[i] = 'url'

                    fs.write(' '.join(l_orig_raw_source)+'\n')
                    ft.write(' '.join(l_orig_raw_target)+'\n')
                    sent_ctr += 1
                    continue

                if special_char == "remove":
                    remaining_pos_source = []
                    for i in range(len(l_orig_raw_source)):
                        tok = l_orig_raw_source[i]
                        if (not tok.startswith('#')) and (not tok.startswith('@')) and (not email_validator.match(tok)) and (not url_validator.match(tok)):
                            remaining_pos_source.append(i)

                    remaining_pos_target = []    
                    for i in range(len(l_orig_raw_target)):
                        tok = l_orig_raw_target[i]
                        if (not tok.startswith('#')) and (not tok.startswith('@')) and (not email_validator.match(tok)) and (not url_validator.match(tok)):
                            remaining_pos_target.append(i)

                    if remaining_pos_source != [] and remaining_pos_target != []:
                        orig_raw_source_res = [l_orig_raw_source[j] for j in remaining_pos_source]
                        orig_raw_target_res = [l_orig_raw_target[j] for j in remaining_pos_target]
                        fs.write(' '.join(orig_raw_source_res)+'\n')
                        ft.write(' '.join(orig_raw_target_res)+'\n')
                        sent_ctr += 1
                        continue


            else:
                print("empty entry!")

        doc_ctr += 1

print("doc_ctr: "+str(doc_ctr))
print("sent_ctr: "+str(sent_ctr))

#with open(prefix+'.'+s, 'w') as f:
#    for item in root.iter('ORIG_RAW_SOURCE'):
#        toks = [tok for tok in item.text.strip().split(' ')]
#        for i in range(len(toks)):
#            tok = toks[i]
#            if (tok.startswith('#')) or (email_validator.match(tok)) or (url_validator.match(tok)):
#                toks[i] = ''
#        f.write(' '.join(toks))
#        f.write('\n')
#
##with open(prefix+'.'+s+'.pos', 'w') as f:
##    for item in root.iter('LRLP_POSTAG_SOURCE'):
##        f.write(item.text)
##        f.write('\n')
#
#with open(prefix+'.'+t, 'w') as f:
#    for item in root.iter('ORIG_RAW_TARGET'):
#        f.write(item.text)
#        f.write('\n')
#
#with open(prefix+'.'+t+'.pos', 'w') as f:
#    for item in root.iter('LRLP_POSTAG_TARGET'):
#        f.write(item.text)
#        f.write('\n')

