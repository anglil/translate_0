import xml.etree.ElementTree as et
import sys 

### prefix of the file name to parse
prefix = "/home/ec2-user/kklab/data/ELISA/evals/y1/JMIST/elisa.il3.package.y1r1.v2/elisa.il3-eng.train.y1r1.v2"#sys.argv[1]

#y='y2'
#r='r1'
#v='v1'
#s='amh'
#t='eng'
s = "il3"#sys.argv[2]
t = "eng"#sys.argv[3]

xml_file = prefix+'.xml'

tree = et.parse(xml_file)
root = tree.getroot()

doc_ctr = 0
seg_ctr_list = []
for doc in root.findall('DOCUMENT'):
    seg_ctr = 0
    for seg in doc.findall('SEGMENT'):
        #print(seg.find('SOURCE').find('ORIG_RAW_SOURCE').text)
        #print(seg.find('TARGET').find('ORIG_RAW_TARGET').text)
        seg_ctr += 1
    doc_ctr += 1
    seg_ctr_list.append(seg_ctr)
print(doc_ctr)
print(seg_ctr_list)
