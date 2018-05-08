import xml.etree.ElementTree as et
import sys
import os.path
import subprocess

def sh(script, stdin=None):
    """Returns (stdout, stderr), raises error on non-zero return code"""
    import subprocess
    # Note: by using a list here (['bash', ...]) you avoid quoting issues, as the 
    # arguments are passed in exactly this order (spaces, quotes, and newlines won't
    # cause problems):
    proc = subprocess.Popen(['bash', '-c', script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise ScriptException(proc.returncode, stdout, stderr, script)
    return stdout.decode('utf-8'), stderr.decode('utf-8')



class ScriptException(Exception):
    def __init__(self, returncode, stdout, stderr, script):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super(ScriptException, self).__init__(stdout.decode('utf-8')+stderr.decode('utf-8'))
        #Exception.__init__()

#def sh(script, stdin=None):
#    """Returns (stdout, stderr), raises error on non-zero return code"""
#    import subprocess
#    # Note: by using a list here (['bash', ...]) you avoid quoting issues, as the 
#    # arguments are passed in exactly this order (spaces, quotes, and newlines won't
#    # cause problems):
#    proc = subprocess.Popen(['bash', '-c', script],
#        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#        stdin=subprocess.PIPE)
#    stdout, stderr = proc.communicate()
#    if proc.returncode:
#        raise ScriptException(proc.returncode, stdout, stderr, script)
#    return stdout, stderr
#
#
#class ScriptException(Exception):
#    def __init__(self, returncode, stdout, stderr, script):
#        self.returncode = returncode
#        self.stdout = stdout
#        self.stderr = stderr
#        Exception.__init__('Error in script')


hyp_data_dir = "/home/ec2-user/kklab/data/ELISA/evals/y1/mt/isi/"
mtsys = ["isi-niu-uroman", "isi-niu-nomorf", "isi-niu-morf", "isi-niu-lexv13-morf-green-uzb5", "isi-niu-lexv13-morf-green-uzb3", "isi-niu-lexv13-morf-green-uzb3-2", "isi-niu-lexv12-morf-green-uzb", "isi-niu-lexv12-morf-green", "isi-niu-lexv09-morf-greenv2", "isi-niu-lexv09-morf-green", "isi-niu-lexv07-morf-green", "isi-niu-lex-nomorf", "isi-niu-lex-morf", "isi-niu-lex-morf-green", "isi-sbmt-v5-uzb-guess", "isi-sbmt-v5-uzb3-oov-guess", "isi-sbmt-v5-uzb3-guess", "isi-sbmt-v5-uzb2-guess", "isi-sbmt-v5-comparelm-guess", "isi-sbmt-v4-guess", "isi-sbmt-v3-guess", "isi-sbmt-v2-guess", "isi-sbmt-v2-guess_firstword", "isi-sbmt-v5-uzb3-uwguess-cased", "isi-sbmt-v5-uzb-rpine", "isi-sbmt-v5-uzb", "isi-sbmt-v5-uzb9", "isi-sbmt-v5-uzb3-oov", "isi-sbmt-v5-uzb3-devdomain", "isi-sbmt-v5-uzb3-comparev1", "isi-sbmt-v5-uzb2", "isi-sbmt-v5-comparelm", "isi-sbmt-v4", "isi-sbmt-v3", "isi-sbmt-v3-dict5-rpine", "isi-sbmt-v3-dict5-rpine-green", "isi-sbmt-v2", "isi-sbmt-lex-cdec", "isi-sbmt-vanilla", "isi-sbmt-v5-uzb3", "sbmt-v5-leidoslarge-dev-rerank", "sbmt-v5-leidoslarge-devdomain-rerank"]
#data = ["test", "dev", "syscomb"]#, "unseq"]
data = ["test"]
v = ["0", "1", "2"]

ref_data_dir = "/home/ec2-user/kklab/data/ELISA/evals/y1/JMIST/elisa.il3.package.y1r1.v2/"
#test=ref_data_dir+"test/elisa.il3-eng.test.y1r1.v2.true.en"
#dev=ref_data_dir+"dev/elisa.il3-eng.dev.y1r1.v2.true.en"
#syscomb=ref_data_dir+"syscomb/elisa.il3-eng.syscomb.y1r1.v2.true.en"

ctr = 0
for i in mtsys:
    for j in data:
        for k in v:
            suffix = ".il3-eng."+j+".y1r1.v"+k+".xml"
            hyp = hyp_data_dir+i+suffix
            hyp_1best = hyp.rsplit('.', 1)[0]+'.1best'

            if os.path.isfile(hyp):                            
                tree = et.parse(hyp)
                root = tree.getroot()
        
                # get the hypothesis
                ctr_sent = 0
                with open(hyp_1best, 'w') as f:
                    for item in root.iter('NBEST'):
                        nbest = item[0][0].text.encode('utf-8')
                        f.write(nbest.strip()+'\n')
                        ctr_sent += 1
                print(hyp_1best.rsplit('/', 1)[1]+', '+str(ctr_sent)+" sentences")
                
                # get the reference
                ref = ref_data_dir+j+"/elisa.il3-eng."+j+".y1r1.v2.true.en"
                print(ref.rsplit('/', 1)[1]+': reference')

                # compute bleu score
                stdout, stderr = sh("/home/ec2-user/kklab/src/mosesdecoder/scripts/generic/multi-bleu.perl -lc " + ref +" < " + hyp_1best)
                print(stdout)

                ctr += 1

print(ctr)

#f_out = f.rsplit('.', 1)[0]+'.1best'
#
#tree = et.parse(f)
#root = tree.getroot()
#
#ctr = 0
#with open(f_out, 'w') as fout:
#    for item in root.iter('TEXT'):
#        fout.write(item.text.encode('utf-8'))
#        ctr += 1
#print(ctr)
