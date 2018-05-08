import sys 
import os
import copy

dir_path = os.path.dirname(os.path.realpath(__file__))+"/"
sys.path.insert(0, dir_path+'../../oov_translate')
from config import *
from utils import *
from oov_candidates_preprocessing import *

def rescore_lattice(onebest_file, candidate_list_file, ref_file, res_file, align=False):
    ctr = 0
    rescore_dir = os.path.join(tmp_dir, "lm_rescore")
    if not os.path.exists(rescore_dir):
        os.makedirs(rescore_dir)

    with open(onebest_file) as fo, open(candidate_list_file) as fc, open(ref_file) as fr, open(res_file, "w") as f:
        for lo in fo:
            lo = lo.strip()
            lc = fc.readline().strip()
            lr = fr.readline().strip()
            if lc == "=" or lc == "":
                f.write(lo+"\n")
            else:
                hyp = lo.split(" ")
                for pairs in lc.split(" "):
                    pos = int(pairs.split(":")[0])
                    words = pairs.split(":")[1].split(",")

                    if align:
                        hyp[pos] = ",".join(words) # special case: when words like 300,000 are being aligned
                        continue


                    tmp_file_hyp = os.path.join(rescore_dir,os.path.basename(res_file)+"_hyp")
                    tmp_file_ref = os.path.join(rescore_dir,os.path.basename(res_file)+"_ref")
                    with open(tmp_file_hyp, "w") as fhyp, open(tmp_file_ref, "w") as fref:
                        for word in words:
                            hyp_sent = copy.deepcopy(hyp)
                            hyp_sent[pos] = word
                            fhyp.write(" ".join(hyp_sent)+"\n")
                            fref.write(lr+"\n")
                    stdout, stderr = sh(sent_bleu+" "+tmp_file_ref+" < "+tmp_file_hyp)
                    bleu_scores = stdout.strip().split('\n')
                    bleu_scores = [float(item) for item in bleu_scores]
                    best_bleu_idx = bleu_scores.index(max(bleu_scores))

                    hyp[pos] = words[best_bleu_idx]
                f.write(" ".join(hyp)+"\n")

            ctr += 1
            if ctr % 200 == 0:
                print("{} sentences processed".format(ctr))

if __name__ == "__main__":
    ocp = oov_candidates_preprocessing()

    for dataset in ["test"]:
        for candidate_source in [["extracted"],["masterlexicon","extracted"],["googletranslate"],["masterlexicon", "googletranslate"],["masterlexicon", "googletranslate", "alignedhyp"]]:#["masterlexicon","aligned","alignedhyp"]]:#,["extracted", "masterlexicon", "aligned","alignedhyp"],["googletranslate", "extracted", "alignedhyp"], ["masterlexicon", "googletranslate", "extracted", "alignedhyp"], ["masterlexicon", "googletranslate", "extracted", "aligned", "alignedhyp"]]:
            # true: based in pbmt, aligned with transformer
            # false: based in transformer, aligned with pbmt
            #_1_2 = True
            #only_oov = False
            #o_o = "-oo" if only_oov else ""
            #no_repeat = False
            #n_r = "-nr" if no_repeat else ""
            #underuse_projection = False
            #u_p = "-up" if underuse_projection else ""

            #if "extracted" in candidate_source and underuse_projection:
            #    prefix = "-".join(sorted(candidate_source)) if "alignedhyp" not in candidate_source else "-".join(sorted(candidate_source))+"-"+str(_1_2)+o_o+n_r+u_p
            #else:
            #    prefix = "-".join(sorted(candidate_source)) if "alignedhyp" not in candidate_source else "-".join(sorted(candidate_source))+"-"+str(_1_2)+o_o+n_r
            #if "False" in prefix:
            #    mt = "_t2t_dim512_layer2_lr0.2_dropout0.1_bpe8000"
            _, _, onebest_file, _, _, candidate_list_file = ocp.init(dataset, candidate_source)
            prefix = os.path.basename(candidate_list_file).split(".")[0]
            if "False" in prefix:
                mt = "_t2t_dim512_layer2_lr0.2_dropout0.1_bpe8000"

            #onebest_file = os.path.join(res_dir,dataset,".".join(["onebest"+mt,t,dataset,yrv]))
            #print("onebest_file: "+onebest_file)
            #assert(os.path.exists(onebest_file))
            #print("----")

            #candidate_list_file = os.path.join(res_dir,dataset,"oov",".".join([prefix,t,dataset,yrv]))
            #print("candidate_list_file: "+candidate_list_file)
            #assert(os.path.exists(candidate_list_file))
            #print("----")

            print(dataset)
            print(candidate_source)
            print("----")

            ref_file = os.path.join(res_dir,dataset,".".join([ref_label+raw,t,dataset,yrv]))
            print("ref_file: "+ref_file)
            assert(os.path.exists(ref_file))
            print("----")

            res_attr = "_".join(["onebest"+mt,prefix,"sentbleu"])
            res_file = os.path.join(res_dir,dataset,".".join([res_attr,t,dataset,yrv]))
            print("res_file: "+res_file)
            align = True if candidate_source == ["aligned"] else False

            if not os.path.exists(res_file):
                rescore_lattice(
                    onebest_file, 
                    candidate_list_file,
                    ref_file,
                    res_file,
                    align=align)
                print("res_file created at: "+res_file)
            else:
                print("res_file exists at: "+res_file)

            stdout, _ = sh(bleu_getter+" -lc "+ref_file+" < "+res_file)
            print(stdout)
