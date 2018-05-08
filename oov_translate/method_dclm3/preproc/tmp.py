
with open("/home/ec2-user/kklab/Projects/lrlp/experiment_2017.07.12.il3-eng.y1r1.v2/translation/eval/oov/eng_vocab.eng.eval.y1r1.v2.inherent") as fin1, open("/home/ec2-user/kklab/Projects/lrlp/experiment_2017.07.12.il3-eng.y1r1.v2/translation/eval/oov/eng_vocab.eng.eval.y1r1.v2.tmp") as fin2, open("/home/ec2-user/kklab/Projects/lrlp/experiment_2017.07.12.il3-eng.y1r1.v2/translation/eval/oov/extracted.eng.eval.y1r1.v2") as fin3, open("/home/ec2-user/kklab/Projects/lrlp/experiment_2017.07.12.il3-eng.y1r1.v2/translation/eval/oov/eng_vocab.eng.eval.y1r1.v2", "w") as fout:
    ctr = 0
    for lin2 in fin2:
        lin2 = lin2.strip()
        lin1 = fin1.readline().strip()
        lin3 = fin3.readline().strip()
        if lin2 == "=":
            assert(lin1 == "=")
            assert(lin3 == "=")
            fout.write('=\n')
        else:
            if lin2 != "" and lin1 != "":
                candidates = lin1.split(' ')[0].split(':')[1].split(',')

                oov_pos = {}
                pairs = lin2.split(' ')
                for pair in pairs:
                    oov_pos[pair.split(':')[0]] = candidates

                pairs = lin3.split(' ')
                if pairs != ['']:
                    for pair in pairs:
                        pos = pair.split(':')[0]
                        candidates = pair.split(':')[1].split(',')
                        assert(pos in oov_pos)
                        oov_pos[pos] += candidates

                output = []
                for pos in oov_pos:
                    output.append(pos+":"+",".join(oov_pos[pos]))
                fout.write(' '.join(output)+"\n")

            else:
                fout.write('\n')


        ctr += 1


