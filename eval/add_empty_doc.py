import sys
infile_ref = sys.argv[1]
infile_hyp = sys.argv[2]
outfile = sys.argv[3]

with open(infile_ref) as fr, open(infile_hyp) as fh, open(outfile, "w") as fo:
    lr = fr.readline().strip()
    while lr:
        while lr == "=":
            fo.write("=\n")
            lr = fr.readline().strip()
        lh = fh.readline().strip()
        if lh == "=":
            lh = fh.readline().strip()
        if lh != "":
            fo.write(lh+"\n")
        lr = fr.readline().strip()
