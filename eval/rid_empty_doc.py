import sys
infile = sys.argv[1]
outfile = sys.argv[2]

last_line = None
with open(infile) as fi, open(outfile, "w") as fo:
    for l in fi:
        if l.strip() == "=":
            if last_line is None or last_line == "=":
                continue
            else:
                last_line = "="
                fo.write("=\n")
        else:
            last_line = l
            fo.write(l)
