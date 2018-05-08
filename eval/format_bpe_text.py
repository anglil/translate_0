import sys

infile = sys.argv[1]
outfile = sys.argv[2]

with open(infile) as f, open(outfile, 'w') as fw:
    delimiter="@@"
    for l in f:
        words = []
        symbols = l.strip().split(' ')
        word = ""
        if isinstance(symbols, str):
            symbols = symbols.encode()
        delimiter_len = len(delimiter)
        for symbol in symbols:
            if len(symbol) >= delimiter_len and symbol[-delimiter_len:] == delimiter:
                word += symbol[:-delimiter_len]
            else:  # end of a word
                word += symbol
                words.append(word)
                word = ""
        fw.write(" ".join(words)+'\n')

