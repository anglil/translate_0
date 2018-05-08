import re

email_validator = re.compile(r"[^@]+@[^@]+\.[^@]+")
url_validator = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

with open("/home/ec2-user/kklab/data/ELISA/evals/y2/JMIST/elisa.som.package.y2r1.v1/elisa.som-eng.dev.y2r1.v1.som") as f, open("jk", "w") as fw:
    for line in f:
        l = line.strip().split(' ')
        l_no_hashtag = ['' if tok.startswith('#') else tok for tok in l]
        l_no_email = ['' if email_validator.match(tok) else tok for tok in l_no_hashtag]
        l_no_url = ['' if url_validator.match(tok) else tok for tok in l_no_email]
        res = " ".join(l_no_url)
        if res != "":
            fw.write(res+"\n")
        
        
