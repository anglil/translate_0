import json
import os
import sys
import yaml
from seq2seq.data import vocab
import IPython.display as display

import numpy as np

vis_html = """
  <span style="user-select:none">
    Layer: <select id="layer"></select>
    Attention: <select id="att_type">
      <option value="inp_inp">Input - Input</option>
      <option value="inp_out">Input - Output</option>
      <option value="out_out">Output - Output</option>
    </select>
  </span>
  <div id='vis'></div>
"""

# include the js file for plotting visualization
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
vis_js = open(os.path.join(__location__, 'attn.js')).read()


def _show_attention(att_json):
  display.display(display.HTML(vis_html))
  display.display(display.Javascript('window.attention = %s' % att_json))
  display.display(display.Javascript(vis_js))

def show(inp_text, out_text, enc_atts, dec_atts, encdec_atts, model_dir):
    attention = _get_attention(
        inp_text, 
        out_text, 
        enc_atts,
        dec_atts,
        encdec_atts,
        model_dir)
    att_json = json.dumps(attention)
    _show_attention(att_json)

def _get_attention(inp_text, out_text, enc_atts, dec_atts, encdec_atts, model_dir):
    '''
    [num_layers, batch_size, num_heads, enc/dec_length, enc/dec_length]
    '''
    def get_attentions(get_attention_fn):
        num_layers = len(enc_atts)
        attentions = []
        for i in range(num_layers):
            attentions.append(get_attention_fn(i))
        return attentions

    def get_inp_inp_attention(layer):
        att = np.transpose(enc_atts[layer][0], (0, 2, 1))
        return [ha.T.tolist() for ha in att]

    def get_out_inp_attention(layer):
        att = np.transpose(encdec_atts[layer][0], (0, 2, 1))
        return [ha.T.tolist() for ha in att]

    def get_out_out_attention(layer):
        att = np.transpose(dec_atts[layer][0], (0, 2, 1))
        return [ha.T.tolist() for ha in att]

    
    config_file = os.path.join(model_dir, "reg_config", "config.yml")
    config = yaml.load(open(config_file))
    lexicon_dict_file = config["bilingual_lexicon"]
    train_src = config["train_src"]
    train_tgt = config["train_tgt"]
    train_tgt_token_set = set()
    with open(train_tgt) as f:
        for l in f:
            l = l.strip().split(' ')
            for tok in l:
                train_tgt_token_set.add(tok)

    lexicon_dict = vocab.get_lexicon_dict(lexicon_dict_file)

    if "usealign" not in model_dir:
        inp_text_1d = map_to_lex_cap(inp_text, lexicon_dict, train_tgt_token_set)
    else:
        alignment_dict_ordered = get_alignment_dict_ordered(train_src, train_tgt)
        inp_text_1d = []
        for token in inp_text:
            if token in alignment_dict_ordered:
                tgt_list = alignment_dict_ordered[token]
            elif token in lexicon_dict:
                tgt_list = get_translation_candidates_by_target(lexicon_dict[token], train_tgt_token_set, 4)
            else:
                tgt_list = ["UNK"]
            tgt_ret = pad_to_lex_cap(tgt_list, 4)
            inp_text_1d.extend(tgt_ret)

    attentions = {
        'inp_inp': {
            'att': get_attentions(get_inp_inp_attention),
            'top_text': inp_text,
            'bot_text': inp_text,
        },
        'inp_out': {
            'att': get_attentions(get_out_inp_attention),
            'top_text': inp_text_1d,
            'bot_text': out_text,
        },
        'out_out': {
            'att': get_attentions(get_out_out_attention),
            'top_text': out_text,
            'bot_text': out_text,
        },
    }

    return attentions
    
# --------

def map_to_lex_cap(token_list, lexicon_dict, train_tgt_token_set):
    ret = []
    for token in token_list:
        ret.extend(pad_to_lex_cap(get_translation_candidates_by_target(lexicon_dict[token], train_tgt_token_set, 4), 4) if token in lexicon_dict else ["UNK"]*4)
    return ret

def pad_to_lex_cap(lex_list, lex_cap):
    return np.tile(lex_list, int(lex_cap/len(lex_list))+1)[:lex_cap]

def get_translation_candidates_by_target(translation_candidates, train_tgt_token_set, lex_cap, in_effect=True):
    if len(translation_candidates) < lex_cap or not in_effect:
        return translation_candidates
    ret = []
    for c in translation_candidates:
        if c in train_tgt_token_set:
            ret.append(c)
            if len(ret) == lex_cap:
                return ret
    ret_set = set(ret)
    for c in translation_candidates:
        if c not in ret_set:
            ret.append(c)
            ret_set.add(c)
    return ret

def get_alignment_dict_ordered(train_src, train_tgt):
    from collections import OrderedDict
    def is_punc(mystr):
        import string
        punctuation_set = set(string.punctuation)
        res = True
        for c in mystr:
            if c not in punctuation_set:
                res = False
                break
        return res

    alignment_dict = dict()
    alignment_file = os.path.abspath(os.path.join(train_src, "..", "..", "..", "model", "train", "model", "aligned.grow-diag-final-and"))
    with open(train_src) as f_src, open(train_tgt) as f_tgt, open(alignment_file) as f_a:
        for l_src in f_src:
            l_src = l_src.strip().split(' ')
            l_tgt = f_tgt.readline().strip().split(' ')
            l_a = f_a.readline().strip().split(' ')
            for pair in l_a:
                pos_src = int(pair.split('-')[0])
                pos_tgt = int(pair.split('-')[1])
                assert(pos_src<len(l_src))
                assert(pos_tgt<len(l_tgt))
                src_word = l_src[pos_src].lower() # src_word in lower case
                tgt_word = l_tgt[pos_tgt].lower() # tgt_word in lower case
    
                if is_punc(src_word):
                    alignment_dict[src_word] = {src_word:1}
                elif src_word not in alignment_dict:
                    alignment_dict[src_word] = {tgt_word:1}
                elif tgt_word in alignment_dict[src_word]:
                    alignment_dict[src_word][tgt_word] += 1
                else:
                    alignment_dict[src_word][tgt_word] = 1
    alignment_dict_ordered = dict()
    for k,v in alignment_dict.items():
        alignment_dict_ordered[k] = OrderedDict(sorted(v.items(),key=lambda t:-t[1]))
        alignment_dict_ordered[k] = list(alignment_dict_ordered[k].keys())
    return alignment_dict_ordered       

