#

# pipelined prediction for tagging and parsing

import sys, os, gzip, bz2
import argparse, logging, json

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import numpy as np
import torch

from neuronlp2.io import conllx_data, utils
from neuronlp2.models import BiRecurrentConvBiAffine, BiRecurrentConv
from neuronlp2.io.conllx_data import PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG
from neuronlp2.io.conllx_data import ROOT, ROOT_CHAR

#
def zopen(filename, mode='r', encoding="utf-8"):
    if filename.endswith('.gz'):
        # "t" for text mode of gzip
        return gzip.open(filename, mode+"t", encoding=encoding)
    elif filename.endswith('.bz2'):
        return bz2.open(filename, mode+"t", encoding=encoding)
    else:
        return open(filename, mode, encoding=encoding)

#
def add_entry(x, k, v):
    c = x.get(k, 0)
    x[k] = c + v

def invocab(w, vocab):
    if w in vocab:
        return 1
    elif str.lower(w) in vocab:
        return 1
    else:
        return 0

def get_idx(w, alphabet):
    vv = alphabet.instance2index
    z = vv.get(w, None)
    if z is None:
        return alphabet.get_index(str.lower(w))
    else:
        return z

def format_float(f, digits):
    factor = 10 ** digits
    return int(f*factor)/factor

#
class DataStreamer:
    def __init__(self, args, vocab, logger):
        # configs
        self.len_thresh_min = args.len_thresh_min
        self.len_thresh_max = args.len_thresh_max
        self.oov_thresh = args.oov_thresh
        self.batch_size = args.batch_size
        self.maxi_batch = args.maxi_batch
        self.K = self.batch_size * self.maxi_batch
        #
        self.stats = {}
        self.report_freq = 10000
        self.vocab = vocab
        self.logger = logger

    def printing(self, x):
        self.logger.info(x)

    def printd(self, d):
        for n in sorted(d.keys()):
            self.printing("-- %s: %s"%(n, d[n]))

    # yield (List-instances [str], Torch-Data)
    # todo(warn): out of order!
    def stream(self, fin):
        bsize = self.batch_size
        last_report_sent = 0
        buffers = []
        for line in fin:
            line = line.strip()
            if len(line) > 0:
                tokens = line.split(" ")        # split by single whitespace
                tok_num = len(tokens)
                add_entry(self.stats, "orig_sent", 1)
                if tok_num >= self.len_thresh_min and tok_num <= self.len_thresh_max:
                    add_entry(self.stats, "lent_sent", 1)
                    invoc_num = sum(invocab(t, self.vocab) for t in tokens)
                    oov_rate = 1. - invoc_num/tok_num
                    if oov_rate <= self.oov_thresh:
                        add_entry(self.stats, "oovt_sent", 1)
                        add_entry(self.stats, "oovt_tok", tok_num)
                        add_entry(self.stats, "oovt_tok_invoc", invoc_num)
                        buffers.append(tokens)
            # yield the ones in buffer?
            if len(buffers) >= self.K:
                buffers.sort(key=len)
                sidx = 0
                while sidx < len(buffers):
                    yield buffers[sidx:sidx+bsize]
                    sidx += bsize
                #
                cur_sent = self.stats["oovt_sent"]
                if cur_sent - last_report_sent >= self.report_freq:
                    self.printing("Streamer has yielded %s sentences." % cur_sent)
                    last_report_sent = cur_sent
                buffers.clear()
        #
        sidx = 0
        while sidx < len(buffers):
            yield buffers[sidx:sidx+bsize]
            sidx += bsize
        #
        self.printing("Finished data streamer")
        self.printd(self.stats)

#
def get_logger(name, level=logging.INFO, handler=sys.stderr,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

#
def main():
    #
    args_parser = argparse.ArgumentParser(description='Pipelined tagging and parsing')
    # path
    args_parser.add_argument('--input', type=str)  # None stdin
    args_parser.add_argument('--output', type=str)  # None stdout
    args_parser.add_argument('--tagger_path', help='path for tagger model file.', required=True)
    args_parser.add_argument('--tagger_name', help='name for tagger model file.', required=True)
    args_parser.add_argument('--parser_path', help='path for parser model file.', required=True)
    args_parser.add_argument('--parser_name', help='name for parser model file.', required=True)
    args_parser.add_argument('--gpu', action='store_true', help='Using GPU')
    # input filters
    args_parser.add_argument('--len_thresh_min', type=int, default=5)
    args_parser.add_argument('--len_thresh_max', type=int, default=80)
    args_parser.add_argument('--oov_thresh', type=float, default=0.25)
    #
    args_parser.add_argument('--batch_size', type=int, default=32)
    args_parser.add_argument('--maxi_batch', type=int, default=20, help="Load how many batches at the same time?")
    #
    args_parser.add_argument('--tagger_topk', type=int, default=1)
    args_parser.add_argument('--parser_topk', type=int, default=1)
    args_parser.add_argument('--parser_mst', type=int, default=1)       # Boolean
    # args
    args = args_parser.parse_args()
    use_gpu = args.gpu
    tagger_topk = args.tagger_topk
    parser_topk = args.parser_topk
    parser_mst = args.parser_mst

    # io
    logger = get_logger("Predictor")
    if args.input is None:
        fin = sys.stdin
    else:
        fin = zopen(args.input)
    if args.output is None:
        fout = sys.stdout
    else:
        fout = zopen(args.output, "w")

    # load tagger
    logger.info("Loading models!")
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = \
        conllx_data.create_alphabets(os.path.join(args.tagger_path, 'alphabets'), None)
    tagger_model_fname = os.path.join(args.tagger_path, args.tagger_name)
    with zopen(tagger_model_fname+'.arg.json') as fd:
        arguments = json.load(fd)
        m_args, m_kwargs = arguments['args'], arguments['kwargs']
        tagger_model = BiRecurrentConv(*m_args, **m_kwargs)
        if not use_gpu:
            tagger_model.load_state_dict(torch.load(tagger_model_fname, map_location='cpu'))
        else:
            tagger_model.load_state_dict(torch.load(tagger_model_fname))
    # load parser
    tmps = conllx_data.create_alphabets(os.path.join(args.parser_path, 'alphabets'), None)
    assert tmps == (word_alphabet, char_alphabet, pos_alphabet, type_alphabet), "Unmatched dictionary of tagger and parser"
    parser_model_fname = os.path.join(args.parser_path, args.parser_name)
    with zopen(parser_model_fname+'.arg.json') as fd:
        arguments = json.load(fd)
        m_args, m_kwargs = arguments['args'], arguments['kwargs']
        parser_model = BiRecurrentConvBiAffine(*m_args, **m_kwargs)
        if not use_gpu:
            parser_model.load_state_dict(torch.load(parser_model_fname, map_location='cpu'))
        else:
            parser_model.load_state_dict(torch.load(parser_model_fname))
    #
    if use_gpu:
        tagger_model.cuda()
        parser_model.cuda()
    tagger_model.eval()
    parser_model.eval()

    # decoding
    logger.info("Yes!! Starting predicting!")
    ROOT_WIDX = word_alphabet.get_index(ROOT)
    ROOT_CIDX = char_alphabet.get_index(ROOT_CHAR)
    with torch.no_grad():
        streamer = DataStreamer(args, word_alphabet.instance2index, logger)
        for instances in streamer.stream(fin):      # List of [tok]
            # prepare for tagger/parser
            max_word_length = max(len(z)+1 for z in instances)      # +1 for sym-ROOT
            max_char_length = min(utils.MAX_CHAR_LENGTH, max(len(w) for z in instances for w in z)+utils.NUM_CHAR_PAD)
            batch_size = len(instances)
            # tensify
            wid_inputs = np.empty([batch_size, max_word_length], dtype=np.int64)
            cid_inputs = np.empty([batch_size, max_word_length, max_char_length], dtype=np.int64)
            masks = np.zeros([batch_size, max_word_length], dtype=np.float32)
            lengths = np.empty(batch_size, dtype=np.int64)
            for b, inst in enumerate(instances):
                inst_size = len(inst) + 1       # adding symbolic-ROOT for parsing, remember to slice for tagging
                wids = [ROOT_WIDX] + [get_idx(utils.DIGIT_RE.sub("0", w), word_alphabet) for w in inst]
                cid_seqs = [[ROOT_CIDX]] + [[char_alphabet.get_index(c) for c in w[:utils.MAX_CHAR_LENGTH]] for w in inst]
                # word ids
                wid_inputs[b, :inst_size] = wids
                wid_inputs[b, inst_size:] = PAD_ID_WORD
                for c, cids in enumerate(cid_seqs):
                    cid_inputs[b, c, :len(cids)] = cids
                    cid_inputs[b, c, len(cids):] = PAD_ID_CHAR
                cid_inputs[b, inst_size:, :] = PAD_ID_CHAR
                # masks
                masks[b, :inst_size] = 1.0
                lengths[b] = inst_size
            # =====
            # tagging
            t_word, t_char, t_mask, t_length = torch.from_numpy(wid_inputs), torch.from_numpy(cid_inputs), \
                                               torch.from_numpy(masks), torch.from_numpy(lengths)
            if use_gpu:
                t_word, t_char, t_mask, t_length = t_word.cuda(), t_char.cuda(), t_mask.cuda(), t_length.cuda()
            topk_t_probs, topk_t_preds = tagger_model.predict(t_word[:,1:], t_char[:,1:], mask=t_mask[:,1:], length=t_length-1, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS, topk=tagger_topk)
            # parsing
            pred_pos = topk_t_preds[:,:,0].cpu().numpy()
            pid_inputs = np.pad(pred_pos, ((0,0),(1,0)), 'constant', constant_values=PAD_ID_TAG)
            for b, inst in enumerate(instances):
                inst_size = len(inst) + 1
                pid_inputs[b, inst_size:] = PAD_ID_TAG
            t_tag = torch.from_numpy(pid_inputs)
            if use_gpu:
                t_tag = t_tag.cuda()
            #
            arr_mst_heads, arr_mst_types, heads, head_probs, types, type_probs = parser_model.predict(t_word, t_char, t_tag, mask=t_mask, length=t_length, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS, greedy_topk=parser_topk, mst=parser_mst)
            # =====
            # output
            arr_tag_probs, arr_tags = topk_t_probs.cpu().numpy(), topk_t_preds.cpu().numpy()
            arr_heads, arr_head_probs, arr_types, arr_type_probs = [z.cpu().numpy() for z in (heads, head_probs, types, type_probs)]
            #
            FLOAT_DIGITS = 4
            for b, inst in enumerate(instances):
                inst_size = len(inst) + 1
                ret = {"tokens": inst}
                ret["tags"] = [[pos_alphabet.get_instance(z1) for z1 in z0] for z0 in arr_tags[b][:inst_size-1]]
                ret["tag_probs"] = [[format_float(z1, FLOAT_DIGITS) for z1 in z0] for z0 in arr_tag_probs[b][:inst_size-1]]
                ret["heads"] = [[int(z1) for z1 in z0] for z0 in arr_heads[b][1:inst_size]]
                ret["head_probs"] = [[format_float(z1, FLOAT_DIGITS) for z1 in z0] for z0 in arr_head_probs[b][1:inst_size]]
                ret["types"] = [[[type_alphabet.get_instance(z2) for z2 in z1] for z1 in z0] for z0 in arr_types[b][1:inst_size]]
                ret["type_probs"] = [[[format_float(z2, FLOAT_DIGITS) for z2 in z1] for z1 in z0] for z0 in arr_type_probs[b][1:inst_size]]
                if parser_mst:
                    # only one prediction
                    ret["mst_heads"] = [int(z) for z in arr_mst_heads[b][1:inst_size]]
                    ret["mst_types"] = [type_alphabet.get_instance(z) for z in arr_mst_types[b][1:inst_size]]
                else:
                    ret["mst_heads"] = None
                    ret["mst_types"] = None
                # check length
                for z in ret:
                    if isinstance(ret[z], list):
                        assert len(ret[z]) == inst_size-1
                fout.write(json.dumps(ret)+"\n")
    logger.info("Finished predicting!")


if __name__ == '__main__':
    main()
