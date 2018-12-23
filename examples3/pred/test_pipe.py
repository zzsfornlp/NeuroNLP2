#

# testing the "pipe_tp.py"

import sys, os, gzip, bz2
import argparse, logging, json

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

#
def add_entry(x, k, v):
    c = x.get(k, 0)
    x[k] = c + v

# iter conllu file
def iter_file(filename):
    with open(filename, 'r') as file:
        ret = {"len": 0, "word": [], "pos": [], "type": [], "head": []}
        for line in file:
            line = line.strip()
            # yield and reset
            if len(line) == 0:
                if ret["len"] > 0:
                    yield ret
                ret = {"len": 0, "word": [], "pos": [], "type": [], "head": []}
            else:
                fields = line.split('\t')
                try:
                    z = int(fields[0])
                except:
                    continue
                ret["len"] += 1
                ret["word"].append(fields[1])
                ret["pos"].append(fields[3])
                ret["head"].append(int(fields[6]))
                ret["type"].append(fields[7])
        if ret["len"] > 0:
            yield ret

def main(args):
    conllu_file, tagger_path, parser_path, gpu_num = args
    gpu_num = int(gpu_num)
    if gpu_num > 0:
        gpu_flag = "--gpu"
    else:
        gpu_flag = ""
    #
    instances = list(iter_file(conllu_file))
    with open("tmp.in", "w") as fd:
        for inst in instances:
            fd.write(" ".join(inst["word"])+"\n")
    tp_file = os.path.join(os.path.dirname(__file__), "pipe_tp.py")
    py_path = os.path.join(os.path.dirname(__file__), "..", "..")
    CMD = "PYTHONPATH=%s CUDA_VISIBLE_DEVICES=%s python3 %s --input tmp.in --output tmp.out --tagger_path %s --tagger_name network.pt --parser_path %s --parser_name network.pt --len_thresh_min 0 --len_thresh_max 1000 --oov_thresh 1.0 %s" % (py_path, gpu_num, tp_file, tagger_path, parser_path, gpu_flag)
    print("Running " + CMD)
    os.system(CMD)
    # eval
    with open("tmp.out") as fd:
        preds = [json.loads(z) for z in fd]
    #
    assert len(instances) == len(preds)
    stats = {}
    instances_map = {"".join(z["word"]):z for z in instances}
    for p in preds:
        key = "".join(p["tokens"])
        if key not in instances_map:
            print("!!Warn, sentence not found (maybe because of tokenization!)")
            continue
        g = instances_map[key]
        for kk in ["tokens", "tags", "heads", "types", "mst_heads", "mst_types"]:
            assert len(g["word"]) == len(p[kk])
        add_entry(stats, "sent", 1)
        for i, g_pos in enumerate(g["pos"]):
            g_head = g["head"][i]
            g_type = g["type"][i]
            #
            tok_prefixes = ["tok_"]
            if g_pos not in ["PUNCT", "SYM"]:
                tok_prefixes.append("tok_np_")
            for prefix in tok_prefixes:
                add_entry(stats, prefix, 1)
                if p["tags"][i][0] == g_pos:
                    add_entry(stats, prefix+"tag_corr", 1)
                if p["heads"][i][0] == g_head:
                    add_entry(stats, prefix+"guas_corr", 1)
                    if p["types"][i][0][0] == g_type:
                        add_entry(stats, prefix+"glas_corr", 1)
                if p["mst_heads"][i] == g_head:
                    add_entry(stats, prefix+"muas_corr", 1)
                    if p["mst_types"][i] == g_type:
                        add_entry(stats, prefix+"mlas_corr", 1)
                if p["heads"][i][0] == p["mst_heads"][i]:
                    add_entry(stats, prefix+"hitU", 1)
                    if p["types"][i][0][0] == p["mst_types"][i]:
                        add_entry(stats, prefix+"hitL", 1)
    for n in sorted(stats.keys()):
        print("-- %s: %s"%(n, stats[n]))
    print("=====")
    print("POS-ACC=%.3f, Greedy-UAS/LAS=%.3f/%.3f, MST-UAS/LAS=%.3f/%.3f, AGREE-U/L=%.3f/%.3f." %
          (stats["tok_tag_corr"]/stats["tok_"], stats["tok_np_guas_corr"]/stats["tok_np_"], stats["tok_np_glas_corr"]/stats["tok_np_"],
           stats["tok_np_muas_corr"]/stats["tok_np_"], stats["tok_np_mlas_corr"]/stats["tok_np_"],
           stats["tok_np_hitU"]/stats["tok_np_"], stats["tok_np_hitL"]/stats["tok_np_"],))

if __name__ == '__main__':
    main(sys.argv[1:])

# python3 -m pdb ../src/examples3/pred/test_pipe.py ../data/ud23/en_test.conllu ../zt_en/models/ ../zp_en/models/ 1
# PYTHONPATH=../src/examples3/pred/../.. CUDA_VISIBLE_DEVICES=1 python3 ../src/examples3/pred/pipe_tp.py --input tmp.in --output tmp.out --tagger_path ../zt_en/models/ --tagger_name network.pt --parser_path ../zp_en/models/ --parser_name network.pt --len_thresh_min 0 --len_thresh_max 1000 --oov_thresh 1.0 --gpu --parser_mst 0 --parser_topk 1 --tagger_topk 1
# python3 ../../../models/src/examples3/pred/test_pipe.py ../../ud23/de_test.conllu ../../../models/{zft_de,zp_de_nopos_freeze500}/models/  $CUDA_VISIBLE_DEVICES
