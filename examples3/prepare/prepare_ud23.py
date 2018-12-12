#

# prepare data and embeddings

# for cur_lang in cs de en fr "fi" it ja zh; do

#print("\n".join([z[1][0].split("-")[0].split("_")[1] for z in x]))
LANGUAGE_LIST = (
    # init group, I think at most these for this project
    ["cs", ["UD_Czech-PDT", "UD_Czech-CAC", "UD_Czech-CLTT", "UD_Czech-FicTree"], "IE.Slavic.West"],
    ["de", ["UD_German-GSD"], "IE.Germanic.West"],
    ["en", ["UD_English-EWT", "UD_English-ParTUT", "UD_English-GUM"], "IE.Germanic.West"],
    ["fr", ["UD_French-GSD", "UD_French-ParTUT", "UD_French-Sequoia"], "IE.Romance.West"],
    ["fi", ["UD_Finnish-TDT"], "Uralic.Finnic"],
    ["it", ["UD_Italian-ISDT", "UD_Italian-ParTUT"], "IE.Romance.Italo"],
    ["ja", ["UD_Japanese-GSD"], "Japanese"],
    ["zh", ["UD_Chinese-GSD"], "Sino-Tibetan"],
)

NAME_MAP = {
    "cs": "Czech", "de": "German", "en": "English", "fr": "French",
    "fi": "Finnish", "it": "Italian", "ja": "Japanese", "zh": "Chinese",
}

# confs (make sure to set up the dir correctly)
UD2_DIR = "./ud-treebanks-v2.3/"
OUT_DIR = "./ud23/"
LIB_DIR = "./ud23/fastText_multilingual/"

# ===== help
import os, subprocess, sys, gzip

printing = lambda x: print(x, file=sys.stderr, flush=True)

def system(cmd, pp=False, ass=False, popen=False):
    if pp:
        printing("Executing cmd: %s" % cmd)
    if popen:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        n = p.wait()
        output = str(p.stdout.read().decode())
    else:
        n = os.system(cmd)
        output = None
    if pp:
        printing("Output is: %s" % output)
    if ass:
        assert n==0
    return output

def zopen(filename, mode='r', encoding="utf-8"):
    if filename.endswith('.gz'):
        # "t" for text mode of gzip
        return gzip.open(filename, mode+"t", encoding=encoding)
    else:
        return open(filename, mode, encoding=encoding)
# =====

# keep only basic-dep fields
def deal_conll_file(fin, fout):
    for line in fin:
        line = line.strip()
        fields = line.split("\t")
        if len(line) == 0:
            fout.write("\n")
        else:
            try:
                z = int(fields[0])
                fout.write("\t".join(fields)+"\n")
            except:
                pass

def main():
    # first get the English one
    # lang = "en"
    # system("wget -nc -O %s/wiki.%s.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.%s.vec" % (OUT_DIR, lang, lang), pp=True)
    # en_dict = FastVector(vector_file='%s/wiki.en.vec' % OUT_DIR)
    for zzz in LANGUAGE_LIST:
        lang, fnames = zzz[0], zzz[1]
        printing("Dealing with lang %s." % lang)
        for curf in ["train", "dev", "test"]:
            out_fname = "%s/%s_%s.conllu" % (OUT_DIR, lang, curf)
            fout = zopen(out_fname, "w")
            for fname in fnames:
                last_name = fname.split("-")[-1].lower()
                path_name = "%s/%s/%s_%s-ud-%s.conllu" % (UD2_DIR, fname, lang, last_name, curf)
                if os.path.exists(path_name):
                    with zopen(path_name) as fin:
                        deal_conll_file(fin, fout)
            # special adding PUD's test into train
            if curf == "train":
                path_name = "%s/UD_%s-PUD/%s_pud-ud-test.conllu" % (UD2_DIR, NAME_MAP[lang], lang)
                with zopen(path_name) as fin:
                    deal_conll_file(fin, fout)
            #
            fout.close()
            # stat
            system('cat %s | grep -E "^$" | wc' % out_fname, pp=True)
            system('cat %s | grep -Ev "^$" | wc' % out_fname, pp=True)
            system("cat %s | grep -Ev '^$' | cut -f 5 -d $'\t'| grep -Ev 'PUNCT|SYM' | wc" % out_fname, pp=True)

# =====
# sys.path.append(LIB_DIR)        # project embeddings
#
# from fasttext import FastVector
#
# def main2():
#     for zzz in LANGUAGE_LIST:
#         lang = zzz[0]
#         # get original embed
#         system("wget -nc -O %s/wiki.%s.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.%s.vec" % (OUT_DIR, lang, lang), pp=True)
#         # project with LIB-matrix
#         lang_dict = FastVector(vector_file='%s/wiki.%s.vec' % (OUT_DIR, lang))
#         lang_dict.apply_transform("%s/alignment_matrices/%s.txt" % (LIB_DIR, lang))
#         lang_dict.export("%s/wiki.multi.%s.vec" % (OUT_DIR, lang))
# =====

if __name__ == '__main__':
    main()
    # main2()

# The prepared files are: OUT_DIR/{*_*.conllu, wiki.multi.*.vec}
# python3 prepare_ud23.py |& grep -v "s$" | tee ud23/log_cu
# for whs in train dev test; do mv zh_${whs}.conllu zht_${whs}.conllu; python3 t2s/zconv_t2s.py <zht_${whs}.conllu >zhs_${whs}.conllu; done
