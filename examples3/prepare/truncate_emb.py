#
import sys

def main(fin, fout, max_num):
    line1 = fin.readline()
    num_vec, num_dim = [int(x) for x in line1.split()]
    print("Read embedding of %s,%s." % (num_vec, num_dim), file=sys.stderr)
    c = 0
    lines = []
    for line in fin:
        lines.append(line)
        c += 1
        if c >= max_num:
            break
    # write
    print("Write embedding of %s,%s" % (c, num_dim), file=sys.stderr)
    fout.write("%d %d\n" % (c, num_dim))
    for line in lines:
        fout.write(line)

if __name__ == '__main__':
    MAX_NUM = 200000
    main(sys.stdin, sys.stdout, MAX_NUM)

#
"""
for lang in ar bg ca zh hr cs da nl en et "fi" fr de he hi id it ja ko la lv no pl pt ro ru sk sl es sv uk;
do
echo $lang;
python3 ../truncate_embed.py < embeds_orig/wiki.${lang}.vec > zwiki.${lang}.vec
done
"""
