#!/bin/sh


# 1. compile giza with:
#	git clone https://github.com/moses-smt/giza-pp.git
#	cd giza-pp
#	make
# then move GIZA++-v2/plain2snt.out GIZA++-v2/snt2cooc.out GIZA++-v2/GIZA++ mkcls-v2/mkcls into the directory of segmented texts

# 2. generate tokens
#    input: zh.total/en.total (source and target texts after segmentation)
#    output: en.vcb/zh.vcb/en_zh.snt/zh_en.snt
./plain2snt.out zh.total en.total

# 3. generate co-occurrence files (can run synchronously)
#    input: vcb and snt fles respectively
#    output: zh_en.cooc/en_zh.cooc
./snt2cooc.out zh.total.vcb en.total.vcb zh.total_en.total.snt > zh_en.cooc
./snt2cooc.out en.total.vcb zh.total.vcb en.total_zh.total.snt > en_zh.cooc

# 4. generate word classes (can run synchronously) !deprecated
#    params: -c  number of classes
#            -n  number of iterations for optimization
#            -p  input file of segmented texts
#            -V  output file of word classes
#            opt for optimization
#./mkcls -pzh.total -Vzh.vcb.classes opt
#./mkcls -pen.total -Ven.vcb.classes opt

# 5. compute perplexity
#    output: z2e/z2e.perp e2z/e2z.perp
mkdir z2e
mkdir e2z
./GIZA++ -S zh.total.vcb -T en.total.vcb -C zh.total_en.total.snt -CoocurrenceFile zh_en.cooc -o z2e -OutputPath z2e
./GIZA++ -S en.total.vcb -T zh.total.vcb -C en.total_zh.total.snt -CoocurrenceFile en_zh.cooc -o e2z -OutputPath e2z

# 6. calc translation ratio with translation probability and word alignment
#    put calc_perp.py into the directory of segmented texts
#    output:  e2z_res.txt / z2e_res.txt
python calc_perp.py e2z
python calc_perp.py z2e
