import sys
af = open(sys.argv[1])
rf = open(sys.argv[1] + ".rep", 'w')
line = af.readline().replace('\n', '')
while line is not None and line != '':
    if '$￥@0010__$' in line:
        line = line.replace('$￥@0010__$', '$￥@010__$')
    if '$￥@' in line and '__$' in line:
        line = line.replace('$￥@', 'DIPML').replace('__$', 'MASK')
    rf.write(line + '\n')
    line = af.readline().replace('\n', '')
af.close()
rf.close()
