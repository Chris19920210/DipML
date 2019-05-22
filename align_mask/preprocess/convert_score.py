import sys

fs = open(sys.argv[1])
fr = open(sys.argv[2], 'w')
cnt = 0
line = fs.readline().replace('\n', '')
while line is not None and line !='':
    cols = line.split(' ||| ')
    #print(cols[0])
    fr.write(cols[0] + '\n')
    cnt += 1
    if cnt % 100000 == 0:
        print(cnt)
    line = fs.readline().replace('\n', '')
fs.close()
fr.close()
