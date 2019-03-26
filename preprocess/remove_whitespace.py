import sys
import re

pattern_1 = re.compile("(?<=\\d) +(?=(\\d ?(年|月|日|:|点|：|号)))")
pattern_2 = re.compile("(?<=\\d)(?=(年|月|日|:|点|：|号))")

if __name__ == '__main__':
    for line in sys.stdin:
        line = line.strip()
        if pattern_1.search(line):
            line = re.sub(pattern_1, "", line)
            line = re.sub(pattern_2, " ", line)
        sys.stdout.write(str(line))
        sys.stdout.write('\n')
