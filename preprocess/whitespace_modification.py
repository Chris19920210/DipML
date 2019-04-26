import sys
import re


def correct_whitespace(line):
    global pattern_1
    line = line.split("\t")
    m = pattern_1.search(line[1])
    if not m:
        return "\t".join(line)
    else:
        for each in pattern_1.finditer(line[1]):
            if len(each.group().strip()) > 2:
                z = re.search("\s*".join(list(each.group().replace(" ", ""))), line[0], re.IGNORECASE)
                if z:
                    line[1] = line[1].replace(each.group().strip(" "), z.group())
        return "\t".join(line)


if __name__ == '__main__':
    pattern_1 = re.compile("([a-zA-Z]\s*){1,}")

    for line in sys.stdin:
        try:
            line_cws = correct_whitespace(line.strip())
            sys.stdout.write(line_cws)
            sys.stdout.write('\n')
        except:
            continue

