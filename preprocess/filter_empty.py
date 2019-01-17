import sys


def filter_empty(string):
    empty_delete = len(list(filter(lambda x: x != '', string.split("|||")))) < 2
    underscore_delete = string.split("|||")[0].strip(' ') == '_'



    return delete


if __name__ == '__main__':
    for line in sys.stdin:
        line = line.strip()
        if not filter_empty(line):
            sys.stdout.write(line)
            sys.stdout.write('\n')
