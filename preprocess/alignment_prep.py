import sys


def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


if __name__ == '__main__':
    for line in sys.stdin:
        line_cws = strQ2B(line.strip())
        sys.stdout.write(line_cws)
        sys.stdout.write('\n')
