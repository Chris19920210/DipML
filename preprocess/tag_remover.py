import sys
import html

if __name__ == '__main__':
    for line in sys.stdin:
        line = html.unescape(line.strip())
        sys.stdout.write(line)
        sys.stdout.write('\n')
