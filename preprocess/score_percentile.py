import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser(description='perplexity')
parser.add_argument('--percentile', type=float, default=95,
                    help='percentile')
args = parser.parse_args()

if __name__ == '__main__':
    nums = []
    for line in sys.stdin:
        nums.append(float(line.strip()))
    nums = np.array(nums)
    printstr="""
    q:{q:.2f}
    p:{p:.4f}
    """
    print(printstr.format(q=args.percentile, p=np.percentile(nums, args.percentile)))
