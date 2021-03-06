import kenlm
import argparse
import sys

parser = argparse.ArgumentParser(description='perplexity')
parser.add_argument('--model-path', type=str, required=True,
                    help='input path for ensemble file')
args = parser.parse_args()


def perplexity(sentence):
    global model
    return model.perplexity(sentence)


if __name__ == '__main__':
    model = kenlm.Model(args.model_path)
    for line in sys.stdin:
        score = perplexity(line.strip())
        sys.stdout.write(str(line.strip()))
        sys.stdout.write("\t")
        sys.stdout.write(str(score))
        sys.stdout.write('\n')
