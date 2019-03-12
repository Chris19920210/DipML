import argparse
import os


parser = argparse.ArgumentParser(description='Bpe')
parser.add_argument('--data-dir', type=str, default=None,
                    help='method for bpe')
args = parser.parse_args()

if __name__ == '__main__':
    datasets = os.listdir(args.data_dir)
    lang1 = open(os.path.join(args.data_dir, datasets[0]), "r", errors='ignore')
    lang2 = open(os.path.join(args.data_dir, datasets[1]), "r", errors='ignore')
    with open(os.path.join("./", datasets[0]), 'w') as lang1_writer, \
            open(os.path.join("./", datasets[1]), 'w') as lang2_writer:
        for line_1, line_2 in zip(lang1.readlines(), lang2.readlines()):
            lang1_writer.write(line_1.strip() + "\n")
            lang2_writer.write(line_2.strip() + "\n")
