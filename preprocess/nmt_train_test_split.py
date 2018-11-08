from sklearn.model_selection import train_test_split
import os
import argparse
import pandas as pd
parser = argparse.ArgumentParser(description='Tokenize English/Chinese Sentence')
parser.add_argument('--base-dir', type=str, default=None,
                    help='path to data base dir')
parser.add_argument('--input-en', type=str, required=True,
                    help='input path for english file')
parser.add_argument('--input-zh', type=str, required=True,
                    help='input path for chinese file')
parser.add_argument('--ratio', type=float, default=0.001,
                    help='ratio for train test split')
args = parser.parse_args()

if __name__ == '__main__':
    data_zh = pd.read_csv(os.path.join(args.base_dir, args.input_zh), delimiter="##########",
                          encoding='utf-8', header=None)
    data_en = pd.read_csv(os.path.join(args.base_dir, args.input_en), delimiter="##########",
                          encoding='utf-8', header=None)
    data_zh_train, data_zh_test, data_en_train, data_en_test = train_test_split(data_zh, data_en, test_size=args.ratio)
    data_zh_train.to_csv((os.path.join(args.base_dir, args.input_zh + '.train')), header=None, index=None)
    data_zh_test.to_csv((os.path.join(args.base_dir, args.input_zh + '.test')), header=None, index=None)
    data_en_train.to_csv((os.path.join(args.base_dir, args.input_en + '.train')), header=None, index=None)
    data_en_test.to_csv((os.path.join(args.base_dir, args.input_en + '.test')), header=None, index=None)
