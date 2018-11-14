import asyncio
from concurrent.futures import ProcessPoolExecutor
import aioprocessing
from collections import defaultdict
import numpy as np
from mbr import Mbr
import argparse
import os
from multi_asyn_utils import start

parser = argparse.ArgumentParser(description='Ensemble')
parser.add_argument('--base-dir', type=str, default=None,
                    help='path to data base dir')
parser.add_argument('--beam-size', type=int, default=10,
                    help='size for beam search')
parser.add_argument('--num-processors', type=int, default=10,
                    help='num of processor')
parser.add_argument('--input', type=str, nargs='+',
                    help='input path for ensemble file')
parser.add_argument('--output', type=str, required=True,
                    help='output path for ensemble result')
args = parser.parse_args()


def sentence_parsing(sentences):
    sentence_probs = defaultdict(float)
    sentence_id = sentences[0]
    for candidate in sentences[1:]:
        sentence_probs[candidate.split("\t")[0]] = max(np.exp(float(candidate.split("\t")[1])),
                                                       sentence_probs[candidate.split("\t")[0]])
    denominator = sum(sentence_probs.values())
    for k in sentence_probs:
        sentence_probs[k] /= denominator
    return sentence_id, sentence_probs


def best_sentence(sentences):
    sentence_id, sentence_probs = sentence_parsing(sentences)
    mbr = Mbr(sentence_probs)
    return sentence_id, mbr.mbr_calc()


if __name__ == '__main__':
    readers = [open(os.path.join(args.base_dir, input_path), 'r') for input_path in args.input]
    m = aioprocessing.AioManager()
    in_queue_m = m.AioQueue()

    counter = 0
    ret = []
    for lines in zip(*[reader.readlines() for reader in readers]):
        counter += 1
        if counter % args.beam_size == 0:
            ret.insert(0, '%d' % (int(counter / args.beam_size)))
            in_queue_m.put(ret)
            counter = 0
            ret = []
        else:
            for line in lines:
                ret.append(line.strip())
    for reader in readers:
        reader.close()
    out_queue_m = m.AioQueue()
    executor = ProcessPoolExecutor(args.num_processors)
    asyncio.get_event_loop().run_until_complete(start(executor, in_queue_m, out_queue_m, best_sentence))
    executor.shutdown()
    result = []
    while not out_queue_m.empty():
        result.append(out_queue_m.get())
    result.sort(key=lambda x: int(x[0]))

    with open(os.path.join(args.base_dir, args.output), 'w') as f:
        for line in result:
            f.write(line[1])
            f.write('\n')
