from collections import defaultdict
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import operator


class Mbr(object):
    def __init__(self, sentence_probs):
        self.sentence_probs = sentence_probs

    def mbr_calc(self):
        if len(self.sentence_probs) == 0:
            return None
        if len(self.sentence_probs) == 1:
            return self.sentence_probs.keys()[0]
        result = defaultdict(float)
        smoothie = SmoothingFunction().method4

        for sentence, prob in self.sentence_probs.items():
            for sentence_candidate, prob_candidate in self.sentence_probs.items():
                if sentence != sentence_candidate:
                    bleu = sentence_bleu([sentence.strip().split(" ")], sentence_candidate.strip().split(" "),
                                         smoothing_function=smoothie)
                    result[sentence] += (1 - bleu) * prob_candidate
        return min(result.items(), key=operator.itemgetter(1))[0]
