# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# Copyright (c) 2018-present, Yunsu Kim
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from math import exp, log
import os
import sys

import numpy as np
import torch

from ..utils import get_nn_avg_dist


# constants
UNK_TOKEN = '<unk>'
EOS_TOKEN = '</s>'
CATEGORY_LABELS = ['$number', '$url']
PROB_DEFAULT = 0.01


# scaling functions for similarity scores
def linear_scaling(x):
    return ((x + 1) / 2).tolist()

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def softmax(x, T=1):
    x = x / T  # temparature
    e_x = np.exp(x - np.max(x))  # max() for numerical stability
    return e_x / e_x.sum()


class Translator(object):

    def __init__(self, src_emb, tgt_emb, src_dico, tgt_dico, params):
        self.emb1 = src_emb.weight.data
        self.emb2 = tgt_emb.weight.data
        self.src_dico = src_dico
        self.tgt_dico = tgt_dico
        self.params = params

        # normalize embeddings
        self.emb1 = self.emb1 / self.emb1.norm(2, 1, keepdim=True).expand_as(self.emb1)
        self.emb2 = self.emb2 / self.emb2.norm(2, 1, keepdim=True).expand_as(self.emb2)
        
        # precompute hub penalty terms for csls
        if self.params.similarity_measure == 'csls':
            self.average_dist1 = get_nn_avg_dist(self.emb2, self.emb1, self.params.csls_k)
            self.average_dist2 = get_nn_avg_dist(self.emb1, self.emb2, self.params.csls_k)
            self.average_dist1 = torch.from_numpy(self.average_dist1).type_as(self.emb1)
            self.average_dist2 = torch.from_numpy(self.average_dist2).type_as(self.emb2)

    # src_word -> [topk_tgt_words], [topk_scores]
    def word_translation(self, src_word):
        if src_word not in self.src_dico.word2id:  # source unknown word
            if self.params.unknown_words == 'copy':
                return [src_word], [PROB_DEFAULT]
            elif self.params.unknown_words == 'unk':
                return ['<unk>'], [PROB_DEFAULT]
            else:
                raise ValueError('Wrong argument for --unknown_words')
        elif src_word in CATEGORY_LABELS:
            return [src_word], [PROB_DEFAULT]
        else:
            src_word_id = self.src_dico.word2id[src_word]
            src_word_vec = self.emb1[src_word_id].view(1, self.params.emb_dim)

            # nearest neighbor
            scores = src_word_vec.mm(self.emb2.transpose(0,1))
            if self.params.similarity_measure == 'csls':
                scores.mul_(2)
                scores.sub_(self.average_dist1[[src_word_id]][:, None])
                scores.sub_(self.average_dist2[None, :])

            topk = scores.topk(self.params.topk, 1, True)
            topk_scores = topk[0].cpu().numpy()[0]
            topk_tgt_ids = topk[1][0]

            # scaling of similarity scores
            if self.params.emb_scaling == 'linear':
                topk_scores = linear_scaling(topk_scores)
            elif self.params.emb_scaling == 'sigmoid':
                topk_scores = sigmoid(topk_scores)
            elif self.params.emb_scaling == 'softmax':
                topk_scores = softmax(topk_scores, self.params.softmax_temparature)

            # convert from ids to words
            topk_tgt_words = []
            topk_tgt_scores = []

            for i in range(len(topk_tgt_ids)):
                tgt_word = self.tgt_dico.id2word[topk_tgt_ids[i]]
                if tgt_word not in CATEGORY_LABELS and tgt_word != EOS_TOKEN:
                    topk_tgt_words.append(tgt_word)
                    topk_tgt_scores.append(topk_scores[i])

            if len(topk_tgt_words) == 0:
                topk_tgt_words = [src_word]
                topk_tgt_scores = [PROB_DEFAULT]

            return topk_tgt_words, topk_tgt_scores

    def sent_translation(self, sent, lm):
        beam = [[list(), 0.0]]
        for n in range(len(sent)):
            src_word = sent[n]
            if type(src_word) == bytes:
                src_word = src_word.decode()
            topk_tgt_words, topk_scores = self.word_translation(src_word)
            
            if lm:
                all_candidates = list()

                topk_zipped = list(zip(topk_tgt_words, topk_scores))
                for sequence, sequence_score in beam:
                    lm_score_history = lm.score(' '.join(sequence), bos=True, eos=False)
                    for tgt_word, score in topk_zipped:
                        eos = False
                        if n == len(sent) - 1:
                            eos = True
                        new_sequence = sequence + [tgt_word]
                        lex_score = self.params.lex_scaling * log(score)
                        lm_score = self.params.lm_scaling * (lm.score(' '.join(new_sequence), bos=True, eos=eos) - lm_score_history)
                        new_sequence_score = sequence_score + lex_score + lm_score
                        all_candidates.append([new_sequence, new_sequence_score])
                ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
                beam = ordered[:self.params.beam_size]
            else:
                beam[0][0].append(topk_tgt_words[0])
                beam[0][1] += log(topk_scores[0])
        return beam[0][0]

    def corpus_translation(self, corpus, lm):
        translations = []
        for sent in corpus:
            trs = self.sent_translation(sent, lm)
            translations.append(trs)

        if self.params.output:
            output_path = os.path.join(self.params.output)
            output_file = open(output_path, 'w')
        else:
            output_path = 'stdout'
            output_file = sys.stdout
        print('Writing translations to %s ...' % output_path, file=sys.stderr)
        for trs in translations:
            output_file.write("%s\n" % ' '.join(trs))
        if output_file is not sys.stdout:
            output_file.close()
