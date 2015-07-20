#!/usr/bin/env python

"""
Script for constructing the moments for language modelling.
"""

import os
import sys
import argparse
from backport_collections import Counter

def get_text(in_file, K):
    with open(in_file) as f:
        txt = [x.strip() for x in f if len(x.split()) >= K]
    with open(in_file) as f:
        alltxt = [x.strip() for x in f]
    lintext = [y for sent in txt
               for y in (['<S>']*K + sent.split())]


    return alltxt, lintext + ['<S>']*K, len(lintext)

def make_moments(lintext, vocounts, vocab, out_file, K, N):

    paircounts = [Counter() for k in range(K)]
    for (i, wrd) in enumerate(lintext[:len(lintext) - K]):
        for k in range(1, K+1):
            paircounts[k-1][(wrd, lintext[i+k])] += 1

    #print moments
    with open(out_file,'w') as f:
        print >>f, N, K + 1

        for _ in range(K+1):
            print >> f, len(vocab)


        for k in range(K):
            print >>f, len(paircounts[k].keys())
            for (a, b), count in paircounts[k].most_common():
                print >>f, vocab[a], vocab[b], count


def write_words(txt, out_file, vocab, K):
    with open(out_file,'w') as f:
        print >>f, len(txt)
        for sent in txt:
            sent_txt = ["<S>"] * K + sent.split() + ["<S>"] * (K+1)
            print >>f, len(sent_txt),
            for wrd in sent_txt:
                print >>f, vocab.get(wrd, vocab['<unk>']),
            print >>f, ''

def write_pseudo_ngrams(in_file, out_file, K):
    "Construct the counts for the p-ngram model."
    ngrams = Counter()
    with open(out_file,'w') as f:
        for l in open(in_file):
            words = l.strip().split()
            words = ["<s>"] * K + words + ["</s>"] * (K+1)
            for i in range(K, len(words)-K):
                total = []
                for k in range(1, K+1):
                    total.append(words[i-k])
                    total.append(words[i+k])
                total.reverse()
                total = total + [words[i]]
                for j in range(len(total)):
                    ngrams[tuple(total[j:])] += 1
        for l in ngrams:
            print >>f, " ".join(l), ngrams[l]


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-K', '--K', help="Window size.", type=int)
    parser.add_argument('-t', '--train',  help="Training file.", type=str)
    parser.add_argument('-v', '--valid', help="Validation file", type=str)

    args = parser.parse_args(arguments)

    vocab_file = args.train + '_vocab_K' + str(args.K) + '.dat'
    train_moments_file = args.train + '_moments_K' + str(args.K) + '.dat'
    valid_moments_file = args.valid + '_moments_K' + str(args.K) + '.dat'
    train_text_file = args.train + '_text_K' + str(args.K) + '.dat'
    valid_text_file = args.valid + '_text_K' + str(args.K) + '.dat'
    pngram_file = args.train + '_pngram_K' + str(args.K) + '.dat'


    K = args.K
    txt, lintext, N = get_text(args.train, args.K)
    valid_txt, valid_lintext, valN = get_text(args.valid, args.K)

    vocounts = Counter()
    vocounts['<unk>'] = 0

    vocounts.update(lintext)


    vocab = dict([(w, i)
                  for i, (w, _) in enumerate(vocounts.most_common())])

    with open(vocab_file,'w') as f:
        for i, (w, c) in enumerate(vocounts.most_common()):
            print >>f, i, w, c

    make_moments(lintext, vocounts, vocab, train_moments_file, args.K, N)
    make_moments(valid_lintext, vocounts, vocab, valid_moments_file, args.K, valN)
    write_words(txt, train_text_file, vocab, args.K)
    write_words(valid_txt, valid_text_file, vocab, args.K)
    write_pseudo_ngrams(args.train, pngram_file, args.K)
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
