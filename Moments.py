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
    lintext = [y for sent in txt
               for y in (['<S>']*K + sent.split())]


    return txt, lintext+ ['<S>']*K, len(lintext)

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
    # train_text_file = args.train + '_text_K' + str(args.K) + '.dat'
    # valid_text_file = args.valid + '_text_K' + str(args.K) + '.dat'


    K = args.K
    txt, lintext, N = get_text(args.train, args.K)
    _, valid_lintext, valN = get_text(args.valid, args.K)

    vocounts = Counter()
    vocounts['<unk>'] = 0

    for sent in txt:
        lsent = ['<S>']*K + sent.split()
        vocounts.update(lsent)


    vocab = dict([(w, i)
                  for i, (w, _) in enumerate(vocounts.most_common())])

    with open(vocab_file,'w') as f:
        for i, (w, c) in enumerate(vocounts.most_common()):
            print >>f, i, w, c

    make_moments(lintext, vocounts, vocab, train_moments_file, args.K, N)
    make_moments(valid_lintext, vocounts, vocab, valid_moments_file, args.K, valN)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
