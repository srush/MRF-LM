#!/usr/bin/env python

"""
Script for constructing the moments for taggign.
"""

import os
import sys
import argparse
from backport_collections import Counter


L = None
M = None

def main(arguments):
    global L, M
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-L', '--ngram-context', help="NGram context size.",
                        type=int, default=1)
    parser.add_argument('-M', '--word-context', help="Word window size.",
                        type=int, default=1)
    parser.add_argument('train', help="CoNLL-formatted train file.", type=str)
    parser.add_argument('valid', help="CoNLL-formatted valid file.", type=str)
    parser.add_argument('test', help="CoNLL-formatted test file.", type=str)
    parser.add_argument('name', help="Data output name.", type=str)

    args = parser.parse_args(arguments)

    L = args.ngram_context
    M = args.word_context

    data = read_conll_as_one_sent(open(args.train))
    data_valid = read_conll_as_one_sent(open(args.valid))
    data_test = read_conll_as_one_sent(open(args.test))

    data_valid_sent = list(read_conll(open(args.valid)))
    data_test_sent = list(read_conll(open(args.test)))

    lexicon = make_vocab(data, data_valid, data_test)

    counts = collect_counts(data, lexicon)
    val_counts = collect_counts(data_valid, lexicon)

    # Dict name.
    write_names(open(args.train + "." + args.name + ".tagnames", 'w'),
                open(args.train + "." + args.name + ".names", 'w'),
                lexicon)

    # Lexical features.
    write_features(open(args.train + "." + args.name + ".features", 'w'),
                   lexicon)

    # Counts for training.
    write_counts(open(args.train + "." + args.name + ".counts", 'w'),
                 counts, lexicon)
    write_counts(open(args.valid + "." + args.name + ".counts", 'w'),
                 val_counts, lexicon)

    # For tagging.
    write_words(data_valid_sent, open(args.valid + "." + args.name + ".words", 'w'),
                lexicon)
    write_words(data_test_sent,  open(args.test  + "." + args.name + ".words", 'w'),
                lexicon)


def read_conll_as_one_sent(handle):
    sent = []
    for l in handle:
        t = l.strip().split()
        if not l.strip():
            word, tag = "<S>", "<T>"
        else:
            word, tag = t[1], t[4]
        sent.append((word, tag))
    return sent

def read_conll(handle):
    sent = []
    for l in handle:
        t = l.strip().split()
        if not l.strip():
            yield sent
            sent = []
        else:
            word, tag = t[1], t[4]
            sent.append((word, tag))
    yield sent


def make_vocab(data, val_data, test_data):
    train_counts = Counter()
    word_set = set()
    tag_set = set()

    train_counts.update((word for word, _ in data))

    all_words = data + val_data + test_data
    word_set.update((word for word, _ in all_words))
    tag_set.update((tag for _, tag in all_words))

    word_dict = {"<S>": 0}
    tag_dict  = {"<T>": 0}

    for word in word_set:
        word_dict.setdefault(word, len(word_dict))

    for tag in tag_set:
        tag_dict.setdefault(tag, len(tag_dict))

    return {"tags": tag_dict,
            "words": word_dict,
            "word_counts": train_counts}

def write_words(data, out, lex):
    print >>out, len(data)
    for sent in data:
        for word, tag in sent:
            print >>out, lex["words"][word], lex["tags"][tag]
        print >>out, -1, -1

def collect_counts(data, lex):
    tag_pair_counts = [Counter() for l in range(L)]
    tag_word_pair_counts = [Counter() for m in range(M)]

    sent = data
    def get_word(pos):
        if pos < 0 or pos >= len(sent):
            word = "<S>"
        else:
            word = sent[pos][0]
        return lex["words"][word]

    def get_tag(pos):
        if pos < 0 or pos >= len(sent):
            tag = "<T>"
        else:
            tag = sent[pos][1]
        return lex["tags"][tag]

    for i in range(len(sent)):
        for l in range(L):
            tag_pair_counts[l][get_tag(i-l-1), get_tag(i)] += 1

        start = i - ((M - 1) / 2);
        for m in range(M):
            tag_word_pair_counts[m][get_tag(i), get_word(start + m)] += 1


    return {"tag": tag_pair_counts,
            "word": tag_word_pair_counts,
            "total": len(sent)}

def write_counts(out_handle, counts, lexicon):
    print >>out_handle, counts["total"], L + M + 1

    for _ in range(L + 1):
        print >>out_handle, len(lexicon["tags"])
    for _ in range(M):
        print >>out_handle, len(lexicon["words"])

    for l in range(L):
        print >>out_handle, len(counts["tag"][l])
        for (t1, t2), count in counts["tag"][l].iteritems():
            print >>out_handle, t1, t2, count

    for k in range(M):
        print >>out_handle, len(counts["word"][k])
        for (t1, w), count in counts["word"][k].iteritems():
            print >>out_handle, t1, w, count

def write_names(tag_out_handle, out_handle, lexicon):
    for tag, _ in sorted(lexicon["tags"].items(), key=lambda a: a[1]):
        print >>tag_out_handle, lexicon["tags"][tag], tag, 0

    for word, _ in sorted(lexicon["words"].items(), key=lambda a: a[1]):
        print >>out_handle, lexicon["words"][word], word, 0


def write_features(out, lexicon):
    """
    Feature from CRFTagger
    1) M=5 surrounding words
    2) prefixes of length up to 4 (only M=cur)
    3) Suffixes of length up to 4 (only M=cur)
    4) Word pairs (last+cur/cur+next) (skip?)
    5) All caps
    6) All caps and ends with s.
    7) First Capital
    10) Ends with s.
    (8,9 11,12) pretoken
    13) Has number
    14) All numbers/./,
    15) has hyphen
    """

    features = []
    feature_dict = {}
    for m in range(M):
        for word, v in lexicon["words"].iteritems():
            active_features = [(0, word), (1,)]

            if lexicon["word_counts"][word] <= 5:
                if any([a.isdigit() for a in word]):
                    active_features = [(-1, "DIGIT")]
                else:
                    active_features = [(-1, "UNK")]

            # Features for rare words.
            if m == (M-1)/2 and lexicon["word_counts"][word] <= 5:
                active_features += [
                                 (2, 1, word[:1]),
                                 (2, 2, word[:2]),
                                 (2, 3, word[:3]),
                                 (2, 4, word[:4]),
                                 (3, 1, word[-1:]),
                                 (3, 2, word[-2:]),
                                 (3, 3, word[-3:]),
                                 (3, 4, word[-4:]),
                                 (5, word.isupper()),
                                 (6, word.isupper() and word[-1] == 'S'),
                                 (7, word[0].isupper() and word[-1] == 's'),
                                 (8, word[-1] == 's'),
                                 (13, any([a.isdigit() for a in word])),
                                 (14, all([a.isdigit() or a in ".," for a in word])),
                                 (15, any([a == '-' for a in word]))]

            feats = []
            for active_feature in active_features:
                if active_feature[-1] is not False:
                    if active_feature not in feature_dict:
                        feature_dict[active_feature] = len(feature_dict)
                    feats.append(feature_dict[active_feature])
            if feats:
                features.append(((m, v), feats))

    print >>out, len(feature_dict), len(features), M
    for (m, v), feats in features:
        print >>out, m, v, len(feats), " ".join(map(str, feats))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
