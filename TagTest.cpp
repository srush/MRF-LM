#include <assert.h>
#include <fstream>

#include "TagTest.h"
#include "Model.h"
#include "Tag.h"


#define INF 1e8
#define START_TAG 0
#define END_TAG 0

#define START_WORD 0
#define END_WORD 0
#define UNK_WORD 1

TagTest::TagTest(string mf) {
    ifstream myfile(mf);
    int n_sents;
    myfile >> n_sents;

    taggings.resize(n_sents);
    words.resize(n_sents);

    for (int sent = 0; sent < n_sents; ++sent) {
        while (true) {
            int word, tag;
            myfile >> word >> tag;
            if (word == -1) break;
            words[sent].push_back(word);
            taggings[sent].push_back(tag);
            assert(tag < 50);
        }
    }
}


double TagTest::TestModel(Model &model) {
    Tag *tmodel = (Tag *)&model;
    vector<int> all_total(words.size(), 0);
    vector<int> all_correct(words.size(), 0);

    #pragma omp parallel for
    for (int i = 0; i < words.size(); ++i) {
        int total = 0;
        int correct = 0;

        vector<int> tags;
        if (words[i].size() == 0) continue;
        Viterbi(*tmodel, words[i], &tags, tmodel->T);
        for (int j = 0; j < taggings[i].size(); ++j) {
            if (taggings[i][j] == tags[j]) {
                correct++;
            }
            total++;
        }
        all_total[i] = total;
        all_correct[i] = correct;

    }
    int total = 0;
    int correct = 0;
    for (int i = 0; i < words.size(); ++i) {
        total += all_total[i];
        correct += all_correct[i];
    }
    cout << "Correct " << correct << " Total " << total << endl;
    return correct / (double)total;
}

double TagTest::potential(Tag &model, const vector<int> &words,
                          int i, int tag, int prev) {
    // Bigram potential.
    double potential = model.theta[0][prev][tag];
    for (int k = 0; k < model.MR; ++k) {
        potential += model.theta[k + 1][tag][words[i+k]];
    }
    return potential;
}

double TagTest::Viterbi(Tag &model, const vector<int> &words,
                        vector<int> *tagging,
                        int T) {
    // Compute the highest scoring sequence.
    int n = words.size();
    vector<vector<int> > bp(n);
    vector<vector<double> > pi(n);

    for (int i = 0; i < n; ++i) {
        bp[i].resize(T, -1);
        pi[i].resize(T, -INF);
    }

    for (int tag = 0; tag < T; ++tag) {
        pi[0][tag] = potential(model, words, 0, tag, START_TAG);
    }

    for (int i = 1; i < n; ++i) {
        for (int tag = 0; tag < T; ++tag) {
            for (int prev = 0; prev < T; ++prev) {
                double score = potential(model, words, i, tag, prev)
                        + pi[i-1][prev];

                // Update.
                if (score > pi[i][tag]) {
                    pi[i][tag] = score;
                    bp[i][tag] = prev;
                }
            }
        }
    }

    // Collect score;
    int final = -1;
    double final_score = -INF;
    for (int tag = 0; tag < T; ++tag) {
        double score = pi[n-1][tag] + model.theta[0][tag][END_TAG];
        if (score > final_score) {
            final_score = score;
            final = tag;
        }
    }


    // Walk back pointers.
    tagging->resize(n);
    int cur = final;
    for (int i = n-1; i >= 1; --i) {
        (*tagging)[i] = cur;
        // cout << i << " " << pi[i][cur] << endl;
        cur = bp[i][cur];

        assert(cur != -1);
    }
    (*tagging)[0] = cur;

    return final_score;
}
