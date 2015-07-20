#include <assert.h>
#include <fstream>
#include <iomanip>

#include "LMTest.h"
#include "Model.h"
#include "LM.h"
#include "core.h"

LMTest::LMTest(string mf, bool print) : print_(print) {
    ifstream myfile(mf);
    int lsent, nsent;
    myfile >> nsent;

    sentences.resize(nsent);

    for (int i = 0; i < nsent; i++) {
        myfile >> lsent;
        sentences[i].resize(lsent);
        for (int j = 0; j < lsent; j++) {
            myfile >> sentences[i][j];
        }
    }
    myfile.close();
}

double LMTest::TestModel(Model &model) {
    LM *lmodel = (LM *)&model;
    scores.resize(sentences.size());

    for (int i = 0; i < sentences.size(); i++) {
        scores[i].resize(2, 0.0);
        SentLL(*lmodel, sentences[i], &scores[i]);
    }

    double l = 0, ll = 0;
    int nwords = 0;
    for (int i = 0; i < sentences.size(); i++) {
        nwords += sentences[i].size() - 2 * lmodel->K;
        l += scores[i][0];
        ll += scores[i][1];
    }

    printf("The average pseudo log-likelihood over %d words is: \t \t \t  ll: %.3e \n",
            nwords, ll/nwords);

    return ll / nwords;
}

double LMTest::WordLL(LM &model,
                      const vector<int> &lcontext,
                      const vector<int> &rcontext,
                      int word,
                      vector<float> *logres) {
    // This function computes the log likelihood of a word
    // given its context
    double partition;

    #pragma omp parallel for
    for (int s = 0; s < model.V; s++) {
        (*logres)[s] = 0;
        for (int k = 0; k < model.K; k++) {
            (*logres)[s] += model.theta[k][lcontext[k]][s];
            (*logres)[s] += model.theta[k][s][rcontext[k]];
        }
    }

    partition = logsumtab(logres->data(), model.V);

    return (*logres)[word] - partition;
}

// This function returns the sum of conditional log-likelihoods
// of words in sentence 'sent'
void LMTest::SentLL(LM &model, vector<int> sent, vector<double> *score) {
    vector<int> lcontext(model.K);
    vector<int> rcontext(model.K);
    double ll = 0;
    vector<float> logres(model.V);
    for (int i = model.K; i < (int)sent.size() - model.K; i++) {
        for (int k = 0; k < model.K; k++) {
            lcontext[k] = sent[(i - 1) - k];
        }
        for (int k = 0; k < model.K; k++) {
            rcontext[k] = sent[(i + 1) + k];
        }


        ll = WordLL(model, lcontext, rcontext, sent[i], &logres);
        if (print_) {
            cout << exp(ll) << " ";
            for (int k = model.K - 1; k >= 0; k--) {
                cout << model.dict[0][lcontext[k]] << " ";
            }
            cout << " {" << model.dict[0][sent[i]] << "} ";
            for (int k = 0; k < model.K; k++) {
                cout << model.dict[0][rcontext[k]] << " ";
            }
            cout << endl;
        }

        (*score)[0] += exp(ll);
        (*score)[1] += ll;
    }
}
