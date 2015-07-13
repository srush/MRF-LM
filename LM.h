#ifndef LM_H
#define LM_H

#include <vector>
#include <fstream>
#include "Model.h"
#include <algorithm>
#include <cmath>
#include <assert.h>

using namespace std;

typedef vector<vector<vector<double> > > vectorThree;
typedef vector<vector<double> > vectorTwo;

class LM : public Model {
public:
    LM() { }
    virtual void InitFromMoments(const Moments &moments) {
        K = train_moments.L-1;
        V = train_moments.sizes[0];
        Init();
    }

    int n_states(int i) { return V; }
    int n_variables() { return K + 1; }
    int n_constraints() { return 1; }
    int n_constraint_size(int cons) { return K; }
    int n_constraint_base(int cons) { return 0; }
    int n_constraint_variable(int cons, int i) { return i+1; }
    int n_trees() { return K + 1; }

    int constrained(int i, int j) { return j != i; }
    void ReadParams(ifstream &myfile) {
        myfile >> K >> V;
    }

    void WriteParams(ofstream &myfile) {
        myfile << K << " " << V << endl;
    }
    int K, V;
};

class LMLowRank : public LM {
public:
    LMLowRank() {}
    LMLowRank(int D_): D(D_) {}
    virtual void Init();

    // Convert grad_theta to a flattened version
    // for parameters update.
    void BackpropGradient(double *grad);

    // Update low-rank parameters-based on a
    // new flat weight vector.
    void SetWeights(const double *const_weight,
                    bool reverse);

    void ReadParams(ifstream &myfile) {
        myfile >> K >> V >> D;
    }

    void WriteParams(ofstream &myfile) {
        myfile << K << " " << V << " " << D << endl;
    }

    void WriteEmbeddings(string emb_file) {
        ofstream output(emb_file);
        for (int v = 0; v < V; ++v) {
            output << dict[0][v] << " ";
            for (int i = 0; i < D; ++i) {
                output << U[v][i] << " ";
            }
            output << endl;
        }
    }

    void kNN(int word, int K, vector<int> *knn) {
        vector<pair<double, int> > scores(V);
        for (int v = 0; v < V; ++v) {
            if (v == word) continue;
            double dot = 0.0;
            double norm_v = 0.0, norm_w = 0.0;
            for (int d = 0; d < D; ++d) {
                dot += U[v][d] * U[word][d];
                norm_v += U[v][d] * U[v][d];
                norm_w += U[word][d] * U[word][d];
            }
            scores[v].first = dot / (sqrt(norm_v) * sqrt(norm_w));
            scores[v].second = v;
        }
        sort(scores.begin(), scores.end());
        reverse(scores.begin(), scores.end());
        knn->resize(K);
        for (int k = 0; k < K; ++k) {
            (*knn)[k] = scores[k].second;
        }
    }

    void WriteKNN(string emb_file, int K) {
        ofstream output(emb_file);
        vector<int> near;
        for (int w = 0; w < V; ++w) {
            output << dict[0][w] << endl;
            kNN(w, K, &near);
            for (int k = 0; k < K; ++k) {
                int v = near[k];
                output << "\t" << dict[0][v] << endl;
            }
        }
    }


private:
    void ExpandModel();

    // Key model sizes.
    double **transpose;
    vector<double> B;
    vectorTwo U;
    vectorThree Wl;
    int D;
};

#endif
