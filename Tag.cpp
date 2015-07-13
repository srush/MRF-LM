#include "Tag.h"
#include <assert.h>
#include <fstream>

inline int map_pair(int k, int t1, int t2, int K, int T, int M, int F) {
    return T * T * k + T * t1 + t2  + M * F * T;
}

inline int map_feature(int m, int f, int t, int M, int F, int T) {
    return m * F * T + t * F + f;
}

void TagFeatures::Init() {
    Tag::Init();

    // Total number of parameters.
    M = 0;
    linear.resize(MR);
    for (int m = 0; m < MR; ++m) {
        linear[m].resize(T);
        for (int t = 0; t < T; ++t) {
            linear[m][t].resize(F);
        }
    }
    M += MR * F * T;

    pair.resize(K);
    for (int k = 0; k < K; ++k) {
        pair[k].resize(T);
        for (int t = 0; t < T; ++t) {
            pair[k][t].resize(T, 0);
        }
    }
    M += K * T * T;

    ExpandModel();
}



// Construct unfactorized thetas.
void TagFeatures::ExpandModel() {
    for (int k = 0; k < K; k++) {
        #pragma omp parallel for
        for (int s = 0; s < T; s++) {
            for (int t = 0; t < T; t++) {
                theta[k][s][t] = pair[k][s][t];
            }
        }
    }
    for (int m = 0; m < MR; ++m) {
        #pragma omp parallel for
        for (int s = 0; s < T; s++) {
            for (int t = 0; t < V; ++t) {
                theta[K+m][s][t] = 0.0;
                for (int i = 0; i < n_features(m, t); ++i) {
                    int f = n_feature(m, t, i);
                    theta[K+m][s][t] += linear[m][s][f];
                }

            }
        }
    }
    Exponentiate();
}

// Backprop gradient to low rank terms and flatten low-rank gradient
// into a vector.
void TagFeatures::BackpropGradient(double *grad) {
    for (int k = 0; k < K; k++) {
        for (int s = 0; s < T; s++) {
            for (int t = 0; t < T; t++) {
                int key = map_pair(k, s, t, K, T, MR, F);
                grad[key] = grad_theta[k][s][t];
            }
        }
    }
    for (int m = 0; m < MR; m++) {
        for (int s = 0; s < T; s++) {
            for (int t = 0; t < V; t++) {
                for (int i = 0; i < n_features(m, t); ++i) {
                    int f = n_feature(m, t, i);
                    int key = map_feature(m, f, s, MR, F, T);

                    int l = K + m;
                    grad[key] += grad_theta[l][s][t];
                }
            }
        }
    }
}

// Set low-rank terms based on flat weight vector.
void TagFeatures::SetWeights(const double *const_weight, bool reverse) {
    double *weight = (double *)const_weight;

    for (int k = 0; k < K; k++) {
        for (int s = 0; s < T; s++) {
            for (int t = 0; t < T; t++) {
                int key = map_pair(k, s, t, K, T, MR, F);
                if (!reverse) {
                    pair[k][s][t] = weight[key];
                } else {
                    assert(key < M);
                    weight[key] = pair[k][s][t];
                }
            }
        }
    }

    for (int m = 0; m < MR; ++m) {
        #pragma omp parallel for
        for (int f = 0; f < F; ++f) {
            for (int t = 0; t < T; ++t) {
                int key = map_feature(m, f, t, MR, F, T);
                if (!reverse) {
                    linear[m][t][f] = weight[key];
                } else {
                    assert(key < M);
                    weight[key] = linear[m][t][f];
                }
            }
        }
    }

    ExpandModel();
}


void TagFeatures::ReadFeatures(string feature_file) {
    ifstream myfile(feature_file);
    int npairs;

    myfile >> F >> npairs >> MR;

    int word, m, nfeats;
    features.resize(MR);
    for (int m = 0; m < MR; ++m) {
        features[m].resize(V);
    }

    cout << "npairs: " << npairs << endl;
    for (int p = 0; p < npairs; ++p) {
        myfile >> m >> word >> nfeats;
        if (word > features[m].size()) {
            features[m].resize(word+1);
        }
        features[m][word].resize(nfeats);
        for (int f = 0; f < nfeats; ++f) {
            myfile >> features[m][word][f];
        }
    }
}
