#include "Tag.h"

inline int map_pair(int k, int t1, int t2, int K, int T, int M, int F) {
    return T * T * k + T * t1 + t2  + M * F * T;
}

inline int map_feature(int m, int f, int t, int M, int F, int T) {
    return m * F * T + t * F + f;
}

TagFeatures::TagFeatures(int _K, int _V, int _M, int _T, int _F)
        : Tag(_K, _V, _M, _T), F(_F) {
    // Total number of parameters.
    M = 0;
    linear.resize(M);
    for (int m = 0; m < M; ++m) {
        linear[m].resize(T);
        for (int t = 0; t < T; ++t) {
            linear[m][t].resize(F);
        }
    }
    M += M * F * T;

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
        for (int s = 0; s < V; s++) {
            for (int t = 0; t < V; t++) {
                theta[k][s][t] = pair[k][s][t];
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
                int key = map_pair(k, s, t, K, T, M, F);
                grad[key] = grad_theta[k][s][t];
            }
        }
    }
    for (int m = 0; m < M; m++) {
        for (int t = 0; t < V; t++) {
            for (int i = 0; i < n_features(m, t); ++i) {
                for (int s = 0; s < T; s++) {
                    int f = n_feature(m, t, i);
                    int key = map_feature(m, f, s, M, F, T);

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
                int key = map_pair(k, s, t, K, T, M, F);
                if (!reverse) {
                    pair[k][s][t] = weight[key];
                } else {
                    weight[key] = pair[k][s][t];
                }
            }
        }
    }

    for (int m = 0; m < M; ++m) {
        for (int f = 0; f < F; ++f) {
            for (int t = 0; t < T; ++t) {
                int key = map_feature(m, f, t, M, F, T);
                if (!reverse) {
                    linear[m][f][t] = weight[key];
                } else {
                    weight[key] = linear[m][f][t];
                }
            }
        }
    }

    ExpandModel();
}
