#include "LM.h"
#include <cmath>
inline int map_pair(int k, int t, int d, int D, int V) {
    return V + (k+1)*V*D + t*D + d;
}

inline int map_word(int s, int d, int D, int V) {
    return V + s*D + d;
}

inline int map_uni(int v) {
    return v;
}

void LMLowRank::Init() {
    LM::Init();

    // Total number of parameters.
    M = 0;
    B.resize(V);
    for (int s = 0; s < V; s++) {
        B[s] = .5 * (rand() / (double)RAND_MAX - 0.5) / V;
    }
    M += V;


    U.resize(V);
    #pragma omp parallel for
    for (int s = 0; s < V; s++) {
        U[s].resize(D);
        for (int d = 0; d < D; d++) {
            U[s][d] = .5 * (rand() / (double)RAND_MAX - 0.5) / sqrt(D);
        }
    }
    M += V * D;

    Wl.resize(K);
    for (int k = 0; k < K; k++) {
        Wl[k].resize(V);
        #pragma omp parallel for
        for (int s = 0; s < V; s++) {
            Wl[k][s].resize(D);
            for (int d = 0; d < D; d++) {
                Wl[k][s][d] = .5 * (rand() / (double)RAND_MAX - 0.5) / sqrt(D);
            }
        }
    }
    M += K * V * D;
    transpose = new double*[V];
    for (int a = 0; a < V; a++) {
        transpose[a] = new double[V];
    }
    ExpandModel();
}



// Construct unfactorized thetas.
void LMLowRank::ExpandModel() {
    for (int k = 0; k < K; k++) {
        #pragma omp parallel for
        for (int s = 0; s < V; s++) {
            for (int t = 0; t < V; t++) {
                theta[k][s][t] = B[t];
                for (int i = 0; i < D; i++) {
                    theta[k][s][t] += U[s][i] * Wl[k][t][i];
                }
            }
        }
    }
    Exponentiate();
}


// Backprop gradient to low rank terms and flatten low-rank gradient
// into a vector.
void LMLowRank::BackpropGradient(double *grad) {
    // We compute the gradients in U

    #pragma omp parallel for
    for (int t = 0; t < V; t++) {
        int key = map_uni(t);
        grad[key] = 0;
        for (int s = 0; s < V; s++) {
            for (int k = 0; k < K; k++) {
                grad[key] += grad_theta[k][s][t];

            }
        }
    }


    #pragma omp parallel for
    for (int s = 0; s < V; s++) {
        for (int d = 0; d < D; d++) {
            int key = map_word(s, d, D, V);
            grad[key] = 0;
            for (int t = 0; t < V; t++) {
                for (int k = 0; k < K; k++) {
                    grad[key] += grad_theta[k][s][t] * Wl[k][t][d];
                }
            }
        }
    }

    // Then compute the gradients in Wl.
    for (int k = 0; k < K; k++) {
        for (int s = 0; s < V; s++) {
            for (int t = 0; t < V; t++) {
                transpose[s][t] = grad_theta[k][t][s];
            }
        }
        #pragma omp parallel for
        for (int t = 0; t < V; t++) {
            for (int d = 0; d < D; d++) {
                int key = map_pair(k, t, d, D, V);
                grad[key] = 0;
                for (int s = 0; s < V; s++) {
                    grad[key] += transpose[t][s] * U[s][d];
                }
            }
        }
    }
}

// Set low-rank terms based on flat weight vector.
void LMLowRank::SetWeights(const double *const_weight, bool reverse) {
    double *weight = (double *)const_weight;

    // And finally apply the gradients.
    #pragma omp parallel for
    for (int s = 0; s < V; s++) {
        int key = map_uni(s);
        if (!reverse) {
            B[s] = weight[key];
        } else {
            weight[key] = B[s];
        }

        for (int d = 0; d < D; d++) {
            int key = map_word(s, d, D, V);
            if (!reverse) {
                U[s][d] = weight[key];
            } else {
                weight[key] = U[s][d];
            }
        }
    }

    for (int k = 0; k < K; k++) {
        #pragma omp parallel for
        for (int t = 0; t < V; t++) {
            for (int d = 0; d < D; d++) {
                int key = map_pair(k, t, d, D, V);
                if (!reverse) {
                    Wl[k][t][d] = weight[key];
                } else {
                    weight[key] = Wl[k][t][d];
                }
            }
        }
    }
    ExpandModel();
}
