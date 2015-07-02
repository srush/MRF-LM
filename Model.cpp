#include "Model.h"

#include <math.h>
#include <omp.h>
#include <assert.h>
#include "core.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

cmdline::parser opts;

inline int map_key(int k, int v1, int v2, int V1, int V2) {
    return k * V1 * V2 + v1 * V2 + v2;
}

void Model::Init() {
    int L = n_variables();
    int K = L-1;

    theta = new double**[K];
    grad_theta = new double**[K];
    expThetaRows = new double**[K];
    expThetaCols = new double**[K];
    int n = 0;
    M = 0;
    for (int l = 0; l < L; l++) {
        if (l == 0) continue;
        theta[n] = new double*[n_states(0)];
        grad_theta[n] = new double*[n_states(0)];
        expThetaRows[n] = new double*[n_states(0)];

        for (int a = 0; a < n_states(0); a++) {
            theta[n][a] = new double[n_states(l)];
            grad_theta[n][a] = new double[n_states(l)];
            expThetaRows[n][a] = new double[n_states(l)+1];

            for (int b = 0; b < n_states(l); b++) {
                theta[n][a][b] = 0;
                // Another param.
                M++;
                grad_theta[n][a][b] = 0;
            }
        }

        // Transpose of expThetaRows.
        expThetaCols[n] = new double*[n_states(l)];
        for (int a = 0; a < n_states(l); a++) {
            expThetaCols[n][a] = new double[n_states(0)+1];
        }
        n++;
    }
}


void Model::BackpropGradient(double *grad) {
    // Full rank model.
    // Backprop is trivial.
    int n = 0;
    for (int l = 0; l < n_variables(); l++) {
        if (l == 0) continue;
        #pragma omp parallel for
        for (int s = 0; s < n_states(0); s++) {
            for (int t = 0; t < n_states(l); t++) {
                int key = map_key(n, s, t, n_states(0), n_states(l));
                grad[key] = grad_theta[n][s][t];
            }
        }
        ++n;
    }
}

void Model::SetWeights(const double *const_weight, bool reverse) {
    // Full rank model.
    // Setweights is trivial.

    double *weight = (double *)const_weight;

    int n = 0;
    for (int l = 0; l < n_variables(); l++) {
        if (l == 0) continue;
        #pragma omp parallel for
        for (int s = 0; s < n_states(0); s++) {
            for (int t = 0; t < n_states(l); t++) {
                int key = map_key(n, s, t, n_states(0), n_states(l));
                if (!reverse) {
                    theta[n][s][t] = weight[key];
                } else {
                    weight[key] = theta[n][s][t];
                }
            }
        }
        ++n;
    }
    Exponentiate();
}


void Model::Exponentiate() {
    cout << "EXPONENTIATE" << endl;
    int n = 0;
    for (int l = 0; l < n_variables(); l++) {
        if (l == 0) continue;
        double *temp = new double[n_states(l)];

        for (int s = 0; s < n_states(0); s++) {
            #pragma omp parallel for
            for (int t = 0; t < n_states(l); t++) {
                temp[t] = n_trees() * theta[n][s][t];
            }
            exptab(temp, expThetaRows[n][s], n_states(l));
        }
        delete temp;

        temp = new double[n_states(0)];
        for (int t = 0; t < n_states(l); t++) {
            #pragma omp parallel for
            for (int s = 0; s < n_states(0); s++) {
                temp[s] = n_trees() * theta[n][s][t];
            }
            exptab(temp, expThetaCols[n][t], n_states(0));
        }
        delete temp;
        ++n;
    }
}

// Read the model file from disk.
void Model::ReadModel(string mf) {
    ifstream myfile(mf);
    myfile >> M;
    double *weights = new double[M];
    for (int m = 0; m < M; ++m) {
        myfile >> weights[m];
    }

    myfile.close();
    SetWeights(weights, false);
}

// Write out the global model file to disk.
void Model::WriteModel(string mf) {
    ofstream myfile(mf);
    myfile << M << "\n"; // << K << " " << V << " " << D << " ";
    double *weights = new double[M];
    SetWeights(weights, true);
    for (int m = 0; m < M; ++m) {
        myfile << weights[m] << " ";
    }
    myfile.close();
}

// This function returns the value of the log-likelihood lower bound
// given the current log-partition
double Model::ComputeObjective(const Moments &mom,
                double partition) {
    double res = 0;
    for (int k = 0; k < mom.L - 1; k++) {
        for (int pa = 0; pa < mom.nPairs[k]; pa++) {
            int *p = mom.Pairs[k][pa];
            res += theta[k][p[0]][p[1]] * ((double)p[2]) / mom.N;
        }
    }
    printf("---Energy %.4e, Partition %.4e \t \t \t \t \t >>> OBJECTIVE %.4e\n",
           res, partition / (n_variables()),
           res - partition / (n_variables()));
    return res - partition / (n_variables());
}

struct Moments train_moments;
struct Moments valid_moments;

// Read the moments text file.
void ReadMoments(string mf, Moments *moments) {
    ifstream myfile(mf);
    myfile >> moments->N >> moments->L;
    moments->sizes = new int[moments->L];
    for (int k = 0; k < moments->L; k++) {
            myfile >> moments->sizes[k];
    }
    moments->nPairs = new int[moments->L-1];
    moments->Pairs = new int**[moments->L-1];
    for (int k = 0; k < moments->L - 1; k++) {
        myfile >> moments->nPairs[k];
        printf("k=%d, npairs=%d\n", k, moments->nPairs[k]);
        moments->Pairs[k] = new int*[moments->nPairs[k]];
        for (int i = 0; i < moments->nPairs[k]; i++) {
            moments->Pairs[k][i] = new int[3];
            for (int j = 0; j < 3; j++) {
                myfile >> moments->Pairs[k][i][j];
            }
        }
    }
    myfile.close();
}


// void ReadMoments(string mf, Moments *moments) {
//     ifstream myfile(mf);
//     myfile >> moments->N >> moments->V >> moments->K;

//     // moments->counts = new int[moments->V];
//     // for (int i = 0; i < moments->V; i++) {
//     //     myfile >> moments->counts[i];
//     // }
//     moments->nPairs = new int[moments->K];
//     moments->Pairs = new int**[moments->K];
//     for (int k = 0; k < moments->K; k++) {
//         myfile >> moments->nPairs[k];
//         printf("k=%d, npairs=%d\n", k, moments->nPairs[k]);
//         moments->Pairs[k] = new int*[moments->nPairs[k]];
//         for (int i = 0; i < moments->nPairs[k]; i++) {
//             moments->Pairs[k][i] = new int[3];
//             for (int j = 0; j < 3; j++) {
//                 myfile >> moments->Pairs[k][i][j];
//             }
//         }
//     }
//     myfile.close();
// }