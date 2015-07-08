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

void Model::Init() {
    int L = n_variables();
    int K = L-1;

    theta.resize(K);
    theta_key.resize(K);
    grad_theta.resize(K);
    expThetaRows.resize(K);
    expThetaCols.resize(K);
    int n = 0;
    M = 0;
    for (int l = 0; l < L; l++) {
        if (l == 0) continue;
        theta[n].resize(n_states(0));
        theta_key[n].resize(n_states(0));
        grad_theta[n].resize(n_states(0));
        expThetaRows[n].resize(n_states(0));

        for (int a = 0; a < n_states(0); a++) {
            theta[n][a].resize(n_states(l), 0.0);
            theta_key[n][a].resize(n_states(l));
            grad_theta[n][a].resize(n_states(l), 0.0);
            expThetaRows[n][a].resize(n_states(l)+1);

            for (int b = 0; b < n_states(l); b++) {
                theta[n][a][b] = 0;
                // Another param.
                theta_key[n][a][b] = M;
                M++;
                grad_theta[n][a][b] = 0;
            }
        }

        // Transpose of expThetaRows.
        expThetaCols[n].resize(n_states(l));
        for (int a = 0; a < n_states(l); a++) {
            expThetaCols[n][a].resize(n_states(0)+1);
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
                int key = theta_key[n][s][t];
                grad[key] = grad_theta[n][s][t];
                ++key;
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
                // int key = map_key(n, s, t, n_states(0), n_states(l));
                int key = theta_key[n][s][t];
                if (!reverse) {
                    theta[n][s][t] = weight[key];
                } else {
                    weight[key] = theta[n][s][t];
                }
                ++key;
            }
        }
        ++n;
    }
    Exponentiate();
}

void Model::Exponentiate() {
    int n = 0;
    for (int l = 0; l < n_variables(); l++) {
        if (l == 0) continue;
        vector<double> temp(n_states(l));
        #pragma omp parallel for
        for (int s = 0; s < n_states(0); s++) {
            for (int t = 0; t < n_states(l); t++) {
                expThetaRows[n][s][t] = n_trees() * theta[n][s][t];
            }
            exptab(expThetaRows[n][s],
                   expThetaRows[n][s], n_states(l));
        }


        #pragma omp parallel for
        for (int t = 0; t < n_states(l); t++) {
            for (int s = 0; s < n_states(0); s++) {
                expThetaCols[n][t][s] = n_trees() * theta[n][s][t];
            }
            exptab(expThetaCols[n][t],
                   expThetaCols[n][t], n_states(0));
        }
        ++n;
    }
}

// Read the model file from disk.
void Model::ReadModel(string mf) {
    ifstream myfile(mf);
    ReadParams(myfile);
    Init();
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
    WriteParams(myfile);
    myfile << M << "\n";
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
            const vector<int> &p = mom.Pairs[k][pa];
            res += theta[k][p[0]][p[1]] * ((double)p[2]) / mom.N;
        }
    }
    printf("---Energy %.4e, Partition %.4e \t \t \t \t \t >>> OBJECTIVE %.4e\n",
           res, partition / (n_trees()),
           res - partition / (n_trees()));
    return res - partition / (n_trees());
}

struct Moments train_moments;
struct Moments valid_moments;

// Read the moments text file.
void ReadMoments(string mf, Moments *moments) {
    ifstream myfile(mf);
    myfile >> moments->N >> moments->L;
    moments->sizes.resize(moments->L);
    for (int k = 0; k < moments->L; k++) {
        myfile >> moments->sizes[k];
    }
    moments->nPairs.resize(moments->L-1);
    moments->Pairs.resize(moments->L-1);
    for (int k = 0; k < moments->L - 1; k++) {
        myfile >> moments->nPairs[k];
        printf("k=%d, npairs=%d\n", k, moments->nPairs[k]);
        moments->Pairs[k].resize(moments->nPairs[k]);
        for (int i = 0; i < moments->nPairs[k]; i++) {
            moments->Pairs[k][i].resize(3);
            for (int j = 0; j < 3; j++) {
                int p;
                myfile >> p;
                moments->Pairs[k][i][j] = p;
                assert(p >= 0);
            }
        }
    }
    myfile.close();
}
