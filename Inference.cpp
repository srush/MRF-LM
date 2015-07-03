#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <iostream>

#include "Model.h"
#include "core.h"
#include "Inference.h"

using namespace std;

Inference::Inference(Model *model) : model_(model) {
        // Set the options.
        dual_rate = opts.get<double>("dual-rate");
        dual_iter = opts.get<int>("dual-iter");
        mult_rate = opts.get<double>("mult-rate");
        keep_deltas = opts.exist("keep-deltas");

        delta.resize(model_->n_constraints());
        for (int c = 0; c < model->n_constraints(); c++) {
            int CS = model_->n_constraint_size(c);
            delta[c].resize(CS);
            for (int k = 0; k < CS; k++) {
                int l = model_->n_constraint_variable(c, k);
                delta[c][k].resize(model_->n_states(l));
                for (int s = 0; s < model_->n_states(l); s++) {
                    delta[c][k][s] = 0;
                }
            }
        }

        int L = model_->n_variables();
        psi .resize(L);
        expPsi.resize(L);
        mu1.resize(L);
        dualtheta0.resize(L);
        expdualtheta0.resize(L);

        for (int l = 0; l < L; l++) {
            int V = model_->n_states(l);
            psi[l] .resize(V);
            expPsi[l].resize(V+1);
            mu1[l].resize(V);
            dualtheta0[l].resize(V);
            expdualtheta0[l].resize(V+1);
        }

        mu2.resize(L-1);
        int n = 0;
        for (int l = 0; l < L; l++) {
            if (l == 0) continue;
            int V = model_->n_states(0);
            mu2[n].resize(V);
            for (int a = 0; a < V; a++) {
                mu2[n][a].resize(model_->n_states(l));
            }
            n++;
        }
}

// Message passing algorithm in real space with unary
// potentials theta0 and edge potentials (K+1)*theta
double Inference::bp(bool compute_pairwise) {
    // BP algorithm with precomputed exp.
    // Going up.
    int L = model_->n_variables();

    // Assume star graph structure. All connected to base.
    #pragma omp parallel for
    for (int s = 0; s < model_->n_states(0); s++) {
        psi[0][s] = dualtheta0[0][s];

        int n = 0;
        for (int l = 0; l < model_->n_variables(); l++) {
            if (l == 0) continue;
            double message = 0;
            for (int t = 0; t < model_->n_states(l); t++) {
                message += model_->expThetaRows[n][s][t]
                        * expdualtheta0[l][t];
            }
            // Add in expmax. (last dim).
            double logsum = log(message)
                    + model_->expThetaRows[n][s][model_->n_states(l)]
                    + expdualtheta0[l][model_->n_states(l)];

            psi[0][s] += logsum;
            psi[l][s] = -logsum;
            n++;
        }
        for (int l = 0; l < L; l++) {
            if (l == 0) continue;
            psi[l][s] += psi[0][s];
        }
    }

    // Partition function and \mu^0;
    double partition = logsumtab(psi[0], model_->n_states(0));

    for (int s = 0; s < model_->n_states(0); s++) {
        mu1[0][s] = exp(psi[0][s] - partition);
    }

    // Make expPsi
    #pragma omp parallel for
    for (int l = 0; l < L; l++) {
        exptab(psi[l], expPsi[l], model_->n_states(l));
    }

    // Going down
    if (compute_pairwise) {
        int n = 0;
        for (int l = 0; l < L; l++) {
            if (l == 0) continue;

            #pragma omp parallel for
            for (int s = 0; s < model_->n_states(0); s++) {
                for (int t = 0; t < model_->n_states(l); t++) {
                    mu2[n][s][t] = exp(psi[l][s]
                                       + model_->n_trees()*model_->theta[n][s][t]
                                       + dualtheta0[l][t]
                                       - partition);
                }
            }
            n++;
        }
    }

    int n = 0;
    for (int l = 0; l < L; l++) {
        if (l == 0) continue;
        #pragma omp parallel for
        for (int t = 0; t < model_->n_states(l); t++) {
            double message = 0;
            for (int s = 0; s < model_->n_states(0); s++) {
                message += expPsi[l][s] * model_->expThetaCols[n][t][s];
            }
            mu1[l][t] = message * exp(expPsi[l][model_->n_states(l)]
                                      + model_->expThetaCols[n][t][model_->n_states(0)]
                                      + dualtheta0[l][t]
                                      - partition);
        }
        n++;
    }
    return partition;
}

// This checks the pseudo-marginals, and returns how far from the primal
// feasible space they are.
double Inference::checkmus() {
    int L = model_->n_variables();
    double res = 0;

    for (int l = 0; l < L; l++) {
        for (int k = 0; k < L; k++) {
            if (!model_->constrained(l, k)) continue;
            double difs = 0;
            for (int s = 0; s < model_->n_states(l); s++) {
                difs += pow(mu1[l][s] - mu1[k][s], 2);
            }
            res += sqrt(difs);
        }
    }
    return res;
}

void Inference::MakeFullGradient(const Moments &train_moments) {
    int L = model_->n_variables();

    int n = 0;
    for (int l = 0; l < L; l++) {
        if (l == 0) continue;
        #pragma omp parallel for
        for (int s = 0; s < model_->n_states(0); s++) {
            for (int t = 0; t < model_->n_states(l); t++) {
                model_->grad_theta[n][s][t] = -mu2[n][s][t];
            }
        }
        n++;
    }

    n = 0;
    for (int l = 0; l < L; l++) {
        if (l == 0) continue;
        for (int pa = 0; pa < train_moments.nPairs[n]; pa++) {
            const vector<int> &p = train_moments.Pairs[n][pa];
            model_->grad_theta[n][p[0]][p[1]] +=
                    ((double)p[2]) / train_moments.N;
        }
        n++;
    }
}


// Compute the dual potentials of the model.
void Inference::MakeDualPotentials() {
    int L = model_->n_variables();

    // Add deltas to unary potentials.
    for (int c = 0; c < model_->n_constraints(); c++) {
        int l0 = model_->n_constraint_base(c);
        for (int s = 0; s < model_->n_states(l0); s++) {
            dualtheta0[l0][s] = 0.0;
        }

        for (int k = 0; k < model_->n_constraint_size(c); k++) {
            int l = model_->n_constraint_variable(c, k);
            for (int s = 0; s < model_->n_states(l); s++) {
                dualtheta0[l][s]   = delta[c][k][s];
                dualtheta0[l0][s] -= delta[c][k][s];
            }
        }
    }

    // This makes exptheta0, which is necessary for bp.
    #pragma omp parallel for
    for (int l = 0; l < L; l++) {
        exptab(dualtheta0[l], expdualtheta0[l],
               model_->n_states(l));
    }
}


// This function performs at most dual-iter steps of sub-gradient
// descent on the dual objective, then projects the obtained marginals to
// primal feasible space and returns the last value of the dual objective.
double Inference::DualInf() {
    double part, part_old = 1e20;
    double difmus = 10;
    double run_dual_rate = dual_rate;
    int C = model_->n_constraints();
    if (!keep_deltas) {
        for (int c = 0; c < C; c++) {
            for (int k = 0; k < model_->n_constraint_size(c); k++) {
                int l = model_->n_constraint_variable(c, k);
                for (int s = 0; s < model_->n_states(l); s++) {
                    delta[c][k][s] = 0;
                }
            }
        }
    }

    // We take sub-gradient steps to solve the dual decomposition
    for (int iter = 0; difmus > 1e-5 && iter < dual_iter; iter++) {
        // We make theta0 and run belief propagation with the current deltas.
        MakeDualPotentials();
        part = bp(false);
        difmus = checkmus();
        if (part > part_old + 1e-5) {
            run_dual_rate *= mult_rate;
        }

        if (run_dual_rate < 1e-7) break;

        // Then take a gradient step in deltas
        for (int c = 0; c < C; c++) {
            int l0 = model_->n_constraint_base(c);
            for (int k = 0; k < model_->n_constraint_size(c); k++) {
                int l = model_->n_constraint_variable(c, k);
                for (int s = 0; s < model_->n_states(l); s++) {
                    double grad_delta = mu1[l][s] - mu1[l0][s];
                    delta[c][k][s] -= run_dual_rate * grad_delta;
                }
            }
        }
        part_old = part;

        printf("Iter %d   \t difmus %f part %f rate %f\r",
               iter, difmus, part, run_dual_rate);
        fflush(stdout);
    }
    printf("Difmus %.3e  \n", difmus);

    // We then run inference with the final deltas, and project the marginals.
    return bp(true);
}
