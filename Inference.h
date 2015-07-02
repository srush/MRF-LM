#ifndef INFERENCE_H
#define INFERENCE_H

#include "Model.h"

class Inference {
public:
    Inference(Model *model) : model_(model) {
        // Set the options.
        dual_rate = opts.get<double>("dual-rate");
        dual_iter = opts.get<int>("dual-iter");
        mult_rate = opts.get<double>("mult-rate");
        keep_deltas = opts.exist("keep-deltas");

        delta = new double**[model_->n_constraints()];
        for (int c = 0; c < model->n_constraints(); c++) {
            int CS = model_->n_constraint_size(c);
            delta[c] = new double*[CS];
            for (int k = 0; k < CS; k++) {
                int l = model_->n_constraint_variable(c, k);
                delta[c][k] = new double[model_->n_states(l)];
                for (int s = 0; s < model_->n_states(l); s++) {
                    delta[c][k][s] = 0;
                }
            }
        }

        int L = model_->n_variables();
        psi = new double*[L];
        expPsi = new double*[L];
        mu1 = new double*[L];
        dualtheta0 = new double*[L];
        expdualtheta0 = new double*[L];

        for (int l = 0; l < L; l++) {
            int V = model_->n_states(l);
            psi[l] = new double[V];
            expPsi[l] = new double[V+1];
            mu1[l] = new double[V];
            dualtheta0[l] = new double[V];
            expdualtheta0[l] = new double[V+1];
        }

        mu2 = new double**[L-1];
        int n = 0;
        for (int l = 0; l < L; l++) {
            if (l == 0) continue;
            int V = model_->n_states(0);
            mu2[n] = new double*[V];
            for (int a = 0; a < V; a++) {
                mu2[n][a] = new double[model_->n_states(l)];
            }
            n++;
        }
    }

    // Minimize the dual.
    double DualInf();

    // Construct gradient with the current best dual values.
    void MakeFullGradient(const Moments &train_moments);

    double **dualtheta0;
private:
    double bp(bool compute_pairwise);
    double checkmus();
    void MakeDualPotentials();

    Model *model_;
    /* int K, V; */
    double **psi, ***delta, **mu1, ***mu2, **expPsi;

    // Dual unary potentials.
    double **expdualtheta0;

    double dual_rate;
    double mult_rate;
    int dual_iter;
    bool keep_deltas;

};


#endif
