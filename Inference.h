#ifndef INFERENCE_H
#define INFERENCE_H

#include "Model.h"

class Inference {
public:
    Inference(Model *model);

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
