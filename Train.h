#ifndef TRAIN_H
#define TRAIN_H

#include "Test.h"
#include "Model.h"
#include "Inference.h"

class Train {
public:
Train(Model *model, Inference *inf, Test *test) :
    model_(model), inf_(inf), test_(test), best_valid_score(-1e10) {}

    // Run LBFGS Training.
    void LBFGS();
    void compute_progress();
    double compute_gradient(const double *, double *gradient);

private:
    Model *model_;
    Inference *inf_;
    Test *test_;
    double valid_score, best_valid_score;
    double current_partition;

};

#endif
