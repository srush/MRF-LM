#include <lbfgs.h>

#include "Model.h"
#include "Inference.h"
#include "Train.h"
#include "Test.h"

typedef const lbfgsfloatval_t real;

static int progress(void *instance, real *x, real *g, real fx,
                    real xnorm, real gnorm, real step, int n,
                    int k, int ls) {
    Train *train = (Train *)instance;
    printf("Iteration %d:\n", k);
    printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");

    train->compute_progress();

    return 0;
}

static lbfgsfloatval_t evaluate(void *instance, real *x,
                                lbfgsfloatval_t *g,
                                const int n,
                                real step) {
    Train *train = (Train *)instance;

    // Set up the problem from x.
    printf("Step %f\n", step);
    for (int i = 0; i < n; ++i) {
        g[i] = 0;
    }

    lbfgsfloatval_t fx = train->compute_gradient(x, g);
    for (int i = 0; i < n; ++i) {
        g[i] = -g[i];
    }

    printf("ret: %f \n", fx);
    return -fx;
}

void Train::compute_progress() {
    if (test_ != NULL) {
        printf("Running Validation: \n");
        valid_score = test_->TestModel(*model_);
        if (opts.exist("valid")) {
            printf("VALIDATION OBJECTIVE: \n");
            model_->ComputeObjective(valid_moments, current_partition);
        }

    } else {
        if (opts.exist("valid")) {
            printf("VALIDATION OBJECTIVE: \n");
            valid_score = model_->ComputeObjective(valid_moments,
                                                   current_partition);
        }
    }
    cout << valid_score << endl;
    printf("Validation Score: %f \n\n", valid_score);
    if (valid_score > best_valid_score) {
        printf("Writing Model \n");
        model_->WriteModel(opts.get<string>("output"));
        best_valid_score = valid_score;
    }


}

double Train::compute_gradient(const double *x,
                               double *gradient) {
    model_->SetWeights(x, false);

    // This runs dual decomposition.
    double part = inf_->DualInf();
    double score = model_->ComputeObjective(train_moments, part);
    current_partition = part;

    // Create the full gradient and back-prop to low-rank parameters.
    inf_->MakeFullGradient(train_moments);
    model_->BackpropGradient(gradient);
    return score;
}

void Train::LBFGS() {
    lbfgsfloatval_t *x = lbfgs_malloc(model_->M);
    printf("Allocated x, M= %d\n", model_->M);
    model_->SetWeights(x, true);
    printf("Weights x\n");

    lbfgsfloatval_t fx;
    lbfgs_parameter_t param;
    printf("L-BFGS \n");
    lbfgs_parameter_init(&param);
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    param.epsilon = 1e-10;

    int ret = lbfgs(model_->M, x, &fx, evaluate, progress, this, &param);

    printf("Result: %f Code: %d %d", fx, ret, LBFGSERR_MAXIMUMLINESEARCH);

    lbfgs_free(x);
}
