#ifndef LM_H
#define LM_H

#include <vector>
#include "Model.h"

using namespace std;

typedef vector<vector<vector<double> > > vectorThree;
typedef vector<vector<double> > vectorTwo;

class LM : public Model {
public:
    LM(int _K, int _V) : K(_K), V(_V) {
        Init();
    }


    int n_states(int i) { return V; }
    int n_variables() { return K + 1; }
    int n_constraints() { return 1; }
    int n_constraint_size(int cons) { return K; }
    int n_constraint_base(int cons) { return 0; }
    int n_constraint_variable(int cons, int i) { return i+1; }
    int n_trees() { return K + 1; }

    int constrained(int i, int j) { return j != i; }

    int K, V;
};

class LMLowRank : public LM {
public:
    LMLowRank(int _K, int _V, int par_D);


    // Convert grad_theta to a flattened version
    // for parameters update.
    void BackpropGradient(double *grad);

    // Update low-rank parameters-based on a
    // new flat weight vector.
    void SetWeights(const double *const_weight,
                    bool reverse);

private:
    void ExpandModel();

    // Key model sizes.
    double **transpose;
    vector<double> B;
    vectorTwo U;
    vectorThree Wl;
    int D;
};

#endif
