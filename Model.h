#ifndef MODEL_H
#define MODEL_H

#include "cmdline.h"
#include <string>

using namespace std;

extern cmdline::parser opts;

struct Moments {
    int N, L;
    int *sizes;
    int *nPairs, ***Pairs;
};

extern struct Moments train_moments;
extern struct Moments valid_moments;

// Read moments from disk.
void ReadMoments(string mf, Moments *);

class Model {
public:
    // Initialize the model. Constructs arrays and
    // randomly initlizes params.
    Model() {}
    void Init();

    // The the objetive score of moments, given the paritition estimate.
    double ComputeObjective(const Moments &moments,
                            double partition);


    // Read and write model from disk.
    void ReadModel(string mf);
    void WriteModel(string mf);

    // The default implementation use a full rank MRF.
    virtual void SetWeights(const double *const_weight, bool reverse);
    virtual void BackpropGradient(double *grad);

    virtual int n_states(int i) = 0;
    virtual int n_variables() = 0;
    virtual int n_constraints() = 0;
    virtual int n_constraint_size(int cons) = 0;
    virtual int n_constraint_base(int cons) = 0;
    virtual int n_constraint_variable(int cons, int i) = 0;
    virtual int constrained(int cons, int i) = 0;
    virtual int n_trees() = 0;


    // Full model potentials.
    // \theta \in R^{KxVxV}
    // \grad_theta \in R^{KxVxV}
    double ***theta,  ***grad_theta;

    // Exponentiated full potentials and transposition.
    // \exp(\theta_k) and  \exp(\theta_k^\T)
    double ***expThetaRows, ***expThetaCols;

    // Number of parameters of the model.
    int M;

protected:
    void Exponentiate();
};


#endif
