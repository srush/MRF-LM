#ifndef MODEL_H
#define MODEL_H

#include "cmdline.h"
#include <string>
#include <fstream>

using namespace std;

extern cmdline::parser opts;

struct Moments {
    int N, L;
    vector<int> sizes;
    vector<int> nPairs;
    vector<vector<vector<int> > > Pairs;
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
    virtual void Init();

    virtual void InitFromMoments(const Moments &moments) = 0;


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

    void set_labels(string vocab, int i) {
        ifstream in_vocab(vocab);
        dict.resize(i+1);
        while (in_vocab) {
            int index, count;
            string name;
            in_vocab >> index >> name >> count;
            dict[i].push_back(name);
        }
    }

    virtual void ReadParams(ifstream &myfile) = 0;
    virtual void WriteParams(ofstream &myfile) = 0;


    // Full model potentials.
    // \theta \in R^{KxVxV}
    // \grad_theta \in R^{KxVxV}
    vector<vector<vector<double> > > theta, grad_theta;
    vector<vector<vector<int> > > theta_key;

    // Exponentiated full potentials and transposition.
    // \exp(\theta_k) and  \exp(\theta_k^\T)
    vector<vector<vector<double> > > expThetaRows, expThetaCols;

    // Number of parameters of the model.
    int M;
    vector<vector<string> > dict;

protected:
    void Exponentiate();


};


#endif
