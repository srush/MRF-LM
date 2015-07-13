#ifndef TAG_H
#define TAG_H

#include "Model.h"
#include <vector>
#include <fstream>

using namespace std;

class Tag : public Model {
public:
    Tag() {}

    virtual void InitFromMoments(const Moments &moments) {
        K = 1;
        V = moments.sizes[moments.L-1];
        MR = moments.L-2;
        T = moments.sizes[0];
        Init();
    }

    int n_variables() { return K + MR + 1; }
    int n_states(int i) {
        if (i <= K) return T;
        else return V;
    }

    int n_constraints() { return 2; }
    int n_constraint_size(int cons) {
        if (cons == 0) return K;
        else return MR-1;
    }

    int n_constraint_base(int cons) {
        if (cons == 0) return 0;
        else return K + 1;
    }
    int n_constraint_variable(int cons, int i) {
        if (cons == 0) return i+1;
        else return (K + 1) + i + 1;
    }

    int constrained(int i, int j) {
        if (i <= K && j <= K && j != i) return true;
        if (i > K && j > K && j != i) return true;
        return false;
    }
    int n_trees() { return max(K+1, MR); }

    void ReadParams(ifstream &myfile) {
        myfile >> K  >> V >> MR >> T;
    }

    void WriteParams(ofstream &myfile) {
        myfile << K  << " " << V << " " << MR << " "  << T << endl;
    }

    int K, V, MR, T;
};


class TagFeatures : public Tag {
public:
    virtual void Init();

    // Convert grad_theta to a flattened version
    // for parameters update.
    void BackpropGradient(double *grad);

    // Update feature parameters-based on a
    // new flat weight vector.
    void SetWeights(const double *const_weight,
                    bool reverse);

    int n_features(int m, int v) {
        if (v >= features[m].size()) return 0;
        return features[m][v].size();
    }
    int n_feature(int m, int v, int i) {
        return features[m][v][i];
    }

    void ReadFeatures(string feature_file);

private:
    void ExpandModel();

    // Features
    int F;

    vector<vector<vector<double> > > linear;
    vector<vector<vector<double> > > pair;
    vector<vector<vector<int> > > features;
};

#endif
