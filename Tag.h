#ifndef TAG_H
#define TAG_H
#include "Model.h"
#include <vector>

using namespace std;

class Tag : public Model {
public:
    Tag(int _K, int _V, int _M, int _T)
            : K(_K), V(_V), MR(_M), T(_T) {
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

    int K, V, MR, T;
};


class TagFeatures : public Tag {
public:
    TagFeatures(int _K, int _V, int _M, int _T,
                string feature_file);

    // Convert grad_theta to a flattened version
    // for parameters update.
    void BackpropGradient(double *grad);

    // Update feature parameters-based on a
    // new flat weight vector.
    void SetWeights(const double *const_weight,
                    bool reverse);

    int n_features(int m, int v) {
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
