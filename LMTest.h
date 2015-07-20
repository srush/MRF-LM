#ifndef LMTEST_H
#define LMTEST_H
#include "Test.h"
#include "Model.h"
#include "LM.h"

class LMTest : public Test {
public:
    LMTest(string mf, bool);
    double TestModel(Model &model);

private:
    void SentLL(LM &model, vector<int> sent, vector<double> *score);
    double WordLL(LM &model, const vector<int> &, const vector<int> &, int word,
                  vector<float> *logres);

    vector<vector<int> > sentences;
    vector<vector<double> > scores;
    bool print_;
};

#endif
