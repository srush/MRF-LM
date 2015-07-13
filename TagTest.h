#ifndef TAGTEST_H
#define TAGTEST_H
#include "Test.h"
#include "Model.h"
#include "Tag.h"

class TagTest : public Test {
public:
    TagTest(string mf, bool);
    double TestModel(Model &model);
    double potential(Tag &model,
                     const vector<int> &words, int i, int tag, int prev);
    double Viterbi(Tag &model, const vector<int> &words,
                   vector<int> *tagging,
                   int T);

private:
    vector<vector<int> > words;
    vector<vector<int> > taggings;
    vector<vector<int> > tag_dict;
    bool print_;
};

#endif
