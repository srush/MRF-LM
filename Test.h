#ifndef TEST_H
#define TEST_H

#include "Model.h"

class Test {
public:
    virtual double TestModel(Model &model) = 0;
};

#endif
