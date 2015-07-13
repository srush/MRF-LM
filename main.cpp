#include <omp.h>
#include <iostream>
#include <string>

#include "cmdline.h"
#include "LM.h"
#include "Tag.h"
#include "Inference.h"
#include "Train.h"
#include "TagTest.h"
#include "core.h"

using namespace std;

int main(int argc, char **argv) {
    srand(1);

    opts.add<string>("train", '\0',
                     "Training moments file.", true, "");

    opts.add<string>("valid", '\0',
                     "Validation moments file.", true, "");
    opts.add<string>("output", 'o', "Output file to write model to.", true, "");
    opts.add<string>("model", 'm',
                     "Model to use, one of LM (LM low-rank parameters), LMFull (LM full-rank parameters), Tag (POS tagger).", false, "LM");

    opts.add<int>("dims", 'D', "Size of embedding for low-rank MRF.",
                  false, 100);

    opts.add<int>("cores", 'c',
                  "Number of cores to use for OpenMP.", false, 20);
    opts.add<double>("dual-rate", 'd',
                     "Dual decomposition subgradient rate (\\alpha_1).",
                     false, 100);
    opts.add<int>("dual-iter", '\0',
                  "Dual decomposition subgradient epochs to run.", false, 500);
    opts.add<double>("mult-rate", '\0',
                     "Dual decomposition subgradient decay rate.", false, 0.5);
    opts.add<bool>("keep-deltas", '\0',
                   "Keep dual delta values to hot-start between training epochs.", false, false);
    opts.add<string>("tag-features", '\0',
                     "Features for the tagging model.", false, "");
    opts.add<string>("valid-tag", '\0',
                     "", false, "");


    opts.parse_check(argc, argv);

    omp_set_num_threads(opts.get<int>("cores"));

    ReadMoments(opts.get<string>("train"), &train_moments);
    Model *model;
    string model_type = opts.get<string>("model");
    if (model_type == "Tag") {
        model = new TagFeatures();
        ((TagFeatures*)model)->ReadFeatures(
            opts.get<string>("tag-features"));
    } else if (model_type == "TagFull") {
        model = new Tag();
    } else if (model_type == "LM") {
        model = new LMLowRank(opts.get<int>("dims"));
    } else {
        model = new LM();
    }
    model->InitFromMoments(train_moments);


    Inference inference(model);

    // If a validation set is given, read it.
    Test *valid_test = NULL;

    // Tagging uses tag-test accuracy as validation.
    if (opts.exist("valid-tag")) {
        valid_test = new TagTest(opts.get<string>("valid-tag"), false);
    }

    if (opts.exist("valid")) {
        ReadMoments(opts.get<string>("valid"), &valid_moments);

        printf("Read validate moments\n");
    }

    // Run training.
    Train train(model, &inference, valid_test);
    train.LBFGS();
}
