#include <omp.h>
#include <iostream>
#include <string>

#include "cmdline.h"
#include "LM.h"
#include "Tag.h"
#include "Inference.h"
#include "Train.h"
#include "core.h"

using namespace std;

int main(int argc, char **argv) {
    srand(1);

    opts.add<string>("train", '\0',
                     "Training moments file.", true, "");

    opts.add<string>("valid", '\0',
                     "Validation moments file.", true, "");
    opts.add<string>("output", 'o', "Output file to write model to.", true, "");
    opts.add<string>("model", 'm', "Model to use, one of (LM, LMFull, Tag).");
    opts.add<int>("dims", 'D', "Size of embedding for low-rank MRF.",
                  false, 100);

    opts.add<int>("cores", 'c',
                  "Number of cores to use for OpenMP.", false, 20);
    opts.add<double>("dual-rate", 'd',
                     "Dual decomposition subgradient rate (\\alpha_1).",
                     false, 20);
    opts.add<int>("dual-iter", '\0',
                  "Dual decomposition subgradient epochs to run.", false, 500);
    opts.add<double>("mult-rate", '\0',
                     "Dual decomposition subgradient decay rate.", false, 0.5);
    opts.add("keep-deltas", '\0',
             "Keep dual values between subgradient epochs.");

    opts.parse_check(argc, argv);

    omp_set_num_threads(opts.get<int>("cores"));

    ReadMoments(opts.get<string>("train"), &train_moments);
    Model *model;
    string model_type = opts.get<string>("model");
    if (model_type == "Tag") {
        int F = 2;
        model = new TagFeatures(train_moments.L, train_moments.sizes[0],
                                train_moments.L, train_moments.sizes[train_moments.L-1], F);
    } else {
        if (model_type == "LM") {
            model = new LMLowRank(train_moments.L-1, train_moments.sizes[0],
                                  opts.get<int>("dims"));
        } else {
            model = new LM(train_moments.L-1, train_moments.sizes[0]);
        }
    }

    Inference inference(model);

    // If a validation set is given, read it.
    Test *train_test = NULL;
    Test *valid_test = NULL;

    // if (opts.exist("train")) {
    //     train_test = new Test(opts.get<string>("train"));
    //     printf("Read training set, %d sentences\n", nsent_train);
    // }
    // if (opts.exist("valid")) {
    //     valid_test = new Test(opts.get<string>("valid"));
    //     printf("Read validation set, %d sentences\n", nsent_valid);
    // }
    if (opts.exist("valid")) {
        ReadMoments(opts.get<string>("val-moments"), &valid_moments);

        printf("Read validate moments\n");
    }

    // Run training.
    Train train(model, &inference, NULL);
    train.LBFGS();

    // WriteModel(opts.get<string>("output"));
}
