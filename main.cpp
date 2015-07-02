#include <omp.h>
#include <iostream>
#include <string>

#include "cmdline.h"
#include "LM.h"
#include "Tag.h"
#include "Inference.h"
#include "TestModel.h"
#include "Train.h"
#include "core.h"

using namespace std;

int main(int argc, char **argv) {
    srand(15);

    opts.add<int>("dims", 'D', "Size of embedding.", false, 100);
    opts.add<double>("dual-rate", 'd',
                     "Dual decomposition subgradient rate.", false, 20);
    opts.add<int>("dual-iter", '\0',
                  "Dual decomposition subgradient epochs to run.", false, 500);
    opts.add<double>("mult-rate", '\0',
                     "Decay rate for subgradient method \alpha.", false, 0.5);

    // opts.add<double>("rate-decay", 'r', ".", false, 0);
    opts.add<string>("output", 'o', "Output file to write model to.", true, "");
    opts.add<int>("cores", 'c',
                  "Number of cores to use for OpenMP.", false, 20);
    opts.add<string>("val-moments", '\0',
                     "Validation moments file.", false, "");
    opts.add<string>("moments", '\0',
                     "Training moments file.", false, "");

    // opts.add<string>("train", '\0', "Train text file.", false, "");
    // opts.add<string>("valid", '\0', "Validation text file.", false, "");

    opts.add("keep-deltas", '\0', "Keep dual values between epochs.");
    opts.parse_check(argc, argv);


    printf("Cores %d\n", opts.get<int>("cores"));
    omp_set_num_threads(opts.get<int>("cores"));

    // Read the moments and initialize the parameters.
    ReadMoments(opts.get<string>("moments"), &train_moments);

    printf("Read training moments\n");
    Model *model;
    if (true) {
        int F = 2;
        model = new TagFeatures(train_moments.L, train_moments.sizes[0],
                                train_moments.L, train_moments.sizes[train_moments.L-1], F);
    } else {
        if (true) {
            model = new LMLowRank(train_moments.L-1, train_moments.sizes[0],
                                  opts.get<int>("dims"));
        } else {
            model = new LM(train_moments.L-1, train_moments.sizes[0]);
        }
    }

    // {
    //     double total = 0;
    //     double log_likelihood = 0.0;
    //     for (int i = 0; i < train_moments.V; i++) {
    //         total += train_moments.counts[i];
    //     }
    //     int count = 0;
    //     for (int i = 0; i < train_moments.V; i++) {
    //         count += train_moments.counts[i];
    //         log_likelihood += train_moments.counts[i]
    //                 * log(train_moments.counts[i] / total);
    //     }

    //     cout << "Unigram:" << " " << log_likelihood / (float) count << endl;
    // }

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
    if (opts.exist("val-moments")) {
        ReadMoments(opts.get<string>("val-moments"), &valid_moments);

        printf("Read validate moments\n");
    }

    // Run training.
    Train train(model, &inference, NULL);
    train.LBFGS();

    // WriteModel(opts.get<string>("output"));
}


    // warm_start = false;
    // debug_mode = 0;
    // check_rate = 10;
    // int num_epochs = 15;


        // par.keep_deltas = opts.exist("keep-deltas");
        // par.dual_rate=1e2, par.dual_iter=1000, par.grad_rate=2e3;
    // par.grad_iter=500, par.rate_decay=0.3, par.mult_rate=0.75;
    // par.SGD=false, par.local=false;
    // par.lbfgs = true, par.keep_deltas=true;
    // par.check_training=false, par.validate_moments=false, par.validate=false;

    // par.batch_size = 1;

    // if ((i = ArgPos((char *)"-D", argc, argv)) > 0) par.D = atoi(argv[i + 1]);
    // if ((i = ArgPos((char *)"-dual-rate", argc, argv)) > 0) par.dual_rate = atof(argv[i + 1]);
    // if ((i = ArgPos((char *)"-dual-iter", argc, argv)) > 0) par.dual_iter = atoi(argv[i + 1]);
    // if ((i = ArgPos((char *)"-grad-rate", argc, argv)) > 0) par.grad_rate = atof(argv[i + 1]);
    // if ((i = ArgPos((char *)"-grad-iter", argc, argv)) > 0) par.grad_iter = atoi(argv[i + 1]);
    // if ((i = ArgPos((char *)"-mult-rate", argc, argv)) > 0) par.mult_rate = atof(argv[i + 1]);
    // if ((i = ArgPos((char *)"-model", argc, argv)) > 0) par.model = atoi(argv[i + 1]);
    // if ((i = ArgPos((char *)"-moments", argc, argv)) > 0) strcpy(moments_file, argv[i + 1]);
    // if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    // if ((i = ArgPos((char *)"-cores", argc, argv)) > 0) par.cores = atoi(argv[i + 1]);
    // if ((i = ArgPos((char *)"-lbfgs", argc, argv)) > 0) par.lbfgs = atoi(argv[i + 1]);
    // if ((i = ArgPos((char *)"-sgd", argc, argv)) > 0) par.SGD = atoi(argv[i + 1]);
    // if ((i = ArgPos((char *)"-local", argc, argv)) > 0) par.local = atoi(argv[i + 1]);
    // if ((i = ArgPos((char *)"-batch", argc, argv)) > 0) par.batch_size = atoi(argv[i + 1]);
    // if ((i = ArgPos((char *)"-keep-deltas", argc, argv)) > 0) par.keep_deltas = atoi(argv[i + 1]);
    // if ((i = ArgPos((char *)"-decay", argc, argv)) > 0) par.rate_decay = atof(argv[i + 1]);
    // if ((i = ArgPos((char *)"-checks", argc, argv)) > 0) check_rate = atoi(argv[i + 1]);
    // if ((i = ArgPos((char *)"-pretrained", argc, argv)) > 0){
    //     strcpy(model_file, argv[i + 1]);
    //     warm_start = true;
    // }
    // if ((i = ArgPos((char *)"-valid", argc, argv)) > 0){
    //     strcpy(valid_file, argv[i + 1]);
    //     par.validate = true;
    // }
    // if ((i = ArgPos((char *)"-train", argc, argv)) > 0){
    //     strcpy(train_text, argv[i + 1]);
    //     par.check_training = true;
    // }
    // if ((i = ArgPos((char *)"-val-moments", argc, argv)) > 0){
    //     strcpy(valid_moments_file, argv[i + 1]);
    //     par.validate_moments = true;
    // }
