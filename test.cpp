#include <omp.h>
#include <iostream>
#include <string>

#include "cmdline.h"
#include "LM.h"
#include "Tag.h"
#include "LMTest.h"
#include "TagTest.h"
#include "core.h"

using namespace std;

int main(int argc, char **argv) {
    srand(1);

    opts.add<string>("model-name", 'o', "Output file to write model to.", true, "");
    opts.add<string>("model", 'm',
                     "Model to use, one of LM (LM low-rank parameters), LMFull (LM full-rank parameters), Tag (POS tagger).", false, "LM");

    opts.add<string>("valid", '\0',
                     "Validation moments file.", false, "");
    opts.add<string>("train", '\0',
                     "Training moments file.", false, "");
    opts.add<string>("embeddings", '\0', "File to write word-embeddings to.",
                     false, "");
    opts.add<string>("vocab", '\0', "Word vocab file.", false, "");
    opts.add<string>("lm-file", '\0', "Language model file.", false, "");
    opts.add<string>("tag-features", '\0',
                     "Features for the tagging model.", false,"");
    opts.add<string>("tag-file", '\0', "Tag test file.", false, "");
    opts.add<string>("tag-vocab", '\0', "Tag vocab file.", false, "");
    opts.add<int>("cores", 'c',
                  "Number of cores to use for OpenMP.", false, 20);


    opts.parse_check(argc, argv);
    if (opts.exist("train")) {
        ReadMoments(opts.get<string>("train"), &train_moments);
    }
    omp_set_num_threads(opts.get<int>("cores"));
    Model *model;
    string model_type = opts.get<string>("model");
    if (model_type == "Tag") {
        model = new TagFeatures();
        ((TagFeatures*)model)->ReadFeatures(
            opts.get<string>("tag-features"));

    } else if (model_type == "TagFull") {
        model = new Tag();
    } else if (model_type == "LM") {
        model = new LMLowRank();
    } else {
        model = new LM();
    }
    cout << "Reading Model " << endl;
    model->ReadModel(opts.get<string>("model-name"));

    if (opts.exist("vocab")) {
        model->set_labels(opts.get<string>("vocab"), 0);
    }
    if (opts.exist("tag-vocab")) {
        model->set_labels(opts.get<string>("tag-vocab"), 1);
    }

    cout << "Running " << endl;
    // If a validation set is given, read it.
    Test *valid_test = NULL;
    if (opts.exist("tag-file")) {
        valid_test = new TagTest(opts.get<string>("tag-file"), true);
        cout << "Score" << " " << valid_test->TestModel(*model)
             << endl;
    } else if (opts.exist("embeddings")) {
        LMLowRank *lmodel = (LMLowRank *)model;
        lmodel->WriteEmbeddings(opts.get<string>("embeddings"));
        lmodel->WriteKNN(opts.get<string>("embeddings") + ".nn", 10);
    } else if (opts.exist("lm-file")) {
        valid_test = new LMTest(opts.get<string>("lm-file"), true);
        cout << "Score" << " " << valid_test->TestModel(*model)
             << endl;

    }
    if (opts.exist("valid")) {
        ReadMoments(opts.get<string>("valid"), &valid_moments);
        cout << model->ComputeObjective(valid_moments, 0.0) << endl;
    }
}
