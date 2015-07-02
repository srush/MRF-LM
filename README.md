# MRF-LM: Fast Markov Random Field Language Models

[Documentation in progress]

An implementation of a fast variational inference algorithm for Markov
Random Field language models as well as other Markov sequence models.

This algorithm implemented in this project is described in the paper

    A Fast Variational Approach for Learning Markov Random Field Language Models
    Yacine Jernite, Alexander M. Rush, and David Sontag.
    Proceedings of ICML 2015.
    [http://people.seas.harvard.edu/~srush/icml15.pdf]

## Building

To build the main C++ library, run

> bash build.sh

This will build liblbfgs (needed for optimization) as well as the main
executables. The package requires a C++ compiler with support for
OpenMP.

## Training

First construct a moments file from the data set of interest. We include the
standard Penn Treebank data set with the distribution in data/ .

> python Moments.py --K 2 --train lm_data/ptb.train.txt --valid lm_data/ptb.valid.txt --output lm_data/


Next run the main `mrflm` executable.

> ./mrflm --train lm_data/ptb.train_moments.txt --valid lm_data/ptb.valid_moments.txt --output model.out -D 100

This will train a language model and write the parameters out to `model.out`. (These parameter settings
will correspond to Figure 6 in the paper.)


## MRF-LM

The main MRF executable has several options for controlling the
model used, training procedure, and the parameters of dual decomposition.

    usage: ./mrflm --train=string --valid=string --output=string --model=string [options] ...
    options:
    --train          Training moments file. (string)
    --valid          Validation moments file. (string)
    -o, --output     Output file to write model to. (string)
    -m, --model      Model to use, one of (LM, LMFull, Tag). (string)
    -D, --dims       Size of embedding for low-rank MRF. (int [=100])
    -c, --cores      Number of cores to use for OpenMP. (int [=20])
    -d, --dual-rate  Dual decomposition subgradient rate (\alpha_1). (double [=20])
    --dual-iter      Dual decomposition subgradient epochs to run. (int [=500])
    --mult-rate      Dual decomposition subgradient decay rate. (double [=0.5])
    --keep-deltas    Keep dual values between subgradient epochs.
    -?, --help       print this message

## Advanced Usage

### Moments File Format

The input to the main implementation is a file containing the moments of the lifted
MRF. The moments file assumes the lifted graph is star-shaped with the central variable
as index one. The format of the file is

    {n = # of samples}
    {l = # of variables}
    {# of states of variable 1} {# of states of variable 2} ...(l columns)
    {m1 = # of 2->1 pairs}
    {State in 2} {State in 1} {Counts}
    {State in 2} {State in 1} {Counts}
    ...(m1 rows)
    {m2 = # of 3->1 pairs}
    {State in 3} {State in 1} {Counts}
    {State in 3} {State in 1} {Counts}
    ...(m2 rows)

This same format is used for language modelling and tagging, and can be used for other models.

### Code Format

The code is broken into three main classes

#### Train.h

Generic L-BFGS training. Implements most of Algorithm 2.

#### Inference.h

Lifted inference on a star-shaped MRF. Implements Algorithm 1.

#### Model.h

Pairwise MRF parameters. Implements likelihood computation, gradient updates, and lifted structure.

The `Model.h` class is a full-rank MRF by default, but can be easily
extended to allow for alternative parameterization. See `LM.h` for the low-rank
language model with back-prop (Model 2 in the paper), and `Tag.h` for a feature
factorized part-of-speech tagging model.
