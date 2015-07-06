# MRF-LM
## Fast Markov Random Field Language Models

[Documentation in progress]

An implementation of a fast variational inference algorithm for Markov
Random Field language models as well as other Markov sequence models.

This algorithm implemented in this project is described in the paper

    A Fast Variational Approach for Learning Markov Random Field Language Models
    Yacine Jernite, Alexander M. Rush, and David Sontag.
    Proceedings of ICML 2015.

Available [here](http://people.seas.harvard.edu/~srush/icml15.pdf).

## Building

To build the main C++ library, run

    bash build.sh

This will build liblbfgs (needed for optimization) as well as the main
executables. The package requires a C++ compiler with support for
OpenMP.

## Training a Language Model

The training procedure requires two steps.

First you construct a moments file from the text data of interest. We include the
standard Penn Treebank language modelling data set as an example. This data is located under `lm_data/` . To extract moments from this file run

    python Moments.py --K 2 --train lm_data/ptb.train.txt --valid lm_data/ptb.valid.txt


Next run the main `mrflm` executable providing the training moments, validation moments, and an output file for the model.

    ./mrflm --train=lm_data/ptb.train.txt_moments_K2.dat --valid=lm_data/ptb.valid.txt_moments_K2.dat --output=model.out

This command will train a language model, compute validation
log-likelihood, and write the parameters out to `model.out`. (These
parameter settings will correspond to Figure 6 in the paper.)

## MRF-LM

The main MRF executable has several options for controlling the
model used, training procedure, and the parameters of dual decomposition.

    usage: ./mrflm --train=string --valid=string --output=string [options] ...
    options:
      --train           Training moments file. (string)
      --valid           Validation moments file. (string)
      -o, --output      Output file to write model to. (string)
      -m, --model       Model to use, one of LM (LM low-rank parameters), LMFull (LM full-rank parameters). (string [=LM])
      -D, --dims        Size of embedding for low-rank MRF. (int [=100])
      -c, --cores       Number of cores to use for OpenMP. (int [=20])
      -d, --dual-rate   Dual decomposition subgradient rate (\alpha_1). (double [=20])
      --dual-iter       Dual decomposition subgradient epochs to run. (int [=500])
      --mult-rate       Dual decomposition subgradient decay rate. (double [=0.5])
      --keep-deltas     Keep dual delta values to hot-start between training epochs. (bool [=0])
      -?, --help        print this message

## Advanced Usage

### Moments File Format

The input to the main implementation is a file containing the moments of the lifted
MRF. The moments file assumes the lifted graph is star-shaped with the central variable
as index one. The format of the file is

    {N = # of samples}
    {L = # of variables}
    {# of states of variable 1} {# of states of variable 2} ...(l columns)
    {M1 = # of 2->1 pairs}
    {State in 2} {State in 1} {Counts}
    {State in 2} {State in 1} {Counts}
    ...(M1 rows)
    {M2 = # of 3->1 pairs}
    {State in 3} {State in 1} {Counts}
    {State in 3} {State in 1} {Counts}
    ...(M2 rows)

This file format is used for both language modelling and tagging.

### LM Moments File

Consider a language modelling setup. 

For example, let's say we were building a language model 
with the training corpus:

    the cat chased the mouse
   
If our model has context K = 2, then we transform the corpus to:

    <S> <S> the cat chased the mouse

After the transformation the number of samples is N = 7, the number of
variables is L = K+1 = 3, the vocabulary size/number of states is V=5, and
the dictionary is:

    <S> 1
    the 2
    cat 3
    chased 4
    mouse 5

The corresponding moments file would then look like:

    7
    3
    5 5 5
    7
    5 1 1
    1 1 1
    1 2 1
    2 3 1
    3 4 1
    4 2 1
    2 5 1
    7
    2 1 1
    5 1 1
    1 2 1
    1 3 1
    2 4 1
    3 2 1
    4 5 1
    
### Tagging Moments File

Now consider a tagging setup. Let's say we were building a tagging
model with the training corpus:

    the/D cat/N chased/V the/D mouse/N
   
If our model has context K=1, M=3 (roughly corresponding to Figure~7 in the paper) then we transform the corpus to:

    <S>/<T> the/D cat/N chased/V the/D mouse/N

After the transformation the number of samples is N = 6, the number of
lifted variables is L = M + K+1 = 5, the number of tag states is T=4 and V=5 as above,
and the tag dictionary is:

    <T> 1
    D 2
    N 3
    V 4

The corresponding moments file would then look like:

    6
    5
    4 5 5 5
    6
    3 1 1
    1 2 1
    2 3 1
    3 4 1
    4 2 1
    2 3 1
    ...



### Code Structure

The code is broken into three main classes

* `Train.h`; Generic L-BFGS training. Implements most of Algorithm 2.

* `Inference.h`; Lifted inference on a star-shaped MRF. Implements Algorithm 1.

* `Model.h`; Pairwise MRF parameters. Implements likelihood computation, gradient updates, and lifted structure.

The `Model.h` class is a full-rank MRF by default, but can be easily
extended to allow for alternative parameterization. See `LM.h` for the low-rank
language model with back-prop (Model 2 in the paper), and `Tag.h` for a feature
factorized part-of-speech tagging model.
