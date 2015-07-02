
CFLAGS=-I. -I./liblbfgs-1.10/include/

DEPS = Inference.h Model.h LM.h Tag.h Test.h Train.h core.h
OBJ  = Inference.o Model.o LM.o Tag.o Test.o Train.o core.o main.o ./liblbfgs-1.10/lib/lbfgs.o
TAG_DEPS = tagger.h tagger_lifter.h tagger_params.h
TAG_OBJ  = tagger.o tagger_lifter.o tagger_params.o tagger_main.o core.o ./liblbfgs-1.10/lib/lbfgs.o
CC=g++ -O2 -Wall -fopenmp -O3 -std=c++0x

%.o: %.cpp $(DEPS) $(TAG_DEPS)
	-O2 -Wall -fopenmp -O3 -std=c++0x -c -o $@ $< $(CFLAGS)

mrflm: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY : clean
clean :
	-rm $(OBJ) $(TAG_OBJ)
