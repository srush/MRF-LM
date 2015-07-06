CC=g++ -std=c++0x -Wall -fopenmp -O3
CFLAGS=-I. -I./liblbfgs-1.10/include/

DEPS = Inference.h Model.h LM.h Tag.h  core.h Train.h Test.h TagTest.h
OBJ  = Inference.o Model.o LM.o Tag.o  Train.o  core.o main.o TagTest.o ./liblbfgs-1.10/lib/lbfgs.o

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

mrflm: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY : clean
clean :
	-rm $(OBJ) $(TAG_OBJ)
