NVCC        = nvcc

NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O0 -g -G
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = othello
OBJ	        = othello_cu.o

default: $(EXE)

othello_cu.o: othello.cu othello_kernel.cu othello_shared.cu othello_shared.h alpha_beta.cu alpha_beta.h
	$(NVCC) -c -o $@ othello.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
