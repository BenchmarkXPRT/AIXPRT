.SUFFIXES: .cpp .cu

CUDA_VER?=cuda-$(shell dpkg-query -f '$${version}\n' -W 'cuda-cudart-[0-9]*' | cut -d . -f 1,2 | sort -n | tail -n 1)

all: ../../uff_ssd

../../uff_ssd: 
	g++ sampleUffSSD.cpp -o ../../uff_ssd -fPIC -Wall -O2 -std=c++11 -I/usr/local/$(CUDA_VER)/include -I/usr/include/python2.7 -I/usr/local/$(CUDA_VER)/targets/x86_64-linux-gnu/include -L/usr/local/$(CUDA_VER)/lib64 -L/usr/local/$(CUDA_VER)/targets/x86_64-linux-gnu/lib -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart  -lrt -ldl -lpthread


.PHONY: clean

clean:
	rm -f *.o ../../uff_ssd
