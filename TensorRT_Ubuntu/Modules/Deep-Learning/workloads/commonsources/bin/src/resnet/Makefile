.SUFFIXES: .cpp .cu

CUDA_VER?=cuda-$(shell dpkg-query -f '$${version}\n' -W 'cuda-cudart-[0-9]*' | cut -d . -f 1,2 | sort -n | tail -n 1)
AARCH=x86_64-linux
all: ../../image_classification

../../image_classification: 
	g++ trtexec.cpp -o ../../image_classification -fPIC -O2 -std=c++11 -I/usr/local/include/opencv4 -I/usr/local/$(CUDA_VER)/include -I/usr/include/python2.7 -I/usr/local/$(CUDA_VER)/targets/$(AARCH)/include -L/usr/local/$(CUDA_VER)/lib64 -L/usr/local/$(CUDA_VER)/targets/$(AARCH)/lib -lnvinfer -lnvparsers -lnvinfer_plugin -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart  -lrt -ldl -lpthread

.PHONY: clean

clean:
	rm -f *.o ../../image_classification

