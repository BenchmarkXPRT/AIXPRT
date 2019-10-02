.SUFFIXES: .cpp .cu

CUDA_VER?=cuda-$(shell dpkg-query -f '$${version}\n' -W 'cuda-cudart-[0-9]*' | cut -d . -f 1,2 | sort -n | tail -n 1)

all: ../../multi_stream_ssd

../../multi_stream_ssd:
	g++ main.cpp -o ../../multi_stream_ssd -g -fPIC -O2 -fpermissive -std=c++11 -I/usr/local/include/opencv4 -I/usr/local/$(CUDA_VER)/include -I/usr/include/python2.7 -I/usr/local/$(CUDA_VER)/targets/x86_64-linux-gnu/include -L/usr/local/$(CUDA_VER)/lib64 -L/usr/local/$(CUDA_VER)/targets/aarch64-linux/lib -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart  -lrt -ldl -lpthread -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs 


.PHONY: clean

clean:
	rm -f *.o ../../multi_stream_ssd
