
## 1. Introduction
This module contains workloads to evaluate the system performance of use cases related to Image Classification and Object Detection  using Tensorflow with TensorRT optimizations.
It has workloads “resnet50_v1”, "ssd-mobilenet-v1" and can run Single and Multi-Batch size scenarios.

## 2. System Requirements
For Nvidia Discrete Graphics Cards
* Operating System:
	Ubuntu 16.04 LTS
* GPU:
	  [CUDA enabled NVIDIA GPUs](https://developer.nvidia.com/cuda-gpus)

For NVidia Tegra Xavier
* JetPACK 4.1.1

## 3. Run Benchmark

#### Steps to configure the machine

1. Clone the AIXPRT Github repository 

2. Install dependencies:

   a. If using NVidia Discrete GFX
   
   * Install [CUDA 10](https://developer.nvidia.com/cuda-downloads)
   * Restart the system after installing CUDA 10
   * NOTE : Requires NVIDIA Driver release 410.xx.However,these drivers are installed during CUDA installation and no 		    seperate driver installation is requiered. 
   * Install docker and nvidia-docker <br />
	 => Install docker: <br />
		https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-from-a-package <br />
		
		```shell
		lsb_release -a # show the Ubuntu version and name.(amd64 is the release for amd and intel 64)
		```
		
		You may also need to install the cli and container packages  <br />
		
		```shell
		sudo docker run hello-world # test the installation of the docker image
		sudo usermod -a -G docker $USER # To be able to run as a regular user.If the problem persists,reboot.
	  	```
		
	 => Install nvidia Docker: <br />
		https://github.com/NVIDIA/nvidia-docker<br />
	
	 => Pull and run TensorRT Docker Container v19.01 <br />
		```
		docker pull nvcr.io/nvidia/tensorflow:19.01-py3
		```
   b. If using Tegra Xavier <br />
   	Flash JetPack v4.1.1 https://developer.nvidia.com/embedded/jetpack

#### Steps to run benchmark
 1. Choose the target machine and run the commands <br />
	 a. If using NVidia Discrete GFX 
	    Run the docker image 
		`nvidia-docker run -v <Path_to_AIXPRT_directory>:/workspace/AIXPRT --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm nvcr.io/nvidia/tensorflow:19.01-py3`

	 b. If using NVidia Tegra Xavier <br />
	    Install pre-reqs 
	```shell
		sudo apt-get install libhdf5-serial-dev hdf5-tools 
		sudo apt-get install python3-pip
		pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v411 tensorflow-gpu==1.12.0-rc2+nv18.11
		sudo apt-get install python3-matplotlib
	```

 2.  Add the models directory to PYTHONPATH to install tensorflow/models and Run the TF Slim setup. 
 Run script setup.sh inside /workspace/AIXPRT. If the script fails run the following commands manually
 
 ```shell
        git clone https://github.com/tensorflow/models.git 
        cd models 
        export PYTHONPATH="$PYTHONPATH:$PWD" 
        cd research
        export PYTHONPATH="$PYTHONPATH:$PWD" 
        wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
        unzip protobuf.zip
        ./bin/protoc object_detection/protos/*.proto --python_out=.
        cd slim
        python setup.py install
        pip install requests pillow
```

 3. Navigate to directory:
 	`cd /workspace/AIXPRT/Harness/`

 4. Run the benchmark:<br />
	`python3 index.py`

#### Results

Benchmark runs and finished pointing to the results location.
All the results are located at AIXPRT/Results/ after the benchmark run.


## 4. Known issues
 -  In int8 mode the benchmark runs longer due to the conversion of model from fp32 to int8. 
 -  SSD-MobileNet-v1 Int8 mode is not currently supported
