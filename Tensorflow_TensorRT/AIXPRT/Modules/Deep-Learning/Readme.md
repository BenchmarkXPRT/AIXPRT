
## 1. Introduction
This module contains workloads to evaluate the system performance of use cases related to Image Classification and Object Detection  using Tensorflow with TensorRT optimizations.
It has workloads “resnet50_v1”, "ssd-mobilenet-v1" and can run Single and Multi-Batch size scenarios.

## 2. System Requirements
For Nvidia Discrete Graphics Cards
* Operating System:
	Ubuntu 16.04 LTS
* GPU:
	  [CUDA enabled Nvidia GPUs](https://developer.nvidia.com/cuda-gpus)

For Nvidia Tegra Xavier
* JetPACK 4.1.1

## 3. Run Benchmark

#### Steps to configure the machine

1. Clone the AIXPRT Github repository. These instructions are found in the repository README.

2. Install dependencies:

   a. If using Nvidia Discrete GFX

		   * Install [CUDA 10](https://developer.nvidia.com/cuda-downloads)
		   * Restart the system after installing CUDA 10
		   * NOTE : Requires Nvidia Driver release 410.xx.However,these drivers are installed during CUDA installation and no separate driver installation is required.
		   * Install docker and nvidia-docker

				 => Install docker from this location: https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-from-a-package
				 Installation notes:
				 		* You will probably also need to install the cli and container packages located in the same docker package folder
				 		* To see your system configuration and identify the following installation packages, run
						```
								lsb_release -a
			 			```
						* To test the installation of the docker image, run
						```
								sudo docker run hello-world
						```
						* If you get a "permission denied" response after installation, run the following command and reboot if the problem persists
						```
							sudo usermod -a -G docker $USER
						```

				 => Install Nvidia Docker from this location: https://github.com/nvidia/nvidia-docker

				 => Pull and run TensorRT Docker Container v18.12
					```
					docker pull nvcr.io/nvidia/tensorflow:18.12-py3
					```
	   b. If using Tegra Xavier <br />
	   			Flash JetPack v4.1.1 https://developer.nvidia.com/embedded/jetpack

#### Steps to run benchmark
 1. Choose the target machine and run the commands
	 a. If using Nvidia Discrete GFX
	    Run the docker image
		`nvidia-docker run -v <Path_to_AIXPRT_directory>:/workspace/AIXPRT --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm nvcr.io/nvidia/tensorflow:18.12-py3`

	 b. If using Nvidia Tegra Xavier
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

 4. Run the benchmark:
	`python3 index.py`

#### Results

Benchmark runs and finished pointing to the results location.
All the results are located at AIXPRT/Results/ after the benchmark run.


## 4. Known issues
 -  In int8 mode the benchmark runs longer due to the conversion of model from fp32 to int8.
 -  SSD-MobileNet-v1 Int8 mode is not currently supported
