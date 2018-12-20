
## 1. Introduction
This module contains workloads to evaluate the system performance of use cases related to Image Classification and Object detection using OpenVINO.
It has workloads “resnet50_v1” and "ssd-mobilenet" and can run Single and Multi-Batch size scenarios.

## 2. System Requirements

* Operating System: 
	Ubuntu 16.04 LTS
* CPU:
	  6th to 8th generation Intel Core and Intel Xeon processors
    Intel® Pentium® processor N4200/5, N3350/5, N3450/5 with Intel® HD Graphics
* GPU:
	  6th to 8th generation Intel Core processor with Iris® Pro graphics and Intel HD Graphics
    6th to 8th generation Intel Xeon processor with Iris Pro graphics and Intel HD Graphics (excluding the e5 product family, which does not have graphics)
* VPU:
    Intel Movidius Neural Compute Stick  ( Multi-Batch is not supported at this time )

## 3. Run Benchmark

##### Steps to configure the machine
1. Checkout the AIXPRT Github repository or download as zip to the /home/[user]/AIXPRT directory. If you download this module as zip , please make sure that the root folder name is /home/[user]/AIXPRT.

2. Install dependencies:
   * Install OpenVINO full package version , following the instructions in
  [INTEL OPENVINO](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux)
   * Install pillow
      sudo apt-get install python-pip python-numpy
      pip install Pillow


3. Build the workloads
   * Give executable permissions to compile_AIXPRT_sources.sh file which is at ~AIXPRT/install .
   * Make sure you are connected to an active internet connection.

	  cd AIXPRT/install
    sudo ./compile_AIXPRT_sources.sh

   Above steps will build the workloads with installed openVINO.

##### Steps to run benchmark
 1. Navigate to directory:
    cd AIXPRT/Harness

 2. Run the benchmark:
    python3 index.py

##### Results

Benchmark runs and finished pointing to the results location.
All the results are located at AIXPRT/Results/ after the benchmark run.


## 4. Known issues
 -  
