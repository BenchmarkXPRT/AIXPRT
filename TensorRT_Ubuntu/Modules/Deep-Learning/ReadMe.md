## 1. Introduction
This module contains workloads to evaluate the system performance of use cases related to Image Classification and Object Detection  using TensorRT optimizations.
It has workloads “resnet50_v1”, "ssd-mobilenet-v1" and can run Single and Multi-Batch size and Muti instance scenarios.

## 2. System Requirements

For Nvidia Discrete Graphics Cards
* Operating System:
	Ubuntu 16.04 / 18.04 LTS . (We recommend 18.04.3)
* GPU:
	  [CUDA enabled NVIDIA GPUs](https://developer.nvidia.com/cuda-gpus)

For NVidia Tegra Xavier
* JetPACK 4.2

## 3. Run Benchmark

#### Steps to configure the machine
   
1. Install dependencies:
    [Note] The instructions below assume the user is not connected to Internet via Proxy.
   a. If using NVidia Discrete GFX
   
   * Install CUDA v10.1 [CUDA](https://developer.nvidia.com/cuda-downloads)
    [Note] Link above may be updated to newer versions. Please use the specific version mentioned above.
   * Restart the system after installing CUDA 
   * Install OpenCV <br />
	 => Download OpenCV: <br />
		https://github.com/opencv/opencv/archive/4.1.0.zip <br />
		
	 => Install OpenCV: <br />
	 ```shell
	    sudo apt-get install build-essential cmake
        unzip opencv-4.1.0.zip
        cd opencv<version>
        mkdir build
        cd build
        cmake ..
        make -j7
        sudo make install
     ``` 
     
   * Install packages <br />
	 `sudo apt-get install python3-numpy python3-opencv python3-pil python-dev` <br />
		
   * Install TensorRT files <br />
     Goto https://developer.download.nvidia.com/compute/machine-learning/repos/ <br />
     Select Ubuntu version (16.04/18.04) and "x86_64" in the next page <br />
     
     Download the following packages: <br />
     libcudnn7_{version}+cuda10.1_amd64.deb (Tested with libcudnn7_7.6.0.64-1+cuda10.1_amd64.deb) <br />
     libcudnn7-dev_{version}+cuda10.1_amd64.deb (Tested with libcudnn7-dev_7.6.0.64-1+cuda10.1_amd64.deb) <br />
     libnccl2_{version}+cuda10.1_amd64.deb (Tested with libnccl2_2.4.7-1+cuda10.1_amd64.deb) <br />
     libnccl-dev_{version}+cuda10.1_amd64.deb (Tested with libnccl-dev_2.4.7-1+cuda10.1_amd64.deb) <br />
     libnvinfer5_{version}+cuda10.1_amd64.deb (Tested with libnvinfer5_5.1.5-1+cuda10.1_amd64.deb) <br />
     libnvinfer-dev_{version}+cuda10.1_amd64.deb (Tested with libnvinfer-dev_5.1.5-1+cuda10.1_amd64.deb) <br />
     
     Install the Packages:
     ```shell
     sudo dpkg -i <deb-file-name>
     ```
     During installation of above package if "dpkg: error processing package" is encountered , please run `sudo apt-get install -f` to resolve it.
     
   b. If using Tegra Xavier <br />
   	Flash JetPack v4.2 https://developer.nvidia.com/embedded/jetpack
    [Note] Link above may be updated to newer versions. Please use the specific version mentioned above.
    
#### Steps to run benchmark
 1. Choose the target machine and run the commands <br />
	 Compile the sources 
		`Goto <AIXPRT_Directory>/Modules/Deep-Learning/workloads/commonsources/bin/src/<workload> and run "make clean" and "make" to create binaries`<br />
         NOTE : <workload> in the above path is each folder in the src directory.


 2. Navigate to directory:
 	`cd /workspace/AIXPRT/Harness/`

 3. Run the benchmark:<br />
	`python3 index.py`

## Results

When the test is complete, the benchmark saves the results to AIXPRT/Resultsin JSONformat, and also generates CSV files with the name {ConfigName}_RESULTS_SUMMARY.csv
To submit results, please follow the instructions in AIXPRT/ResultSubmission.md or at https://www.principledtechnologies.com/benchmarkxprt/aixprt/2019/submit-results.php.


##### Sample results summary file <br/>

Each results summary file has three sections: SYSTEM INFORMATION, RESULTSSUMMARY and DETAILED RESULTS.<br/>
 1. SYSTEM INFORMATION <br/>
    This section provides basic information about the system under test. <br/>
    ![alt text](https://github.com/BenchmarkXPRT/AIXPRT/tree/master/Tensorflow/AIXPRT/Harness/assets/tensorflow_systemInfo.png)

 2. RESULTSSUMMARY <br/>
    AIXPRT measures inference latency and throughput for image recognition (ResNet-50) and object detection (SSD-MobileNet) tasks. Batching tasks allows AI applications to achieve 
higher levels of throughput, but higher throughput may come at the expense of increased latency per task. In real-time or near real-time use cases like performing image recognition 
on individual photos being captured by a camera, lowerlatency is important to enable better user experience. In other cases, like performing image recognition on a large library of 
photos, higher throughput through batching images or concurrent instances may allow faster completion of the overall workload. The achieve optimal latency and/or throughput levels, 
AI applications often tune batch sizes and/or concurrent instances according to a system’s hardware capabilities, such as the number of available processor cores and threads.
To represent a spectrum of common tunings, AIXPRT tests AI tasks in different batch sizes (1 –32 is the default in this package) that are relevant to the target test system. 
AIXPRT then reports the maximum throughput and minimum latency for image recognition (ResNet-50) and object detection (SSD-MobileNet v1) usages.<br/>
The AIXPRT results summary (example below) makes iteasier to quickly identify relevant comparisons between systems. <br/>

 ![alt text](https://github.com/BenchmarkXPRT/AIXPRT/tree/master/Tensorflow/AIXPRT/Harness/assets/results_summary.png)


 3. DETAILED RESULTS <br/>
   This section shows the throughput and latency results for each AItask configuration tested by the benchmark. 
AIXPRT runs each AI task (e.g. ResNet-50, Batch1, on CPU) multiple times andreports the average inference throughput and corresponding latency percentiles.

![alt text](https://github.com/BenchmarkXPRT/AIXPRT/tree/master/Tensorflow/AIXPRT/Harness/assets/detailed_results.png)