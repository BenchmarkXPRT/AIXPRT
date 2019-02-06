
## 1. Introduction
This module contains workloads to evaluate the system performance of use cases related to Image Classification and Object detection using OpenVINO.
It has workloads “resnet50_v1” and "ssd-mobilenet" and can run Single Batch, Multi-Batch and Multi Instance scenarios.

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
1. Clone the AIXPRT Github repository.

2. Install dependencies:
   * Install OpenVINO full package version , following the instructions in
  [INTEL OPENVINO](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux) .
  NOTE : This version is developed and tested with OpenVINO-Linux-R5

   * For GPU and VPU support ,also follow the sections "Additional installation steps for Intel® Processor Graphics (GPU)" and
   "Additional installation steps for Intel® Movidius™ Neural Compute Stick and Intel® Neural Compute Stick 2" in the guide.

   * Install pillow
      ```
      sudo apt update
      sudo apt-get install python-pip python-numpy git
      pip install Pillow

      ```

3. Build the workloads
   * Give executable permissions to compile_AIXPRT_sources.sh and install_GPU_VPU_HDDLR_dep.sh files which is at ~/AIXPRT/install .
   * Make sure you are connected to an active internet connection.
    ```
    cd ~/AIXPRT/install
    sudo ./install_GPU_VPU_HDDLR_dep.sh
    sudo ./compile_AIXPRT_sources.sh <path/to/AIXPRT/folder>
    ```
   Above steps will build the workloads with installed openVINO.

##### Steps to run benchmark
 1. Navigate to directory:

    ```
    cd ~/AIXPRT/Harness
    ```
 2. Run the benchmark:

    ```
    python3 index.py
    ```
    If the above command fails , please try running with  ```sudo ```

  3. Above step will generate a default configuration json file and runs the benchmark with it, which is CPU targer with fp32 precision on batches 1,2,4,8,16,32,64 and 128.
   If you wish to change the configuration , please edit the json file under AIXPRT/Config/. Instructions to edit the configuration json are [here](TODO: Add path to /AIXPRT/EditConfig.md).

##### Results

Benchmark runs and finishes pointing to the results location.
All the results are located at AIXPRT/Results/ after the benchmark run.


## 4. Known issues
- Apt update commands in install/.sh scripts may not execute properly, **causing installation issues**

     For instance, when `sudo apt update` fails with `E: Could not get lock /var/lib/apt/lists/lock`.
     - **Fix**: Run `sudo rm /var/lib/dpkg/lock /var/cache/apt/archives/lock /var/lib/apt/lists/lock` before running setup scripts
