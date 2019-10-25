# About

AIXPRT is an AI benchmark which has the capability of running on below platforms  
* AMD CPU
* AMD GPU
* Intel CPU
* Intel GPU
* Intel VPU (Myriad)
* NVIDIA GPU
* NVIDIA Xavier

Workloads are implemented using the publicly available libraries and SDKs for each platform.

# Run the Benchmark
    ### Download as zip file ( recommended)
        1. Navigate to aixprt [download](https://www.principledtechnologies.com/benchmarkxprt/aixprt/preview.php) page.
        
        2. Enter the requested information to select and download the package. 
        
        3.Navigate to AIXPRT/Modules/Deep-Learning/README.md of the downloaded package and follow instructions for running the benchmark.
        
        4. AIXPRT application runs on default configuration provided by the workloads. However once a default config file is generated, user can edit this config to change the workload behavior.
        
    ### By Cloning the repo 
       1.Install git with lfs.Instructions are found at https://packagecloud.io/github/git-lfs/install.
    
       2. Clone the repository 
        * git clone https://github.com/BenchmarkXPRT/AIXPRT.git
    
       3. Navigate to AIXPRT/Modules/Deep-Learning/README.md of the cloned branch and follow instructions for running the benchmark.
    
       4. AIXPRT application runs on default configuration provided by the workloads. However once a default config file is generated, user can edit this config to change the workload behavior.
    
    
Note: A config file will be generated at AIXPRT/Config/{config_name}.json after the first run of index.py. One can edit this config file to run the specific workload in a specific way or use one of the provided config files. 

[How to edit config](https://github.com/BenchmarkXPRT/AIXPRT/blob/master/Tensorflow/AIXPRT/EditConfig.md)

# Result Submission. 
Please follow the guidelines in [ResultSubmission](https://github.com/BenchmarkXPRT/AIXPRT/blob/master/Tensorflow/AIXPRT/ResultSubmission.md)

# Support
If you need technical support or have any questions, please send a message to BenchmarkXPRTsupport@principledtechnologies.com.

# Contribute
Please follow the [instructions](https://github.com/BenchmarkXPRT/AIXPRT/blob/master/Tensorflow/AIXPRT/Add_Edit_Workload.md) to add new workloads to the benchmark.
