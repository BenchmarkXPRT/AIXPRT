# About

AIXPRT is an AI benchmark which has the capability of running on below platforms  
* AMD CPU
* AMD GPU
* Intel CPU
* Intel GPU
* Intel VPU (Myriad)
* Nvidia GPU
* Nvidia Xavier

Workloads are implemented using the publicly available libraries and SDKs for each platform.

# Run the Benchmark

### By Cloning the repo 
   1.Install git with lfs.Instructions are found at https://packagecloud.io/github/git-lfs/install.

   2. Clone the repository 
    * git clone https://github.com/BenchmarkXPRT/AIXPRT.git

   3. Navigate to AIXPRT/Modules/Deep-Learning/README.md of the cloned branch and follow instructions to run the benchmark.

   4. AIXPRT application runs on default configuration provided by the workloads. However once a default config file is generated, user        can edit this config to change the workload behavior.

Note: A config file will be generated at AIXPRT/Config/{config_name}.json after the first run of index.py. One can edit this config file to run the specific workload in a specific way .

[How to edit config](TODO : add url to AIXPRT/EditConfig.md file )

## Report Bugs
Please Report your bugs under Issues tab of this project

## Develop
Please follow the [instructions](TODO : Add url to AIXPRT AddWorkload.md) to add new workloads to the benchmark.
