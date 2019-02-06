### How to edit config

AIXPRT application runs on default confugaration provided by the workloads. However once a default config file is generated , user can edit this config to change the workload behaviour. 

NOTE : Please perform atleast 1 run of the benchmark so that a default config is generated at AIXPRT/Config/

1. If you wish to run same config for n number of times , navigate to AIXPRT/Config open the configuration json file and edit the “iterations” value to the number you want. Save the file and run the benchmark. Application generates results for each iteration in different result files.

2. If you have multiple configuration json files present in AIXPRT/Config folder, then applications runs all of the configuration and generates a separate result file for each config. This is mainly intended for automation purposes .If you do not want this to happen, simple delete the unwanted configuration json files.

3. If you wish to run only a specific workload/workloads in a config, please navigate to AIXPRT/Config/jsonFileYouWantToRun.json under the key “workloads_config”, each item is a workload. Delete the workloads which you do not want to run.

Example  :  The json below has only Resnet-50 workload.

```
{
    "iteration": 1,
    "module": "Deep-Learning",
    "workloads_config": [
        {
            "batch_sizes": [
                1,
                2,
                4,
                8
            ],
            "hardware": "cpu",
            "iterations": 10,
            "name": "Resnet-50",
            "precision": "fp32",
            "runtype": "performance"
        }
    ]
}
```
4. Each workload item has components like "hardware" , "iterations" , "precision" which are configurable. Here are the options each can take
* "hardware":"cpu" (or) "hardware":"gpu" (or) "hardware":"myriad"
* "precision": "fp32" (or) "precision": "fp16" (or) "precision": "int8"
