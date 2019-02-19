import sys
import os

import time
import mmap
import re
from os.path import expanduser
import numpy as np
import json
import subprocess
from subprocess import Popen
from numpy import genfromtxt

import os.path
sys.path.insert(1, os.path.join(os.environ['APP_HOME'], 'Harness'))
import resultsapi
import utils
import tensorflow as tf

workload_dir = "ssdmobilenetv1"

def writeBatchResults(path_list,batchsize,aarch,iterations,instances, total_requests, precision):
    #read timings from csv file and log results

    for path in path_list:
        csv_data = genfromtxt(path, delimiter=',')
        if 'np_from_csv_data' in dir():
            np_from_csv_data = np.vstack((np.array(np_from_csv_data), csv_data))
        else:
            np_from_csv_data = csv_data
    async_timings = (np_from_csv_data[:,1] - np_from_csv_data[:,0])*1000

    if (np_from_csv_data.shape == (2,)):
        tend_max = np_from_csv_data[1]
        tstart_min = np_from_csv_data[0]
    else:
        tstart_max, tend_max = np_from_csv_data.max(axis=0)
        tstart_min, tend_min = np_from_csv_data.min(axis=0)
    tcalc = (tend_max - tstart_min)/(iterations * instances)
    speed_mean = (batchsize)/tcalc
    time_mean = tcalc*1000
    labelstr = "Batch "+ str(batchsize)
    additional_info_details = {}
    additional_info_details["total_requests"] = total_requests
    additional_info_details["concurrent_instances"] = instances
    additional_info_details["50_percentile_time"] = np.percentile(async_timings, 50)
    additional_info_details["90_percentile_time"] = np.percentile(async_timings, 90)
    additional_info_details["95_percentile_time"] = np.percentile(async_timings, 95)
    additional_info_details["99_percentile_time"] = np.percentile(async_timings, 99)
    accelerator_lib_details = {}

    if (aarch.lower()=="cpu"):
        accelerator_lib_details["cpu_accelerator_lib"] = ""
    else:
        accelerator_lib_details["gpu_accelerator_lib"] = ""
    workloadInput={
          "Tensorflow": "1.10",
          "architecture":aarch,
          "precision":precision,
          "iterations":iterations,
          "instances": instances,
          "accelerator_lib": [accelerator_lib_details],
          "framework": utils.getTensorflowInfo()
         }
    results=[
          {
          "label":labelstr,
          "system_latency":time_mean,
          "system_latency_units":"milliseconds",
          "system_throughput":speed_mean,
          "system_throughput_units":"imgs/sec",
          "additional info":[additional_info_details]
          }
        ]
    resultsapi.createResultJson("SSD-MobileNet-v1", workloadInput, results)


def get_params_from_json(dir_name):
    # defaults
    aarch="CPU"
    precision="fp32"
    batch_size_number=[1,2,4,8]
    total_requests = 10
    concurrent_instances = 1
    framework_graphTransform = None
    workload_details = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",dir_name,"workload_details.json")
    with open(workload_details) as data_file:
        data = json.load(data_file)
    data_file.close()
    batch_size_number=[]
    workloadName = data["name"]
    workloadID = data["id"]
    instance_allocation = []
    if not (data.get("requested_config",None)==None):
        requested_config = data["requested_config"]
        if not (requested_config.get("hardware",None)==None):
            aarch = data["requested_config"]["hardware"].upper()
        if not (requested_config.get("precision",None)==None):
            precision = data["requested_config"]["precision"]
        if not (requested_config.get("batch_sizes",None)==None):
            batch_size_number = data["requested_config"]["batch_sizes"]
        if not (requested_config.get("total_requests",None)==None):
            total_requests = data["requested_config"]["total_requests"]
        if not (requested_config.get("concurrent_instances",None)==None):
            concurrent_instances = data["requested_config"]["concurrent_instances"]
        if not (requested_config.get("framework_config",None)==None):
            framework_config = data["requested_config"]["framework_config"]
        if not (requested_config.get("setNUMA",None)==None):
            setNUMA = data["requested_config"]["setNUMA"]
        if not (requested_config.get("env_variables",None)==None):
            env_variables = data["requested_config"]["env_variables"]
        if not (requested_config.get("framework_graphTransform",None)==None):
            framework_graphTransform = data["requested_config"]["framework_graphTransform"]
        if not (requested_config.get("instance_allocation",None)==None):
            instance_allocation = data["requested_config"]["instance_allocation"]

    return(aarch, precision, batch_size_number , workloadName , workloadID,total_requests , concurrent_instances ,framework_config,setNUMA,env_variables,framework_graphTransform,instance_allocation)

# get user requested run parameters
aarch, precision, batch_size_number, workloadName, workloadID, total_requests, concurrent_instances ,framework_config,setNUMA,env_variables,framework_graphTransform,instance_allocation= get_params_from_json(workload_dir)

# Set environment variable if provided
if(env_variables):
    for key, value in env_variables.items():
        os.environ[key] = value

os.chdir(os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","bin","object_detection"))
path = os.path.join(os.getcwd(), 'object_detection_coco_ssdmobilenetv1.py')
outpath = os.path.join(os.environ['APP_HOME'], "Modules", "Deep-Learning", "workloads", workload_dir,
                       "result", "output", "console_out_ssdmobilenetv1.txt")
f = open(outpath, "w")

#  total_requests NUMBER should always be perfectly divisible by concurrent_instances number
if total_requests % concurrent_instances == 0:
    iterations = int(total_requests/concurrent_instances)
else:
    print("ERROR: total_requests should be a mutiple of concurrent_instances")
    sys.exit()
model_dir=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","models")
if(precision=='int8'):
    frozen_graph = 'ssdmobilenet_int8_pretrained_model.pb'
else:
    frozen_graph = 'ssd_mobilenet_v1_coco_2017_graph.pb'
if not os.path.isfile(model_dir+"/"+frozen_graph):
    print("ERROR: Model file not found.")
    sys.exit()
for j in batch_size_number:
    commands = []
    path_list = []
    allocation = []
    if setNUMA:
        if  not len(instance_allocation) == concurrent_instances:
            print("Please add instance allocation to your config as the NUMA is set to true")
            sys.exit()
        else:
            for item in instance_allocation:
                cmd = ""
                for key ,value in item.items():
                    cmd+= "--"+key+"="+value+" "
                allocation.append(str(cmd))  
    for ins in range(concurrent_instances):
        #For writing timings from all instances
        csv_file_path = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",
        workload_dir,'result','output','ssdmobilenetv1_batch_'+str(j)+'_'+precision+'_concurrent_instance'+str(ins)+'.csv')
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
        command = ""
        if(setNUMA):
            #instantiate numactl variables
            command = "numactl "+allocation[ins]
        command = command + str('python'+' '+ path+' '+ '--frozen_graph'+' '+ frozen_graph+' '+ '--batch_size'+' '+ str(j)+
        ' '+ '--aarch'+' '+ aarch+' '+ '--iterations'+' '+ str(iterations)+' '+
        '--instance'+' '+str(ins)+' '+'--workload_dir'+' '+ workload_dir+' '+'--csv_file_path'+' '+csv_file_path+' '+ '--precision'+' '+ precision)
        commands.append(command)
        path_list.append(csv_file_path)
        # run in parallel
    print(commands)
    processes = [Popen(cmd, shell=True,stdout=f) for cmd in commands]
    for p in processes: p.wait()
    writeBatchResults(path_list,j,aarch,iterations,concurrent_instances, total_requests, precision)
