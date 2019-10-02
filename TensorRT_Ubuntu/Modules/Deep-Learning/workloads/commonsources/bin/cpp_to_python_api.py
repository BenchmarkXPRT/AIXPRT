# The caffe module needs to be on the Python path;
import sys
import os
import platform
import csv
import json
import subprocess
import result_dnn_api

# Check if the argument is passed for the model wprefix
modelPrefix = sys.argv[1]
batchSize = sys.argv[2]
aarch = sys.argv[3]
prec = sys.argv[4]
kernal_throughput = sys.argv[5]
kernal_latency = sys.argv[6]
system_throughput = sys.argv[7]
system_latency = sys.argv[8]
throughput_units = sys.argv[9]
latency_units = sys.argv[10]
iterCount = sys.argv[11]
dataset = sys.argv[12]
pct50 = sys.argv[13]
pct90 = sys.argv[14]
pct95 = sys.argv[15]
pct99 = sys.argv[16]
max_time = sys.argv[17]
min_time = sys.argv[18]
concurrent = sys.argv[19]

trt_version=subprocess.getoutput("dpkg -l | grep 'Meta package of TensorRT' | awk '{print $3}'")
#'v4.0.1.6'


def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`, 
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True


def writeBatchResultsToAPI():
    # Add the result API code here
    if is_number(kernal_throughput):
        ker_throughput = float(kernal_throughput)
    else:
        ker_throughput = "N/A"

    if is_number(kernal_latency):
        ker_latency = float(kernal_latency)
    else:
        ker_latency = "N/A"

    if is_number(system_throughput):
        sys_throughput = float(system_throughput)
    else:
        sys_throughput = "N/A"

    if is_number(system_latency):
        sys_latency = float(system_latency)
    else:
        sys_latency = "N/A"


    workLoadName = result_dnn_api.returnWorkloadName(modelPrefix)
    inputString = result_dnn_api.returnInputMap("TensorRT", dataset,batchSize , aarch, modelPrefix, trt_version, prec, iterCount)
    resultsString = result_dnn_api.returnBatchsizeResults(int(batchSize),
            ker_throughput, ker_latency, int(iterCount), modelPrefix,
            throughput_units, latency_units, sys_throughput, sys_latency,
            pct50,pct90,pct95,pct99,max_time,min_time, concurrent)
    # Now wrtie the info to the API
    result_dnn_api.writeResultsToAPI(workLoadName, inputString, resultsString)

sys.path.append('../../../../../Harness/')
import resultsapi	#For MLXBench results API
writeBatchResultsToAPI()




