# ============================================================================
# Copyright (C) 2018 BenchmarkXPRT Development Community
# Licensed under the BENCHMARKXPRT DEVELOPMENT COMMUNITY MEMBER LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License by contacting Principled Technologies, Inc.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing grants and
# restrictions under the License.
#============================================================================

import sys
import os

import time
import mmap
import re
import numpy as np
import json
import subprocess

import os.path

sys.path.insert(1, os.path.join(os.environ['APP_HOME'], 'Harness'))
import resultsapi
# Setting default parameters
aarch="GPU"
precision="fp32"
batch_size_number=[1,2,4,8]
iterations=100
concurrent_instances=1

def get_params_from_json(dir_name):
    workload_details = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",dir_name,"workload_details.json")
    with open(workload_details) as data_file:
        data = json.load(data_file)
    data_file.close()
    batch_size_number=[]
    workloadName = data["name"]
    workloadID = data["id"]
    if not (data.get("requested_config",None)==None):
        requested_config = data["requested_config"]
        if not (requested_config.get("hardware",None)==None):
            aarch = data["requested_config"]["hardware"].upper()
        if not (requested_config.get("precision",None)==None):
            precision = data["requested_config"]["precision"]
        if not (requested_config.get("total_requests",None)==None):
            iterations = data["requested_config"]["total_requests"]
        if not (requested_config.get("batch_sizes",None)==None):
            batch_size_number = data["requested_config"]["batch_sizes"]
        if not (requested_config.get("concurrent_instances",None)==None):
            concurrent_instances = data["requested_config"]["concurrent_instances"]

    return(aarch, precision, iterations, batch_size_number , workloadName , workloadID,concurrent_instances)

aarch, precision, iterations, batch_size_number, workloadName, workloadID, concurrent_instances = get_params_from_json("ssdmobilenetv1")

if int(concurrent_instances)>1:
    print("Workload cannot run concurrent instances > 1")
    exit()

os.chdir(os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","bin","object_detection"))
path = os.path.join(os.getcwd(), 'object_detection_coco_ssdmobilenetv1.py')

outpath = os.path.join(os.environ['APP_HOME'], "Modules", "Deep-Learning", "workloads", "ssdmobilenetv1",
                       "result", "output", "console_out_ssdmobilenetv1.txt")
f = open(outpath, "w")
for j in batch_size_number:
	return_code = subprocess.call(['python', path, '--frozen_graph', 'ssd_mobilenet_v1_coco_2018_graph.pb', '--batch_size', str(j), '--aarch', aarch,'--prec', precision, '--iterations', str(iterations)], stdout=f)
