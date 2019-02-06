# The caffe module needs to be on the Python path;
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
import platform
import csv
import json
import result_dnn_api
import subprocess
import re

# Check if the argument is passed for the model wprefix
modelPrefix = sys.argv[1]
batchSize = sys.argv[2]
aarch = sys.argv[3]
prec = sys.argv[4]
imgPerSec = sys.argv[5]
iterCount = sys.argv[6]
timeInMillisec = sys.argv[7]
StandardDev = sys.argv[8]
# --- Synchronize logging
concurrent_instances = sys.argv[9]

# version update from: deployment_tools/documentation/InstallingForLinux.html

def getOpenVINOversion():
    exePATH = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","bin","image_classification")
    command = "ldd %s" % exePATH
    ldd_out_temp = (subprocess.check_output(command, shell=True).strip())
    ldd_out = ldd_out_temp.decode("utf-8")
    OpenVINOversion = ""
    for line in ldd_out.splitlines():
        line = str(line)
        if ("/opt/intel/computer_vision_sdk" in str(line)) and ("/deployment_tools" in str(line)):
            match = re.match(r'\t(.*) => (.*) \(0x', line)
            if match:
                myString = match.group(2)
                OpenVINOversion = myString[(myString.index("/opt/intel/")+len("/opt/intel/")):myString.index("/deployment_tools")]
    return OpenVINOversion


version = getOpenVINOversion()
s = {
'cpu': 'MKLDNN 18WW32.5',
'gpu': 'clDNN 18WW32.5',
'myriad': 'myriadPlugin 18WW32.5',
'fpga': 'DLA 18WW32.5',
'hddl': 'hddlPlugin 18WW32.5',
'openvx': 'OpenVX v1.1',
'opencv': 'OpenCV v3.4.2',
'opencl': 'OpenCL v2.1',
'intel-opencl': 'intel-opencl_18.26.10987_amd64'
}

usedAcceleratorList = []
usedAcceleratorList.append(s[aarch.lower()])
def writeBatchResultsToAPI():
    # Add the result API code here
    resInImgPerSec = float(imgPerSec)
    forwardTime = float(timeInMillisec)
    workLoadName = result_dnn_api.returnWorkloadName(modelPrefix)
    inputString = result_dnn_api.returnInputMap("OpenVINO ", version , "ILSVRC 2012",batchSize , aarch, modelPrefix+'net',prec,iterCount,usedAcceleratorList)
    resultsString = result_dnn_api.returnBatchsizeResults(int(batchSize), round(resInImgPerSec,3), forwardTime, int(iterCount), modelPrefix, aarch.lower(), StandardDev, int(concurrent_instances) )
    # Now wrtie the info to the API
    result_dnn_api.writeResultsToAPI(workLoadName, inputString, resultsString)

writeBatchResultsToAPI()
