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
import subprocess
import os
import json
from PIL import Image
#import numpy as np
import shutil
sys.path.insert(0, os.path.join(os.environ['APP_HOME'],"Harness","CallBacks"))
import workloadEventCallBack as event
import re
import random

DL_DTK_Version="v2018.5.445"
display_target=5

# Setting default parameters
aarch="CPU"
precision="fp32"
batch_size_number=[1,2,4,8]
iterations=10
concurrent_requests=1

def sorted_alphanumerically(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

#get and set paramsset
def get_params_from_json(dir_name):
    workload_details = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",dir_name,"workload_details.json")
    with open(workload_details) as data_file:
        data = json.load(data_file)
    data_file.close()
    requests=1
    batch_size_number=[]
    workloadName = data["name"]
    workloadID = data["id"]
    if not (data.get("requested_config",None)==None):
        requested_config = data["requested_config"]
        if not (requested_config.get("hardware",None)==None):
            aarch = data["requested_config"]["hardware"].upper()
            if("," in aarch):
                # hetero mode 
                aarch = "HETERO:"+aarch
        if not (requested_config.get("precision",None)==None):
            precision = data["requested_config"]["precision"]
        if not (requested_config.get("total_requests",None)==None):
            iterations = data["requested_config"]["total_requests"]
        if not (requested_config.get("batch_sizes",None)==None):
            batch_size_number = data["requested_config"]["batch_sizes"]
        if not (requested_config.get("concurrent_instances",None)==None):
            requests = data["requested_config"]["concurrent_instances"]

    return(aarch, precision, iterations, batch_size_number , workloadName , workloadID, requests)

# Image Classification
def image_classification(model_name,dir_name,model_input_size):
    aarch, precision, iterations, batch_size_number, workloadName, workloadID, concurrent_requests = get_params_from_json(dir_name)

    if (aarch.upper() in ["GPU", "HDDL", "MYRIAD"]) and (precision=="int8"):
        print("INT8 not supported on {}".format(aarch))
        sys.exit()

    display_target=5

    set_env_path()
    if aarch == "FPGA":
        aarch = "HETERO:FPGA,CPU"
        os.environ["DLA_AOCX"]=os.environ["BITSTREAM"]
        os.environ["CL_CONTEXT_COMPILER_MODE_INTELFPGA"]="3"
    model_path = set_model_path(model_name,precision)
    os.chdir(os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","bin"))

    for j in batch_size_number:

        if (j > 1):
            if (aarch in ["MYRIAD", "HDDL"]):
                print("No support for BATCH SIZE {} on {}".format(j, aarch.upper()))
                sys.exit()

        image_folder = create_batch_files(j,"input_images",model_input_size)
        print("   Running "+model_name+" batch"+str(j)+" "+precision + " " + aarch)
        file_out = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",dir_name,'result','output',model_name+'_'+precision+'_batch'+str(j)+'_'+aarch+'.txt')
        #  base command 
        application = os.path.realpath(os.getcwd()+"/benchmark_app")
        command = application + " -a "+model_name+" -b "+str(j)+" -aarch "+aarch+" -prec "+precision+" -d "+aarch+\
                      " -i "+image_folder+" -m "+model_path
        #  check for the provided concurrent_instances            #   
        if not (str(concurrent_requests) == "auto"):
            command = command +" -nireq "+str(concurrent_requests)
            # sdk defaults to asyc mode . If user inputs 1 instance then pass that to the sdk
            if (int(concurrent_requests) == 1):
                    command = command +" -api sync"
    #     check for the provided total_requests
        if not (str(iterations) == "auto"):
            command = command + " -niter "+str(iterations)
            
        with open(file_out, 'w') as out:
            command = event.workloadStarted(workloadName , workloadID , j , precision , aarch, command)
            #Starting workload
            return_code = subprocess.call(command, stdout=out, shell=True)
            #Ending Workload
            event.workloadEnded(workloadName , workloadID , j , precision , aarch)
        remove_batch_files(image_folder)

# Object Detection SSD
def object_detection_ssd(model_name,dir_name,dataset):
    aarch, precision, iterations, batch_size_number, workloadName, workloadID, concurrent_requests = get_params_from_json(dir_name)
    set_env_path()

    if (aarch.upper() in ["GPU", "HDDL", "MYRIAD"]) and (precision=="int8"):
        print("INT8 not supported on {}".format(aarch))
        sys.exit()

    model_path = set_model_path(model_name,precision)
    os.chdir(os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","bin"))
    for j in batch_size_number:

        print("   Running "+model_name+" batch"+str(j)+" "+precision + " " + aarch)

        if (j > 1):
            if (aarch in ["MYRIAD", "HDDL"]):
                print("No support for BATCH SIZE {} on {}".format(j, aarch.upper()))
                sys.exit()

        model_input_size = 300
        image_folder = create_batch_files(j,"input_images",model_input_size)
        file_out = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",dir_name,'result','output',model_name+'_'+precision+'_batch'+str(j)+'_'+aarch+'.txt')
        #  base command 
        application = os.path.realpath(os.getcwd()+"/benchmark_app")
        command = application + " -a "+model_name+" -b "+str(j)+" -aarch "+aarch+" -prec "+precision+" -d "+aarch+\
                      " -i "+image_folder+" -m "+model_path
        #  check for the provided concurrent_instances            #   
        if not (str(concurrent_requests) == "auto"):
            command = command +" -nireq "+str(concurrent_requests)
            # sdk defaults to asyc mode . If user inputs 1 instance then pass that to the sdk
            if (int(concurrent_requests) == 1):
                    command = command +" -api sync"
        # check for the provided total_requests
        if not (str(iterations) == "auto"):
            command = command + " -niter "+str(iterations)
      
        with open(file_out, 'w') as out:
            command = event.workloadStarted(workloadName , workloadID , j , precision , aarch, command)
            #Starting workload
            return_code = subprocess.call(command, stdout=out, shell=True)
            #Ending Workload
            event.workloadEnded(workloadName , workloadID , j , precision , aarch)
        remove_batch_files(image_folder)

def create_batch_files(batch_size,validation_folder,size):
    images_list = list_all_files_sorted(validation_folder)
    random.shuffle(images_list)
    Num_images = len(images_list)
    current_batch = images_list[:min(batch_size, Num_images)]

    image_folder = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages",validation_folder,"batch"+str(batch_size))
    if os.path.isdir(image_folder):
            shutil.rmtree(image_folder)
    os.makedirs(image_folder)

    for file in current_batch[:]:
        filename, fileext = os.path.splitext(file)
        image_out = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages",validation_folder,"batch"+str(batch_size),os.path.basename(filename) + '.bmp')
        img_path = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages",validation_folder,str(file))
        i = Image.open(img_path).convert("RGB")

        if size == 1024:
            j = i.resize((int(size*2), int(size)), Image.ANTIALIAS)
            j.save(image_out)
        else:
            j = i.resize((int(size), int(size)), Image.ANTIALIAS)
            j.save(image_out)

    # top up if batch size is more than available images
    images_list = list_all_files_sorted(image_folder)
    num_created_images = len(images_list)
    copy_iteration = 0
    while num_created_images < batch_size:
        for file in images_list:

            filename, fileext = os.path.splitext(file)
            image_out = os.path.join(image_folder,str(copy_iteration) + "_" + os.path.basename(filename) + '.bmp')
            img_path = os.path.join(image_folder, str(file))
            i = Image.open(img_path).convert("RGB")

            if size == 1024:
                j = i.resize((int(size*2), int(size)), Image.ANTIALIAS)
                j.save(image_out)
            else:
                j = i.resize((int(size), int(size)), Image.ANTIALIAS)
                j.save(image_out)

            num_created_images+=1 # Update number of created images

            if num_created_images == batch_size:
                break

        copy_iteration+=1
        images_list = list_all_files_sorted(image_folder)
        num_created_images = len(images_list)

    #print("\nCreated {} additional copies.\n".format(num_created_images - Num_images))
    return(image_folder)

def list_all_files_sorted(validation_folder):
    mypath=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages",validation_folder)
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    onlyfiles_sorted = sorted_alphanumerically(onlyfiles)
    return(onlyfiles_sorted)

def remove_batch_files(image_folder):
    shutil.rmtree(image_folder)

# Set ENV Paths
def set_env_path():
    import sys

    if 'win' in sys.platform:
        try:
            os.environ['PATH']=os.environ['PATH']+";"+os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","plugin")
        except KeyError:
            os.environ['PATH']=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","plugin")

    else:
        try:
            os.environ['LD_LIBRARY_PATH']=os.environ['LD_LIBRARY_PATH']+":"+os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","plugin")
        except KeyError:
            os.environ['LD_LIBRARY_PATH']=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","plugin")
        if not 'HDDL_INSTALL_DIR' in os.environ.keys():
            openvino_version_file = os.path.join(os.environ['APP_HOME'],"Harness","OpenVINO_BUILD.txt")
            if os.path.isfile(openvino_version_file):
                with open(openvino_version_file,'r') as fid:
                    text = fid.read().splitlines()
                    OpenVINO_PATH = text[0].split(':')[-1]

                os.environ['HDDL_INSTALL_DIR']= os.path.join(OpenVINO_PATH,"deployment_tools","inference_engine","external","hddl")

# Set Model Path
def set_model_path(model_name,precision):
    model_path=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","models",model_name,model_name+'_'+precision+'.xml')
    return(model_path)
