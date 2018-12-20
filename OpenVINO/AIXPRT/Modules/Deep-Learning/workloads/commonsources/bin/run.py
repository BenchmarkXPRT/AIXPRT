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
import numpy as np
import shutil
sys.path.insert(0, os.path.join(os.environ['APP_HOME'],"Harness","CallBacks"))
import workloadEventCallBack as event
import re

DL_DTK_Version="v2018.4.420"
display_target=5

# Setting default parameters
aarch="GPU"
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
        if not (requested_config.get("precision",None)==None):
            precision = data["requested_config"]["precision"]
        if not (requested_config.get("iterations",None)==None):
            iterations = data["requested_config"]["iterations"]
        if not (requested_config.get("batch_sizes",None)==None):
            batch_size_number = data["requested_config"]["batch_sizes"]
        if not (requested_config.get("concurrent_requests",None)==None):
            requests = data["requested_config"]["concurrent_requests"]

    return(aarch, precision, iterations, batch_size_number , workloadName , workloadID, requests)

# Image Classification
def image_classification(model_name,dir_name,model_input_size):
    aarch, precision, iterations, batch_size_number, workloadName, workloadID, concurrent_requests = get_params_from_json(dir_name)
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
            if (aarch == "MYRIAD"):
                print("Workload doesn't support batch size greater than 1")
                exit()

        image_folder = create_batch_files(j,"input_images",model_input_size)
        print("   Running "+model_name+" batch"+str(j)+" "+precision + " " + aarch)
        file_out = os.path.join('..','..',dir_name,'result','output',model_name+'_'+precision+'_batch'+str(j)+'_'+aarch+'.txt')

        if (concurrent_requests>1):
            display_target=1 #limit the size of output file
            if (concurrent_requests>iterations):
                iterations=concurrent_requests
                print("    iterations should be bigger than concurrent_requests, ammending iterations = " + str(iterations))
            print("    Delivering "+ str(concurrent_requests) + " concurrent requests to inference engine....")
            application = os.path.realpath(os.getcwd()+"/image_classification_async")
            command = application + " -a "+model_name+" -b "+str(j)+" -aarch "+aarch+" -prec "+precision+" -d "+aarch+\
                      " -i "+image_folder+" -m "+model_path+" -nt "+str(display_target)+" -ni "+str(iterations)+" -nireq "+str(concurrent_requests)
        else:
            application = os.path.realpath(os.getcwd()+"/image_classification")
            command = application + " -a "+model_name+" -b "+str(j)+" -aarch "+aarch+" -prec "+precision+" -d "+aarch+\
              " -i "+image_folder+" -m "+model_path+" -nt "+str(display_target)+" -ni "+str(iterations)

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

    model_path = set_model_path(model_name,precision)
    os.chdir(os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","bin"))
    for j in batch_size_number:
        print("   Running "+model_name+" batch"+str(j)+" "+precision + " " + aarch)

        if (j > 1):
            if (aarch == "MYRIAD"):
                print("Workload doesn't support batch size greater than 1")
                exit()
        model_input_size = 300
        image_folder = create_batch_files(j,"input_images",model_input_size)

        model_path=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","models",model_name,model_name+'_'+precision+'_b'+str(j)+'.xml')

        file_out = os.path.join('..','..',dir_name,'result','output',model_name+'_'+precision+'_batch'+str(j)+'_'+aarch+'.txt')

        if (concurrent_requests>1):
            display_target=1 #limit the size of output file
            if (concurrent_requests>iterations):
                iterations=concurrent_requests
                print("    iterations should be bigger than concurrent_requests, ammending iterations = " + str(iterations))
            print("    Delivering "+ str(concurrent_requests) + " concurrent requests to inference engine....")
            application = os.path.realpath(os.getcwd()+"/object_detection_ssd_async")
            command = application + " -i " + image_folder + " -m " + model_path + " -ni " + str(iterations) + " -d " + aarch +\
             " -nireq " + str(concurrent_requests) + " -b "+str(j) + " -prec "+precision+" -dir "+ dir_name + " -a "+model_name + " -aarch "+aarch + " "
        else:
            application = os.path.realpath(os.getcwd()+"/object_detection_ssd")
            command = application + " -i " + image_folder + " -m " + model_path + " -ni " + str(iterations) + " -d " + aarch +\
                      " -b "+str(j) + " -prec "+precision+" -dir "+ dir_name + " -a "+model_name + " -aarch "+aarch + " "

        with open(file_out, 'w') as out:
            command = event.workloadStarted(workloadName , workloadID , j , precision , aarch, command)
            #Starting workload
            return_code = subprocess.call(command, stdout=out, shell=True)
            #Ending Workload
            event.workloadEnded(workloadName , workloadID , j , precision , aarch)
        remove_batch_files(image_folder)

def create_batch_files(batch_size,validation_folder,size):
    images_list = list_all_files_sorted(validation_folder)

    current_batch = images_list[0:batch_size]
    image_folder = os.path.join("..","..","..","packages",validation_folder,"batch"+str(batch_size))
    if os.path.isdir(image_folder):
            shutil.rmtree(image_folder)
    os.makedirs(image_folder)
    for file in current_batch[:]:
        filename, fileext = os.path.splitext(file)
        image_out = os.path.join("..","..","..","packages",validation_folder,"batch"+str(batch_size),os.path.basename(filename) + '.bmp')
        img_path = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages",validation_folder,str(file))
        i = Image.open(img_path)

        if size == 1024:
            j = i.resize((int(size*2), int(size)), Image.ANTIALIAS)
            j.save(image_out)
        else:
            j = i.resize((int(size), int(size)), Image.ANTIALIAS)
            j.save(image_out)
    return(image_folder)

def list_all_files_sorted(validation_folder):
    mypath=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages",validation_folder)
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    onlyfiles_sorted = sorted_alphanumerically(onlyfiles)
    return(onlyfiles_sorted)

def remove_batch_files(image_folder):
    shutil.rmtree(image_folder)

# For cropping images
def central_crop(img, central_fraction=0.875):
    y,x,c = img.shape
    r = 1-central_fraction # How much to discard

    starty = int(round(y*r/2.0))# starting crop for y
    endy = y - starty

    startx = int(round(x*r/2.0))
    endx = x - startx
#    print("Startx: {}\tEndx {}\tStarty {}\tEndy {}".format(startx, endx, starty, endy))
    return img[starty:endy, startx:endx]

#Resize and centercrop for validation
def resize(size,direc,current_batch):
    for file in current_batch[:]:
        path = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","input_images",str(file))
        filename, fileext = os.path.splitext(file)

        #%% PIL + Numpy preproc
        img = Image.open(path).convert('RGB')
        img = central_crop(np.array(img,dtype=np.float32)) # Crop central area of image
        img = Image.fromarray(np.uint8(img), 'RGB')
        crop_img = img.resize((int(size), int(size)), resample=Image.BILINEAR) # Resize to input dimensions

        image_out = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","input_images",str(direc)+'_'+str(size),os.path.basename(filename) + '.bmp')
        crop_img.save(image_out)

# Set ENV Paths
def set_env_path():
    try:
        os.environ['LD_LIBRARY_PATH']=os.environ['LD_LIBRARY_PATH']+":"+os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","plugin")
    except KeyError:
        os.environ['LD_LIBRARY_PATH']=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","plugin")

# Set Model Path
def set_model_path(model_name,precision):
    model_path=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","models",model_name,model_name+'_'+precision+'.xml')
    return(model_path)
