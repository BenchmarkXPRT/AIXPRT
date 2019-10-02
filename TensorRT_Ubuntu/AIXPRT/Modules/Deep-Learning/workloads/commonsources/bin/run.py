'''
File: run.py
Developer: Rohit Kalidindi
Date: 03/19/2018
Desc: Common file to run all workloads.
'''
import sys
import subprocess
import os
import json
import glob
from PIL import Image
import numpy as np
import shutil
sys.path.insert(0, os.path.join(os.environ['APP_HOME'],"Harness","CallBacks"))
import workloadEventCallBack as event
import cv2

# Setting default parameters
aarch="GPU"
precision="fp32"
batch_size_number=[1,2,4,8]

#get and set paramsset
def get_params_from_json(dir_name):
    workload_details = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",dir_name,"workload_details.json")
    with open(workload_details) as data_file:
        data = json.load(data_file)
    data_file.close()
    iterations=10
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
        if not (requested_config.get("total_requests",None)==None):
            iterations = data["requested_config"]["total_requests"]
        if not (requested_config.get("batch_sizes",None)==None):
            batch_size_number = data["requested_config"]["batch_sizes"]
        if not (requested_config.get("concurrent_instances",None)==None):
            requests = data["requested_config"]["concurrent_instances"]
    return(aarch, precision, iterations, batch_size_number , workloadName , workloadID, requests)

# Image Classification
def image_classification(model_name,dir_name):
    aarch, precision, iterations, batch_size_number,workloadName , workloadID, requests= get_params_from_json(dir_name)
    if int(requests)>=1:
        multistream_resnet(model_name,dir_name)
        exit()
    dataset="input_images"
    set_env_path()
    model_path = set_model_path(model_name,"caffemodel")
    deploy_path = set_model_path(model_name,"prototxt")
    calib_path = set_calibration_path(model_name,precision)
    precision_flag=set_precision_flag(precision)
    image_path=set_image_path(dataset)
    os.chdir(os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","bin"))
    for j in batch_size_number:
        print("Running "+model_name+" batch"+str(j)+" "+precision+"\n")

        file_out = os.path.join('..','..',dir_name,'result','output',model_name+'_'+precision+'_batch'+str(j)+'_'+aarch+'.txt')
        command = "./image_classification --output='prob' --iterations="+str(iterations)+" --mod="+model_name+" --deploy="+deploy_path+" --model="+model_path+" --batch="+str(j)+" --calib="+calib_path+" "+precision_flag+" -d "+image_path+"/"
        if not os.path.exists(os.path.dirname(file_out)):
            os.makedirs(os.path.dirname(file_out))
        with open(file_out, 'w') as out:
            return_code = subprocess.call(command, stdout=out, shell=True)

def multistream_resnet(model_name,dir_name):
    aarch, precision, iterations, batch_size_number,workloadName , workloadID, requests= get_params_from_json(dir_name)
    dataset="input_images"
    set_env_path()
    model_path = set_model_path(model_name,"caffemodel")
    deploy_path = set_model_path(model_name,"prototxt")
    calib_path = set_calibration_path(model_name,precision)
    precision_flag=set_precision_flag(precision)
    image_path=set_image_path(dataset)
    os.chdir(os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","bin"))
    for j in batch_size_number:
        print("Running "+model_name+" batch"+str(j)+" "+precision+"\n")
        file_out = os.path.join('..','..',dir_name,'result','output',model_name+'_'+precision+'_batch'+str(j)+'_'+aarch+'.txt')
        command = "./multi_stream_resnet "+deploy_path+" "+model_path+" "+str(requests)+" "+str(iterations)+" "+str(j)+" "+image_path+"/"
        if precision == "int8":
                command=command+" int8 "+calib_path
        elif precision == "fp16":
                command=command+" fp16"
        if not os.path.exists(os.path.dirname(file_out)):
            os.makedirs(os.path.dirname(file_out))
        with open(file_out, 'w') as out:
            return_code = subprocess.call(command, stdout=out, shell=True)

def object_detection_ssd(dir_name):
    aarch, precision, iterations, batch_size_number,workloadName , workloadID, requests= get_params_from_json(dir_name)
    if int(requests)>=1:
        multistream_ssd(dir_name)
        exit()
    set_env_path()

    os.chdir(os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","bin"))
    for j in batch_size_number:
        print("Running "+dir_name+" batch"+str(j)+" "+precision+"\n")
        file_out = os.path.join('..','..',dir_name,'result','output',dir_name+'_'+precision+'_batch'+str(j)+'_'+aarch+'.txt')
        command = "./uff_ssd --t "+str(iterations)+" --p "+precision+" --b "+str(j)
        if precision == "int8":
            command = command + " --i"
        if not os.path.exists(os.path.dirname(file_out)):
            os.makedirs(os.path.dirname(file_out))
        with open(file_out, 'w') as out:
            return_code = subprocess.call(command, stdout=out, shell=True)

def multistream_ssd(dir_name):
    aarch, precision, iterations, batch_size_number,workloadName , workloadID, requests= get_params_from_json(dir_name)
    set_env_path()

    os.chdir(os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","bin"))
    model_path=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","models","ssd-mobilenet_v1","sample_ssd.uff")
    ppm_folder="../../../packages/input_ppm/"
    preparePPMImages("../../../packages/input_images/",ppm_folder)
    for j in batch_size_number:
        print("Running SSD-MobileNet-v1 batch"+str(j)+" "+precision+"\n")
        file_out = os.path.join('..','..',dir_name,'result','output','SSD-MobileNet-v1_'+precision+'_batch'+str(j)+'_'+aarch+'.txt')
        command = "./multi_stream_ssd deploy "+model_path+" "+str(requests)+" "+str(iterations)+" "+str(j)+" "+ppm_folder
        if precision == "int8":
                command=command+" int8 calib"
        elif precision == "fp16":
                command=command+" fp16"
        if not os.path.exists(os.path.dirname(file_out)):
            os.makedirs(os.path.dirname(file_out))
        with open(file_out, 'w') as out:
            return_code = subprocess.call(command, stdout=out, shell=True)

# Set ENV Paths
def set_env_path():
    try:
        os.environ['LD_LIBRARY_PATH']=os.environ['LD_LIBRARY_PATH']+":"+os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","plugin")
    except KeyError:
        os.environ['LD_LIBRARY_PATH']=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","plugin")

# Set Model Path
def set_model_path(model_name,ext):
    model_path=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","models",model_name,model_name+'.'+ext)
    return(model_path)


def set_calibration_path(model_name,precision):
    if(precision == "int8"):
        calibration_path=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","models",model_name,"CalibrationTable")
    else:
        calibration_path=""
    return(calibration_path)

def set_image_path(dataset):
    dataset_path=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages",dataset)
    return(dataset_path)

def set_precision_flag(precision):
    if(precision == "fp32"):
        preciosn_flag=""
    elif(precision == "fp16"):
        preciosn_flag="--fp16"
    elif(precision == "int8"):
        preciosn_flag="--int8"
    else:
        print("Unsupported precision")
        exit()
    return preciosn_flag

def preparePPMImages(image_folder,ppm_folder):
    if not os.path.exists(image_folder):
        print("Image folder does not exist")
        exit()
    if os.path.exists(ppm_folder):
        return
        #shutil.rmtree(image_folder
    os.makedirs(ppm_folder)
    files = glob.glob(image_folder+'*')
    for im_file in files:
        filename, fileext = os.path.splitext(im_file)
        i = Image.open(im_file)
        image_out = ppm_folder+os.path.basename(filename)+'.ppm'
        i.save(image_out)
