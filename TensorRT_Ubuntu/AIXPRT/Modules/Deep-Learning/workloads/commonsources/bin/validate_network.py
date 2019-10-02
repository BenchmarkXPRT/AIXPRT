import numpy as np
import sys
import os
import csv
import random
import shutil
import subprocess
from shutil import copyfile

modelPrefix = "alex"
toolkit = "default"
aarch = "gpu"
precision = "fp32"
#numerfo categories
catCount = 1000
#number of samples for each category
catSampleCount = 5
#validation file
validationPath = "../data/val.txt"
validation_set_root_folder = '../data/'

# Get the first argument as the model name
# Check if the argument is passed for the model wprefix
if (len(sys.argv) > 1):
        modelPrefix = sys.argv[1]
if (len(sys.argv) > 2):
        precision = sys.argv[2]
if (len(sys.argv) > 3):
        toolkit = sys.argv[3]
if (len(sys.argv) > 4):
        aarch = sys.argv[4]

#### Functions ###############3
newDict = {}
def read_validation_set():
        with open(validationPath,'r') as f:
                for line in f:
                        splitLine = line.split()
                        newDict[splitLine[0]] = ",".join(splitLine[1:])

def csv_writer_single_image(data, path):
        with open(path, "a") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(data)

def csv_writer_single_line(data, path):
        nl = '\n'
        with open(path, "a") as text_file:
                text_file.write(data+nl)



#Function: Randolmly pick 5000 samples from coldBootArray and find the accuracy
def findAccuracyFromColdBoot(resArray, modelPrefix, target_count):
    nTopAccuracyCount = 0
    nTop5AccuracyCount = 0
    nRows = resArray.shape[0]
    for k in range(1, target_count+1):
        randomIndex = np.random.random_integers(0, nRows-1)
        nTopAccuracyCount += resArray[randomIndex, 0]
        nTop5AccuracyCount += resArray[randomIndex, 1]

    #Write the accuracy from this round as one entry
    validation_write = [nTopAccuracyCount/target_count, nTop5AccuracyCount/target_count]
    csv_writer_single_image(validation_write,'../../'+modelPrefix+'validation/result/result/random_validation_%s_details.csv'%modelPrefix)
    return nTopAccuracyCount/target_count, nTop5AccuracyCount/target_count

def runBootStrapFor10kTimes(dataArray, modelPrefix, nTimes, target_count=5000):
    grandArray = np.zeros((nTimes, 2))
    for k in range(0, nTimes):
            top1, top5 = findAccuracyFromColdBoot(dataArray, modelPrefix, target_count)
            #print("Top 1 , 5 = ", top1, top5)
            grandArray[k, 0] = top1
            grandArray[k, 1] = top5

    top1 = np.round(np.average(grandArray[:,0]), 4)
    top1_std = np.round(np.std(grandArray[:,0]), 4)
    top5 = np.round(np.average(grandArray[:,1]), 4)
    top5_std = np.round(np.std(grandArray[:,1]), 4)

    return top1, top1_std, top5, top5_std

# Load the predicted top5 data from '../result/result/validation_alex_details.csv'
def loadPredictedDataFromFile(pFileName, numRecords):
    coldBootArray = np.zeros((numRecords, 2))

    file_index = 0
    #Open the file and read each line
    fHandle = open(pFileName, "r")
    for fLine in fHandle:
        # remove '\n' at the end
        fLine = fLine[:-2]
        #Check for blank lines
        if len(fLine.strip()) == 0 :
            continue;
        # Parse the line with comma sep
        row_list = fLine.split(',')
        int_list = row_list[1:]
        _filename = row_list[0]
        #print("%s %s %s %s" %(_filename, int(newDict[_filename]), int_list, int_list[0]))
        if (int(newDict[_filename]) == int(int_list[0])):
            coldBootArray[file_index, 0] = 1
        if (newDict[_filename] in int_list):
            coldBootArray[file_index, 1] = 1
        file_index += 1

    fHandle.close()
    return coldBootArray

import result_dnn_api

trt_version = subprocess.getoutput("dpkg -l | grep 'Meta package of TensorRT' | awk '{print $3}'")

## writeAccuracyResultsToAPI(accType, topResult, Std, iterCount, model, ext)
def writeAccuracyResultsToAPI(accType, topResult, Std, iterCount, model, ext):
    # Add the result API code here
    resInImgPerSec = topResult
    forwardTime = Std
    workLoadName = result_dnn_api.returnWorkloadName(model+'validation')
    inputString = result_dnn_api.returnInputMap(toolkit, "ILSVRC 2012", "Accuracy", aarch, workLoadName,trt_version,precision,iterCount)
    resultsString = result_dnn_api.returnAccuracyResults(accType, topResult, Std, iterCount, model+'net', precision)
    result_dnn_api.writeResultsToAPI(workLoadName, inputString, resultsString)

def reportTop1Top5Accuracy(modelPrefix, prec):
    #Reading Validation file
    read_validation_set()

    predResultsFile = '../../'+modelPrefix+'validation/result/result/validation_'+modelPrefix+'_'+prec+'_details.csv'
    # Load the predicted top5 data from '../result/result/validation_alex_details.csv'

    inputRecordCount = 5000
    cbArray = loadPredictedDataFromFile(predResultsFile, inputRecordCount)
    print("Top 1 , 5 = %s %s" %(np.round(np.sum(cbArray[:,0])/inputRecordCount, 4), np.round(np.sum(cbArray[:,1])/inputRecordCount, 4)))
    t1 = np.round(np.sum(cbArray[:,0])/inputRecordCount, 4)
    t5 = np.round(np.sum(cbArray[:,1])/inputRecordCount, 4)
    t1_s = 0
    t5_s = 0
    #t1, t1_s, t5, t5_s = runBootStrapFor10kTimes(cbArray, modelPrefix, 10000)
    #print("Top 1 Mean = %s Std = %s with %s points" %(t1, t1_s, 10000))
    #print("Top 5 Mean = %s Std = %s with %s points" %(t5, t5_s, 10000))
    writeAccuracyResultsToAPI(1, t1*100, t1_s, 5000, modelPrefix, precision)
    writeAccuracyResultsToAPI(5, t5*100, t5_s, 5000, modelPrefix, precision)
    return t1, t1_s, t5, t5_s
#### Functions ###############3


#Reading Validation file
#read_validation_set()



# Test case for calculating accuracy & variance
reportTop1Top5Accuracy(modelPrefix, prec=precision)
