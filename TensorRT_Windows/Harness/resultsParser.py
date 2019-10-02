#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
import os
import csv
import statistics
import inspect
import codecs


def writeToCSV(resultJsonPathArray,FolderName,AIXPRT_dir):
    medianMap = {}
    workloadsInResults = {}
    for resultJsonPath in resultJsonPathArray:
        with open(resultJsonPath,'rb') as data_file:
            reader = codecs.getreader("utf-8")
            x = json.load(reader(data_file))
        csvFileName = resultJsonPath +".csv"
        f = csv.writer(open(csvFileName, "w", newline=''))
        systemInfo = x["Result"]["Deep-Learning"]["System Info"]
        # Write system Info
        f.writerow(["TEST INFORMATION"])
        f.writerow([""])
        for key , value in systemInfo.items():
            f.writerow([key , value])
        f.writerow([""])
        f.writerow(["TEST RESULTS"])
        f.writerow([""])
        f.writerow(["Workload Name","Batch Size","Hardware","Precision","Total Requests","Instances","System Throughput","System Throughput Units","System Latency(50th percentile)","System Latency(90th percentile)","System Latency(95th percentile)","System Latency(99th percentile)","System Latency Units"])
        workloads = x["Result"]["Deep-Learning"]["Workloads"]
        for workload in workloads:
            hardware = "-"
            precision = "-"
            total_request = "-"
            batchSize = "-"
            instances = "-"
            workloadName = workload["workloadName"]
            if not (workload.get('workload run information',None)==None):
                if not (workload['workload run information'].get('architecture')==None):
                    hardware = str(workload['workload run information']['architecture']).lower()
                if not (workload['workload run information'].get('precision')==None):
                    precision = str(workload['workload run information']['precision']).lower()
            for result in workload["results"]:
                fiftyLatency  =  "-"
                nintyLatency = "-"
                nintyFiveLatency = "-"
                nintyNineLatency = "-"
                systemLatencyUnits = "-"
                batchSize = result["label"]
                systemThrougput = result["system_throughput"]
                systemThrougputUnits = result["system_throughput_units"]
                if not (len(result['additional info'][0])==0):
                    total_request = result['additional info'][0]['total_requests']
                    instances = result['additional info'][0]['concurrent_instances']
                    fiftyLatency = result['additional info'][0]['50_percentile_time']
                    nintyLatency = result['additional info'][0]['90_percentile_time']
                    nintyFiveLatency = result['additional info'][0]['95_percentile_time']
                    nintyNineLatency = result['additional info'][0]['99_percentile_time']
                    systemLatencyUnits = result['additional info'][0]['time_units']

                f.writerow([workloadName,batchSize,hardware,precision,total_request,instances,systemThrougput,systemThrougputUnits,fiftyLatency,nintyLatency,nintyFiveLatency,nintyNineLatency,systemLatencyUnits])
                line  = workloadName+"_"+batchSize+"_"+hardware+"_"+precision
                if not workloadName in workloadsInResults.keys():
                    workloadsInResults[workloadName] = {}
                    workloadsInResults[workloadName]["all_throughputs"] = []
                    workloadsInResults[workloadName]["all_90latency"] = []

                # nintyLatencyKey = workloadName+","+batchSize+","+hardware+","+precision

                if not line in medianMap:
                    medianMap[line] = {}
                    medianMap[line]["throughput"] = []
                    medianMap[line]["50th"] =[]
                    medianMap[line]["90th"] =[]
                    medianMap[line]["95th"] =[]
                    medianMap[line]["99th"] = []
                if isinstance(systemThrougput,float):
                    medianMap[line]["throughput"].append(systemThrougput)
                if isinstance(fiftyLatency,float):
                    medianMap[line]["50th"].append(fiftyLatency)
                if isinstance(nintyLatency,float):
                    medianMap[line]["90th"].append(nintyLatency)
                if isinstance(nintyFiveLatency,float):
                    medianMap[line]["95th"].append(nintyFiveLatency)
                if isinstance(nintyNineLatency,float):
                    medianMap[line]["99th"].append(nintyNineLatency)
                medianMap[line]["system_throughput_units"] = systemThrougputUnits
                medianMap[line]["time_units"] = systemLatencyUnits

    summaryFile = os.path.join(AIXPRT_dir,"Results",FolderName)
    csvFileName = summaryFile +"_RESULTS_SUMMARY.csv"
    summaryFile = csv.writer(open(csvFileName, "w", newline=''))
    # Parse requierd system information
    summaryFile.writerow(["SYSTEM INFORMATION :"])
    for key , value in systemInfo.items():
        if not (key == "Thread(s) per core (CPU)"):
            summaryFile.writerow([key , value])
    # Add results summary
    summaryFile.writerow([" "])
    summaryFile.writerow(["RESULT SUMMARY:"])
    for eachWorkload , allResults in workloadsInResults.items():
        for line_key , line_value in medianMap.items():
            if line_value["throughput"] :
                if(eachWorkload in line_key):
                    allResults["all_throughputs"].append(statistics.median(line_value["throughput"]))
                    allResults["all_90latency"].append(statistics.median(line_value["90th"]))
        if allResults["all_throughputs"]:
            for key , value in medianMap.items():
                if(eachWorkload in key ):
                    if medianMap[key]["throughput"]:
                        if max(allResults["all_throughputs"]) == statistics.median(medianMap[key]["throughput"]):
                            maxThoughputWorkload = key
                            throughputUnits = medianMap[key]["system_throughput_units"]
                        if min(allResults["all_90latency"]) ==  statistics.median(medianMap[key]["90th"]):
                            minLatencyWorkload = key
                            latencyUnits = medianMap[key]["time_units"]
           
            summaryFile.writerow([eachWorkload+" Maximum Inference Throughput",max(allResults["all_throughputs"]),throughputUnits, maxThoughputWorkload])
            summaryFile.writerow([eachWorkload+" Minimum  Inference Latency Per Image (90th percentile)",min(allResults["all_90latency"]),latencyUnits,minLatencyWorkload])
    summaryFile.writerow([" "])

    # Parse and add detailed results
    summaryFile.writerow(["DETAILED RESULTS (Median in case of multiple iterations):"])
    summaryFile.writerow(["Workload","Inference Throughput","Inference Throughput Units","Inference Latency(50th percentile time)","Inference Latency(90th percentile time)","Inference Latency(95th percentile time)","Inference Latency(99th percentile time)","Inference Latency Units"])
    for line_key , line_value in medianMap.items():
        if line_value["throughput"] :
            summaryFile.writerow([line_key, statistics.median(line_value["throughput"]) , "images/sec", statistics.median(line_value["50th"]) ,statistics.median(line_value["90th"]) ,statistics.median(line_value["95th"]) ,statistics.median(line_value["99th"]) , "milliseconds"])

def summarizeResults():
    #path to where this file is
    path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    # Home and harness is split from path
    AIXPRT_dir , Harness = os.path.split(path)

    resultsFolderMap = {}
    if(os.path.isdir(os.path.join(AIXPRT_dir,"Results"))):
        for resultsFolder in (os.listdir(os.path.join(AIXPRT_dir,"Results"))):
            resultsList =[]
            if(os.path.isdir(os.path.join(AIXPRT_dir,"Results",resultsFolder))):
                for fname in os.listdir(os.path.join(AIXPRT_dir,"Results",resultsFolder)):
                    if fname.endswith('.json'):
                        resultsList.append(os.path.join(AIXPRT_dir,"Results",resultsFolder,fname))
                resultsFolderMap[resultsFolder] = resultsList
    if(resultsFolderMap):
        for key , value in resultsFolderMap.items():
            writeToCSV(value,key,AIXPRT_dir)

summarizeResults()