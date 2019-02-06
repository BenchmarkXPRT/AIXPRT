#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
import HTML
import os

#from urllib import pathname2url


def writeHtmlFile(input_json_file , runtype):

    with open(input_json_file) as json_data:
        d = json.load(json_data) #read json from file
        output_csv_file = input_json_file + '.csv'
        HTMLFILE = input_json_file + '.html'
        html_file = open(HTMLFILE, 'w')

        with open(output_csv_file, 'w') as f:
            for module_name in d['Result']:

                # Add Test information
                html_file.write('<header><h3>Test Information:</h3></header>')
                #create HTML Table for Overall information
                overall_info_table_data = [['Module', module_name]]
                f.write('Module' + ',' + module_name + '\n')
                for attribute, value in d['Result'][module_name]['System Info'].items():
                    overall_info_table_data.append([attribute, value])
                    f.write(str(attribute) + ',' + str(value) + '\n')
                overall_info_table = HTML.Table(overall_info_table_data)
                html_file.write(str(overall_info_table))
                #Create HTML Table for workloads results
                html_file.write('<header><h3>Results:</h3></header>')
                if (runtype == "performance"):
                    t = HTML.Table(header_row=['Workload Name', 'Hardware', 'Precision','Iterations','Label',
                    'System Latency','System Latency Units','System Throughput','System Throughput Units'])
                    workloads = d['Result'][module_name]['Workloads']
                    for wd in workloads:
                        ranOnHardware ="-"
                        ranOnPrecision = "-"
                        iteration = "-"
                        if not (wd.get('workload run information',None)==None):
                            if not (wd['workload run information'].get('architecture')==None):
                                ranOnHardware = str(wd['workload run information']['architecture']).lower()
                            if not (wd['workload run information'].get('precision')==None):
                                ranOnPrecision = str(wd['workload run information']['precision']).lower()

                        for result in wd['results']:
                            workloadName =   str(wd['workloadName'])
                            label =  result['label']
                            systemLatency = str(result["system_latency"])
                            systemLatencyUnits = result["system_latency_units"]
                            systemthroughput = str(result["system_throughput"])
                            systemThroughputUnits = result["system_throughput_units"]

                            line = workloadName +','+ ranOnHardware + ',' + ranOnPrecision + ',' + iteration + ',' + label+ ','+ systemLatency + ',' + systemLatencyUnits + ',' + systemthroughput + ',' + systemThroughputUnits
                            t.rows.append([workloadName,ranOnHardware,ranOnPrecision,iteration,label,systemLatency,systemLatencyUnits,systemthroughput,systemThroughputUnits])
                            f.write(line + '\n')
                else:
                    t = HTML.Table(header_row=['Workload Name', 'Hardware', 'Precision','Iterations','Label',
                    'Accuracy','Accuracy Units'])
                    workloads = d['Result'][module_name]['Workloads']
                    for wd in workloads:
                        ranOnHardware ="-"
                        ranOnPrecision = "-"
                        iteration = "-"
                        if not (wd.get('workload run information',None)==None):
                            if not (wd['workload run information'].get('architecture')==None):
                                ranOnHardware = str(wd['workload run information']['architecture']).lower()
                            if not (wd['workload run information'].get('precision')==None):
                                ranOnPrecision = str(wd['workload run information']['precision']).lower()

                        for result in wd['results']:
                            workloadName =   str(wd['workloadName'])
                            label =  result['label']
                            iteration = str(iteration)
                            accuracy = str(result["accuracy"])
                            accuracyUnits = result["accuracy_units"]

                            line = workloadName +','+ ranOnHardware + ',' + ranOnPrecision + ',' + iteration + ',' + label+ ','+ accuracy + ',' + accuracyUnits
                            t.rows.append([workloadName,ranOnHardware,ranOnPrecision,iteration,label,accuracy,accuracyUnits])
                            f.write(line + '\n')
            html_file.write(str(t))
            notes = d['Result'][module_name]['notes']
            html_file.write('<header><h2>NOTES:</h2></header>')
            for note in notes:
                html_file.write(note)
