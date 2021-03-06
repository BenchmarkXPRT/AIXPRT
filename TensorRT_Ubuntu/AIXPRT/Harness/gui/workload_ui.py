# Scripts creates and shows the UI as the workload writes results for each image.
# NOTE: Demo only supported with Resnet-50 workload with BatchSize=1 , iterations = 1

import cv2
import numpy as np
import os
import json
import sys
sys.path.insert(1, os.path.join(os.environ['APP_HOME'], 'Harness'))
import constants
import time
import utils
import platform

# constants
FONT_SIMPLE = cv2.FONT_HERSHEY_TRIPLEX
WHITE_COLOR = (255,255,255)
RED_COLOR = (10,49,236)
BLACK_COLOR = (0,0,0)
BLUE_COLOR = (242,61,21)
MEDIUM_FONT = 0.5
LARGE_FONT = 0.6
SYS_INFO_PANET_START = 140
BENCHMARK_INFO_PANET_START = 230
RESULTS_PANET_START = 440
OUTPUT_PANEL_START = 500
SPACING_SMALL = 30
SPACING_MEDIUM = 100
FONT_THICK = 2
FONT_THIN = 1
LEFT_SECTION_COLOR = WHITE_COLOR
RIGHT_SECTION_COLOR = (204,205,208)
workloadThroughput = []

def showWorkloadUI(workloadName,workloadDir,hardware,precision):

    cpuName = utils.getCpuName().strip()
    if platform.system() == "Windows":
        import pythoncom
        pythoncom.CoInitialize()
    gpuName = utils.getGpuName()

    imagesInfered = 0
    #  create a black image of the below size (H x W)
    window = np.zeros((680,1000,3), np.uint8)
    # partition the images into 2 sections by filling one part with pale red (b,g,r)(178,178,255)
    window[:,0:500] = LEFT_SECTION_COLOR
    window[:,500:1000] = RIGHT_SECTION_COLOR # which color to fill the second part ?
    # AIXPRT Logo location
    ly0, ly1, lx0, lx1 = [0, 100, 120, 348]
    logoPath = os.path.join(os.environ['APP_HOME'],"Harness","gui","ui_assets","AIXPRT-logo-MD.PNG")
    if os.path.exists(logoPath):
        logo=cv2.imread(logoPath)
        resized_logo=cv2.resize(logo,(lx1-lx0,ly1-ly0))
        window[ly0:ly1,lx0:lx1]=resized_logo

    # putText (image ,text ,, fontStyle ,fontSize ,color,fontThickness )
    # System Info panel
    cv2.putText(window,"Image Classification Test Demo",(70,90), FONT_SIMPLE, LARGE_FONT,BLACK_COLOR,FONT_THIN)
    cv2.putText(window,"SYSTEM INFORMATION:",(10,SYS_INFO_PANET_START), FONT_SIMPLE, LARGE_FONT,RED_COLOR,FONT_THIN)
    cv2.putText(window,"CPU:",(10,SYS_INFO_PANET_START+SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
    cv2.putText(window,cpuName,(70,SYS_INFO_PANET_START+SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
    cv2.putText(window,"GPU:",(10,SYS_INFO_PANET_START+2*SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
    cv2.putText(window,gpuName[0].strip(),(70,SYS_INFO_PANET_START+2*SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)

    # Benchmark info panel
    cv2.putText(window,"DEMO TEST INFORMATION:",(10,BENCHMARK_INFO_PANET_START), FONT_SIMPLE, LARGE_FONT,RED_COLOR,FONT_THIN)
    cv2.putText(window,"Model : "+workloadName,(10,BENCHMARK_INFO_PANET_START+SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
    cv2.putText(window,"Use Case : Image Classification",(10,BENCHMARK_INFO_PANET_START+2*SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
    cv2.putText(window,"Precision : "+precision.upper(),(10,BENCHMARK_INFO_PANET_START+3*SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
    cv2.putText(window,"Batch Size : 1",(10,BENCHMARK_INFO_PANET_START+4*SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
    cv2.putText(window,"Target hardware : "+hardware.upper(),(10,BENCHMARK_INFO_PANET_START+5*SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
    

    # Benchmark Result
    cv2.putText(window,"DEMO TEST RESULT FOR DEMO PURPOSES ONLY:",(10,RESULTS_PANET_START), FONT_SIMPLE, LARGE_FONT,RED_COLOR,FONT_THIN)

    cv2.putText(window,"Developed by BenchmarkXPRT Development Community.",(2,630), FONT_SIMPLE, 0.4,BLACK_COLOR,FONT_THIN)
    cv2.putText(window,"For more information visit ",(2,650), FONT_SIMPLE, 0.4,BLACK_COLOR,FONT_THIN)
    cv2.putText(window,"https://www.principledtechnologies.com/benchmarkxprt/aixprt/",(2,670), FONT_SIMPLE, 0.4,BLUE_COLOR,FONT_THIN)

    labelsFile = open(os.path.join(os.environ['APP_HOME'],"Harness","labels.txt"))
    labels=labelsFile.readlines()
    labelsFile.close()
    category1 = ""
    category2 = ""
    category3 = ""
    category4 = ""
    category5 = ""
    while True :
        # time.sleep(1)
        # poll the workload  to get info to display
        resultPath = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",workloadDir,"result",workloadName+".json")
        if os.path.exists(resultPath):
            # Check if a results json file is availabel and readable.
            try:
                with open(resultPath,'r') as data_file:
                    data = json.load(data_file)
                    #  show the new results generated as avaialble. NOTE : Workload pick input images with random funtion.
                    # There is a possibility of showing the same image again
                    # since the workload might infere the same image mutiple time.
                    if(len(data["Result"]["results"]) > imagesInfered):
                        result = data["Result"]["results"][imagesInfered]
                        framework = data["Result"]["workload run information"]["framework"]
                        cv2.putText(window,"Framework : "+framework,(10,BENCHMARK_INFO_PANET_START+6*SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
                        imagesInfered += 1
                        for image , output in result["additional info"][0]["output"].items():
                            workloadThroughput.append(result["system_throughput"])
                            throughput = sum(workloadThroughput) / len(workloadThroughput) 
                            latency = result["additional info"][0]["99_percentile_time"]
                            # making sure that workloads are asking to show only jpg format and not anyother like bmp
                            imageName = os.path.splitext(image)[0]+".jpg"
                            imagePath = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","input_images",imageName)
                            if(os.path.exists(imagePath)):
                                if(workloadName == "ResNet-50"):
                                    if(len(output)>0):
                                        category1 = labels[int(output[0])+1]
                                        category2 = labels[int(output[2])+1]
                                        category3 = labels[int(output[4])+1]
                                        category4 = labels[int(output[6])+1]
                                        category5 = labels[int(output[8])+1]
                                    # clear feild before updating
                                window[:,600:1300] = RIGHT_SECTION_COLOR
                                window[RESULTS_PANET_START+10:520,0:500] = LEFT_SECTION_COLOR

                                cv2.putText(window,"TEST IMAGE",(600,50), FONT_SIMPLE, LARGE_FONT,BLACK_COLOR,FONT_THIN)
                                cv2.putText(window,"TOP 5 Inference Predictions:",(600,OUTPUT_PANEL_START), FONT_SIMPLE, LARGE_FONT,RED_COLOR,FONT_THIN)

                                image=cv2.imread(imagePath)
                                height, width, channels = image.shape
                                # Image co-ordinates
                                y0, y1, x0, x1 = [100, 400, 600, 900]
                                resized_image=cv2.resize(image,(x1-x0,y1-y0))
                                window[y0:y1,x0:x1]=resized_image


                                # Goes into results panel
                                cv2.putText(window,"Throughput: "+str('%.2f'%(throughput))+" images/second",(10,RESULTS_PANET_START+SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
                                cv2.putText(window,"Latency: "+str('%.2f'%(latency))+" milliseconds",(10,RESULTS_PANET_START+2*SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
                                # Inference output panel
                                cv2.putText(window,category1.strip(),(600,OUTPUT_PANEL_START+SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
                                cv2.putText(window,category2.strip(),(600,OUTPUT_PANEL_START+2*SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
                                cv2.putText(window,category3.strip(),(600,OUTPUT_PANEL_START+3*SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
                                cv2.putText(window,category4.strip(),(600,OUTPUT_PANEL_START+4*SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
                                cv2.putText(window,category5.strip(),(600,OUTPUT_PANEL_START+5*SPACING_SMALL), FONT_SIMPLE, MEDIUM_FONT,BLACK_COLOR,FONT_THIN)
                                cv2.imshow("AIXPRT",window)
                                cv2.waitKey(2000)
            except ValueError:  # includes simplejson.decoder.JSONDecodeError
                print("Checking updates to display")

        else:
            cv2.putText(window,"Preparing workload...",(600,300), FONT_SIMPLE, LARGE_FONT,WHITE_COLOR,FONT_THICK)
            cv2.namedWindow("AIXPRT", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("AIXPRT",window)
            cv2.waitKey(2000)
        # once the number of images showen on UI is equal to number of input images provided , end the UI
        if(imagesInfered == (len(os.listdir(os.path.join(constants.INSTALLED_MODULES_PATH,"Deep-Learning","packages","input_images"))))):
            break
    #  Below code can clear the UI to show results.
    # time.sleep(2)
    # while(True):
    #     if os.environ['RUN_RESULT'] is not None:
    #         window[:,600:1300] = RIGHT_SECTION_COLOR
    #         cv2.imshow("AIXPRT",window)
    #         cv2.waitKey(2000)
    #         break
