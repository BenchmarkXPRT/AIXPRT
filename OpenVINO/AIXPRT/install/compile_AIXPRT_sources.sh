#!/bin/bash

# Copyright (c) 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

NUM_ARGS=$#
BATCH_LIST=(1 2 4 8 16 32 64 128) # Modify to include batch sizes required to run on GPU
usage() {
    echo "Usage:"
    echo -e "\t compile_AIXPRT_sources.sh </path/to/AIXPRT>"
    echo -e "\t compile_AIXPRT_sources.sh -dir </path/to/AIXPRT>"
    echo -e "\t compile_AIXPRT_sources.sh -h [PRINT HELP MESSAGE]"
    echo -e "Assumes:\n\t--- you have installed openVINO in /opt/intel/computer_vision_sdk/"
    echo -e "\t--- you have cloned AIXPRT"
    exit 1
}

error() {
    local code="${3:-1}"
    if [[ -n "$2" ]];then
        echo "Error on or near line $1: $2; exiting with status ${code}"
    else
        echo "Error on or near line $1; exiting with status ${code}"
    fi
    exit "${code}"
}
trap 'error ${LINENO}' ERR

#============================= parse command line options and set AIXPRT install directory ===============================================
key="$1"
case $key in
    -h | -help | --help)
     usage
    ;;    
    -dir | --dir)
    flag=$key
    AIXPRT_DIR="$2"
    ;;
esac


if [[ ${NUM_ARGS} -lt 1 ]]; then
  AIXPRT_DIR="$( dirname $PWD )"
  if [ -d ${AIXPRT_DIR} ] && [ -d "${AIXPRT_DIR}/Modules" ]; then
     echo -e "\e[1;33mAIXPRT install directory not provided. Installing to ${AIXPRT_DIR}\e[0m"
  else
     echo -e "\e[1;31mCannot determine AIXPRT distribution directory. Please pass the source directory with -dir </path/to/AIXPRT/folder>.\e[0m"
     usage
  fi
fi

if [[ ${NUM_ARGS} == 1 ]]; then
   AIXPRT_DIR=$key
  if  [ ! -d ${AIXPRT_DIR} ]; then   
     echo -e "\e[1;31m\n\nTarget folder ${key} does not exist.\n\e[0m"
     usage
  fi

elif [[ "${NUM_ARGS}" -ge "2" ]] && [ "$flag" == "-dir" ] && [ ! -d ${AIXPRT_DIR} ]; then

   echo -e "\e[1;31m\n\nTarget folder ${AIXPRT_DIR} does not exist.\n\e[0m"
   usage

elif [[ "${NUM_ARGS}" -ge "2" ]] && [ "${1}" != "-dir" ]; then
   AIXPRT_DIR=${1}

   echo -e "\e[1;31mProvided install directory ${AIXPRT_DIR} does not exist.\n\e[0m"
   usage

elif [[ ${NUM_ARGS} -gt 2 ]]; then
   echo -e "\e[1;31m\nCannot parse inputs\n\e[0m"
   usage
fi  

# Last sanity check
if [ ! -d "${AIXPRT_DIR}/Modules" ]; then
   echo -e "\e[1;31mProvided install directory ${AIXPRT_DIR} does not have Modules subdir.\e[0m"
   usage
fi

RUN_AGAIN="Then run the script again\n\n"
DASHES="\n\n==================================================\n\n"
PYTHON_BINARY=python3
PIP_BINARY=pip3
CUR_PATH=$PWD

OPENVINO_DIR="/opt/intel/computer_vision_sdk/deployment_tools/"
OPENVINO_CV_DEP_DIR="/opt/intel/computer_vision_sdk/install_dependencies"

if [ ! -e $OPENVINO_DIR ]; then
   echo -e "\e[1;31m\nOpenVINO install directory ${OPENVINO_DIR} does not exists.\n\e[0m"
   echo -e "\e[1;0mPlease install OpenVINO distribution in /opt/intel\n\e[0m"
   
   exit 1
fi

AIXPRT_MODELS="${AIXPRT_DIR}/Modules/Deep-Learning/packages/models"
AIXPRT_PLUGIN="${AIXPRT_DIR}/Modules/Deep-Learning/packages/plugin/"
AIXPRT_BIN="${AIXPRT_DIR}/Modules/Deep-Learning/workloads/commonsources/bin/"
AIXPRT_SOURCES="${AIXPRT_DIR}/Modules/Deep-Learning/workloads/commonsources/bin/src/"

#========================= Install dependencies =======================================================

# Step 1. Install Dependencies

printf "${DASHES}"
printf "Installing dependencies"
printf "${DASHES}"

if [[ -f /etc/lsb-release ]]; then
    DISTRO="ubuntu"
else
    echo -e "\e[1;31m\n AIXPRT: Ubuntu is the only operating system supported.\e[0m"
    exit 1
fi

printf "Run sudo -E apt -y install build-essential python3-pip virtualenv cmake libpng12-dev libcairo2-dev libpango1.0-dev libglib2.0-dev libgtk2.0-dev libswscale-dev libavcodec-dev libavformat-dev libgstreamer1.0-0 gstreamer1.0-plugins-base python-imaging\n"
sudo -E apt update
sudo -E apt -y install build-essential python3-pip virtualenv cmake libpng12-dev libcairo2-dev libpango1.0-dev libglib2.0-dev libgtk2.0-dev libswscale-dev libavcodec-dev libavformat-dev libgstreamer1.0-0 gstreamer1.0-plugins-base python-imaging

if ! command -v $PYTHON_BINARY &>/dev/null; then
    printf "\n\nPython 3.5 (x64) or higher is not installed. It is required to run Model Optimizer, please install it. ${RUN_AGAIN}"
    exit 1
fi
sudo -E $PIP_BINARY install pyyaml requests numpy

printf "${DASHES}"
printf "Installing cv sdk dependencies\n\n"
cd ${OPENVINO_CV_DEP_DIR}

sudo -E ./install_cv_sdk_dependencies.sh
cd ${CUR_PATH}

#========================= Setup Model Optimizer =======================================================
# Step 2. Enter OpenVINO environment and Configure Model Optimizer

printf "${DASHES}"
printf "Setting OpenVINO environment and Configuring Model Optimizer"
printf "${DASHES}"

if [[ -z "${INTEL_CVSDK_DIR}" ]]; then
            printf "\n\nINTEL_CVSDK_DIR environment variable is not set. Trying to run ./setvars.sh to set it. \n"

    if [ -e "${OPENVINO_DIR}/inference_engine/bin/setvars.sh" ]; then # for Intel Deep Learning Deployment Toolkit package
        SETVARS_PATH="${OPENVINO_DIR}/inference_engine/bin/setvars.sh"
    elif [ -e "${OPENVINO_DIR}/../bin/setupvars.sh" ]; then # for Intel CV SDK package
        SETVARS_PATH="${OPENVINO_DIR}/../bin/setupvars.sh"
    elif [ -e "${OPENVINO_DIR}/../setupvars.sh" ]; then # for Intel GO SDK package
        SETVARS_PATH="${OPENVINO_DIR}/../setupvars.sh"
    else
        printf "Error: setvars.sh is not found\n"
    fi
    if ! source ${SETVARS_PATH} ; then
        printf "Unable to run ./setvars.sh. Please check its presence. ${RUN_AGAIN}"
        exit 1
    fi
fi

OPENVINO_IE_DIR="${INTEL_CVSDK_DIR}/deployment_tools/inference_engine/"
OPENVINO_MO_DIR="${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/"
MO_PATH="${OPENVINO_MO_DIR}/mo.py"

printf "${DASHES}"
printf "Install Model Optimizer dependencies"
cd "${OPENVINO_MO_DIR}/install_prerequisites"
bash install_prerequisites.sh "caffe"

cd ${CUR_PATH}

#========================= Download and Convert pre-trained models ===========================================
# Step 3. Download and Convert pretrained Models

printf "${DASHES}"
printf "Downloading and Converting pretrained models"
printf "${DASHES}"

MODEL_NAMES=("mobilenet-ssd" "resnet-50")

PRECISION_LIST=("FP16" "FP32")
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp

for idx in "${!MODEL_NAMES[@]}"
  do
    MODEL_DIR="$CUR_PATH/models"
    
    MODEL_NAME=${MODEL_NAMES[idx]}

	if [ $MODEL_NAME == "mobilenet-ssd" ]
	     then
		MODEL_PATH="${TAR_DIR}/frozen_inference_graph.pb"
		MODEL_DEST="ssd_mobilenet"

		IR_PATH="$CUR_PATH/${MODEL_NAME}"
		if [ -d ${IR_PATH} ];then rm -r ${IR_PATH}; fi;

		for PRECISION in "${PRECISION_LIST[@]}"
		   do
			precision=${PRECISION,,}
			printf "Run $OPENVINO_DIR/model_downloader/downloader.py --name \"${MODEL_NAME}\" --output_dir \"${MODEL_DIR}\"\n\n"
			MODEL_PATH="$MODEL_DIR/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel"
			$PYTHON_BINARY "$OPENVINO_DIR/model_downloader/downloader.py" --name "${MODEL_NAME}" --output_dir "${MODEL_DIR}"
			MEAN_VALUES="data[127.5,127.5,127.5]"
			SCALE_VALUES="data[127.50223128904757]"
			MODEL_DEST="ssd_mobilenet"
			INPUT_SHAPE="[1,3,300,300]"

			IR_MODEL_XML=${IR_PATH}/${MODEL_DEST}".xml" # Name of generated IR
			IR_MODEL_BIN=${IR_PATH}/${MODEL_DEST}".bin"

			IR_MODEL_AIXPRT_XML=${IR_PATH}/${MODEL_DEST}"_${precision}.xml" # For renaming generated IR
			IR_MODEL_AIXPRT_BIN=${IR_PATH}/${MODEL_DEST}"_${precision}.bin"


			printf "Run $PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --output_dir $IR_PATH --data_type FP16 --input_shape $INPUT_SHAPE --mean_values $MEAN_VALUES --scale_values $SCALE_VALUES\n\n"
			$PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --output_dir $IR_PATH --model_name $MODEL_DEST --data_type "${PRECISION}" --input_shape $INPUT_SHAPE --mean_values $MEAN_VALUES --scale_values $SCALE_VALUES

			#------- rename generated IRs
			mv ${IR_MODEL_XML} ${IR_MODEL_AIXPRT_XML}
			mv ${IR_MODEL_BIN} ${IR_MODEL_AIXPRT_BIN}
		done

    elif [ $MODEL_NAME == "resnet-50" ]
           then

              MODEL_DEST="resnet-50"
	      printf "Run $OPENVINO_DIR/model_downloader/downloader.py --name \"${MODEL_NAME}\" --output_dir \"${MODEL_DIR}\"\n\n"
	      MODEL_PATH="$MODEL_DIR/classification/resnet/v1/50/caffe/resnet-50.caffemodel"
              $PYTHON_BINARY "$OPENVINO_DIR/model_downloader/downloader.py" --name "${MODEL_NAME}" --output_dir "${MODEL_DIR}"
	      MEAN_VALUES="data[0.0,0.0,0.0]"
	      SCALE_VALUES="data[1.0]"
	      MODEL_DEST="resnet-50"
	      INPUT_SHAPE="[1,3,224,224]"
               for PRECISION in ${PRECISION_LIST[@]}
                  do
	                precision=${PRECISION,,}
			IR_MODEL_XML=${IR_PATH}/${MODEL_DEST}".xml"
			IR_MODEL_BIN=${IR_PATH}/${MODEL_DEST}".bin"
			IR_MODEL_AIXPRT_XML=${IR_PATH}/${MODEL_DEST}"_${precision}.xml"
			IR_MODEL_AIXPRT_BIN=${IR_PATH}/${MODEL_DEST}"_${precision}.bin"


			printf "Run $PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --output_dir $IR_PATH --data_type FP16 --input_shape $INPUT_SHAPE --mean_values $MEAN_VALUES --scale_values $SCALE_VALUES\n\n"
			$PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --output_dir $IR_PATH --model_name $MODEL_DEST --data_type "${PRECISION}" --input_shape $INPUT_SHAPE --mean_values $MEAN_VALUES --scale_values $SCALE_VALUES

			#------- rename generated IRs
			mv ${IR_MODEL_XML} ${IR_MODEL_AIXPRT_XML}
			mv ${IR_MODEL_BIN} ${IR_MODEL_AIXPRT_BIN}
               done
    fi

	
	if [ ! -e $IR_PATH ]; then
	    printf "\n\nTarget folder ${IR_PATH} does not exists"
	    exit 1
	else
	    cp -r ${IR_PATH}/* ${AIXPRT_MODELS}/${MODEL_DEST}/
	    rm -r ${IR_PATH}
	fi

done

#========================= Build Classification and Detection binaries ================================================================================
# Step 4. Build samples
printf "${DASHES}"
printf "Building AIXPRT sources ${AIXPRT_SOURCES}"
printf "${DASHES}"

if ! command -v cmake &>/dev/null; then
    echo -e "\e[0;32m\nCMAKE is not installed. It is required to build Inference Engine samples. Please install it and run again.\n\e[0m"
    exit 1
fi

# copy sources here
AIXPRT_TMP_SRC="$CUR_PATH/aixprt_sources/"
BUILD_DIR="$CUR_PATH/aixprt_compiled"
COMPILED_APP_DIR="${BUILD_DIR}/intel64/Release"
if [ -d "${AIXPRT_TMP_SRC}" ]; then rm -Rf $AIXPRT_TMP_SRC; fi
if [ -d "${BUILD_DIR}" ]; then rm -Rf $BUILD_DIR; fi

mkdir ${AIXPRT_TMP_SRC}
cp -r ${AIXPRT_SOURCES}/* ${AIXPRT_TMP_SRC}
cp -r ${OPENVINO_DIR}/inference_engine/samples/thirdparty ${AIXPRT_TMP_SRC}
cp -r ${OPENVINO_DIR}/inference_engine/samples/common ${AIXPRT_TMP_SRC}

if [ ! -e "${BUILD_DIR}/intel64/Release/image_classification" ]; then
    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}
    cmake -DCMAKE_BUILD_TYPE=Release ${AIXPRT_TMP_SRC}
    make -j8
else
    printf "\n\nTarget folder ${BUILD_DIR} already exists. Skipping samples building."
    printf "If you want to rebuild samples, remove the entire ${BUILD_DIR} folder. ${RUN_AGAIN}"
fi

if [ ! -e $COMPILED_APP_DIR ]; then
   printf "\n\nTarget folder ${COMPILED_APP_DIR} does not exists.\n"
   exit 1
else
   cp ${COMPILED_APP_DIR}/image_classification ${AIXPRT_BIN}
   cp ${COMPILED_APP_DIR}/image_classification_async ${AIXPRT_BIN}
   cp ${COMPILED_APP_DIR}/object_detection_ssd ${AIXPRT_BIN}
   cp ${COMPILED_APP_DIR}/object_detection_ssd_async ${AIXPRT_BIN}

   # these are found after compiling AIXPRT
   cp ${COMPILED_APP_DIR}/lib/libformat_reader.so ${AIXPRT_PLUGIN}/
   cp ${COMPILED_APP_DIR}/lib/libcpu_extension.so ${AIXPRT_PLUGIN}/

fi

#========================= Copy libraries ================================================================================
# Step 5. Finally copy plugins folder

#printf ${DASHES}
#printf "Copying OpenVINO Libraries"

PLUGIN_DIR="$CUR_PATH/plugin"
if [ -d "${PLUGIN_DIR}" ]; then rm -Rf $PLUGIN_DIR; fi
mkdir ${PLUGIN_DIR}

OPERATING_SYSTEM="ubuntu_16.04"
OPENVINO_IE_DIR="/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/"

cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libclDNN64.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/external/omp/lib/libiomp5.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/external/mkltiny_lnx/lib/libmkl_tiny_omp.so $PLUGIN_DIR

cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libcpu_extension_avx2.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libclDNNPlugin.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libHeteroPlugin.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libinference_engine.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libMKLDNNPlugin.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libmyriadPlugin.so $PLUGIN_DIR

# HDDL libraries
cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libHDDLPlugin.so $PLUGIN_DIR
find /opt/intel/computer_vision_sdk/inference_engine/external/hddl/lib/ -type f -name 'lib*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

# openCV
find /opt/intel/computer_vision_sdk/opencv/lib/ -type f -name 'libopencv_core*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

find /opt/intel/computer_vision_sdk/opencv/lib/ -type f -name 'libopencv_imgcodecs*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

find /opt/intel/computer_vision_sdk/opencv/lib/ -type f -name 'libopencv_imgproc*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

cp -r ${PLUGIN_DIR}/* ${AIXPRT_PLUGIN}/

#printf ${DASHES}
#======================= Remove temp directories =================================================
# Step 6. Post-install Clean-up

cd ${CUR_PATH}
aixprt_compiled="${CUR_PATH}/aixprt_compiled"
aixprt_sources="${CUR_PATH}/aixprt_sources"
#plugin="${CUR_PATH}/plugin"
models="${CUR_PATH}/models"

if [ -d "${aixprt_compiled}" ]; then rm -Rf ${aixprt_compiled}; fi
if [ -d "${aixprt_sources}" ]; then rm -Rf ${aixprt_sources}; fi
#if [ -d "${plugin}" ]; then rm -Rf ${plugin}; fi
if [ -d "${models}" ]; then rm -Rf ${models}; fi

printf ${DASHES}
echo -e "\e[1;32mSetup completed successfully.\e[0m"
printf ${DASHES}
exit 1
