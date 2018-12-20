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

usage() {
    echo "compile_AIXPRT_sources"
    echo "Assumes you have installed openVINO under /opt/intel/computer_vision_sdk/"
    echo "Assumes you have installed AIXPRT under ~/AIXPRT/"
    echo "-help       print help message"
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

# parse command line options
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h | -help | --help)
    usage
    ;;
    *)
    # unknown option
    ;;
esac
shift
done

RUN_AGAIN="Then run the script again\n\n"
DASHES="\n\n###################################################\n\n"
PYTHON_BINARY=python3
PIP_BINARY=pip3
CUR_PATH=$PWD

OPENVINO_DIR="/opt/intel/computer_vision_sdk/deployment_tools/"
AIXPRT_DIR="${HOME}/AIXPRT"

# make sure dirs exists
if [ ! -e $OPENVINO_DIR ]; then
   printf "\n\nTarget folder ${OPENVINO_DIR} does not exists.\n"
   exit 1
fi
if [ ! -e $AIXPRT_DIR ]; then
   printf "\n\nTarget folder ${AIXPRT_DIR} does not exists.\n"
   exit 1
fi

AIXPRT_MODELS="${AIXPRT_DIR}/Modules/Deep-Learning/packages/models"
AIXPRT_PLUGIN="${AIXPRT_DIR}/Modules/Deep-Learning/packages/plugin/"
AIXPRT_BIN="${AIXPRT_DIR}/Modules/Deep-Learning/workloads/commonsources/bin/"
AIXPRT_SOURCES="${AIXPRT_DIR}/Modules/Deep-Learning/workloads/commonsources/bin/src/"

#####################################################

# Step 1. Install Dependencies

printf "${DASHES}"
printf "\n\nDownloading the Caffe model and the prototxt"

printf "\nInstalling dependencies\n"
if [[ -f /etc/lsb-release ]]; then
    DISTRO="ubuntu"
else
    printf "\n\n AIXPRT: Ubuntu is the only operative system supported ${RUN_AGAIN}"
    exit 1
fi

printf "Run sudo -E apt -y install build-essential python3-pip virtualenv cmake libpng12-dev libcairo2-dev libpango1.0-dev libglib2.0-dev libgtk2.0-dev libswscale-dev libavcodec-dev libavformat-dev libgstreamer1.0-0 gstreamer1.0-plugins-base\n"
sudo -E apt update
sudo -E apt -y install build-essential python3-pip virtualenv cmake libpng12-dev libcairo2-dev libpango1.0-dev libglib2.0-dev libgtk2.0-dev libswscale-dev libavcodec-dev libavformat-dev libgstreamer1.0-0 gstreamer1.0-plugins-base

if ! command -v $PYTHON_BINARY &>/dev/null; then
    printf "\n\nPython 3.5 (x64) or higher is not installed. It is required to run Model Optimizer, please install it. ${RUN_AGAIN}"
    exit 1
fi
sudo -E $PIP_BINARY install pyyaml requests

####################################################
# Step 1. Configure Model Optimizer
printf "${DASHES}"
printf "Configure Model Optimizer\n\n"

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
printf "Install Model Optimizer dependencies\n\n"
cd "${OPENVINO_MO_DIR}/install_prerequisites"
bash "${OPENVINO_MO_DIR}/install_prerequisites/install_prerequisites.sh" tf

####################################################
# Step 2. Download the model

cd ${CUR_PATH}
MODELS_DIR="$CUR_PATH/models"

MODEL_NAMES=("mobilenet-ssd") #"resnet-50" )
BATCH_LIST=(1 2 4 8 16 32)
PRECISION_LIST=("FP16" "FP32")

for MODEL_NAME in "${MODEL_NAMES[@]}"
    do

	if [ $MODEL_NAME == "mobilenet-ssd" ]
	   then
		wget -P ${MODELS_DIR} http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz

		tar -xzvf "${MODELS_DIR}/ssd_mobilenet_v1_coco_2018_01_28.tar.gz" -C $MODELS_DIR

		TAR_DIR="${MODELS_DIR}/ssd_mobilenet_v1_coco_2018_01_28"
		if [ ! -d $TAR_DIR ]
		   then
			mkdir $TAR_DIR
		fi

		MODEL_PATH="${MODELS_DIR}/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb"
		MODEL_DEST="ssd_mobilenet"

		ALL_PARAMETERS="--tensorflow_use_custom_operations_config /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --output=detection_boxes,detection_scores,num_detections --tensorflow_object_detection_api_pipeline_config $TAR_DIR/pipeline.config"
		for PRECISION in "${PRECISION_LIST[@]}"
		   do
			for BATCH in "${BATCH_LIST[@]}"
			    do
			       
			       precision=${PRECISION,,}
			       OUTPUT_NAME="${MODEL_DEST}_${precision}_b${BATCH}"
			       printf "Run $PYTHON_BINARY $MO_PATH --input_model $INPUT_MODEL --output_dir $MODEL_DEST --model_name ${OUTPUT_NAME} --data_type ${precision} ${ALL_PARAMETERS} --batch ${BATCH}\n\n"
			       $PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --output_dir ${MODEL_DEST} --model_name ${OUTPUT_NAME} --data_type ${PRECISION} ${ALL_PARAMETERS} --batch ${BATCH}
			done
		done

        elif [ $MODEL_NAME == "resnet-50" ]
	   then
	      printf "Run $OPENVINO_DIR/model_downloader/downloader.py --name \"${MODEL_NAME}\" --output_dir \"${MODELS_DIR}\"\n\n"
	      MODEL_PATH="$MODELS_DIR/classification/resnet/v1/50/caffe/resnet-50.caffemodel"
              $PYTHON_BINARY "$OPENVINO_DIR/model_downloader/downloader.py" --name "${MODEL_NAME}" --output_dir "${MODELS_DIR}"
	      MEAN_VALUES="data[0.0,0.0,0.0]"
	      SCALE_VALUES="data[1.0]"
	      MODEL_DEST="resnet-50"
	      INPUT_SHAPE="[1,3,224,224]"
	      for PRECISION in ${PRECISION_LIST[@]}
		do
	          precision=${PRECISION,,}
	          OUTPUT_NAME=${MODEL_DEST}_${precision}
	          printf "Run $PYTHON_BINARY $MO_PATH --input_model $INPUT_MODEL --output_dir $MODEL_DEST --model_name ${OUTPUT_NAME} --data_type ${precision}\n\n"
	          $PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --output_dir ${MODEL_DEST} --model_name ${OUTPUT_NAME} --data_type ${PRECISION} --input_shape "[1,3,224,224]"	   
	      done
        fi

	
	if [ ! -e $MODEL_DEST ]; then
	    printf "\n\nTarget folder ${MODEL_DEST} does not exists.\n"
	    exit 1
	else
	    cp -r ${MODEL_DEST}/* ${AIXPRT_MODELS}/${MODEL_DEST}/
	    rm -r ${MODEL_DEST}
	fi

done

####################################################
# Step 4. Build samples
printf "${DASHES}"
printf "Build AIXPRT sources ${AIXPRT_SOURCES}\n\n"

if ! command -v cmake &>/dev/null; then
    printf "\n\nCMAKE is not installed. It is required to build Inference Engine samples. Please install it. ${RUN_AGAIN}"
    exit 1
fi

# copy sources here
AIXPRT_TMP_SRC="$CUR_PATH/aixprt_sources/"
BUILD_DIR="$CUR_PATH/aixprt_compiled"
COMPILED_APP_DIR="${BUILD_DIR}/intel64/Release/"
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
   cp ${COMPILED_APP_DIR}/object_detection_ssd ${AIXPRT_BIN}
fi

# Finally copy plugins folder
# configurable
PLUGIN_DIR="$CUR_PATH/plugin"
if [ -d "${PLUGIN_DIR}" ]; then rm -Rf $PLUGIN_DIR; fi
mkdir ${PLUGIN_DIR}

OPERATIVE_SYSTEM="ubuntu_16.04" # ubuntu_16.04 centos_7.4
OPENVINO_IE_DIR="/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/"

# these are found after compiling AIXPRT
cp ${COMPILED_APP_DIR}/lib/libformat_reader.so ${PLUGIN_DIR}
cp ${COMPILED_APP_DIR}/lib/libcpu_extension.so ${PLUGIN_DIR}

cp $OPENVINO_IE_DIR/lib/ubuntu_16.04/intel64/libclDNN64.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/external/omp/lib/libiomp5.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/external/mkltiny_lnx/lib/libmkl_tiny_omp.so $PLUGIN_DIR
#cp -r $OPENVINO_IE_DIR/lib/$OPERATIVE_SYSTEM/intel64/mvnc $PLUGIN_DIR
#cp -r $OPENVINO_IE_DIR/lib/$OPERATIVE_SYSTEM/intel64/arch_descriptions/ $PLUGIN_DIR
#cp -r $OPENVINO_IE_DIR/lib/$OPERATIVE_SYSTEM/intel64/cldnn_global_custom_kernels/ $PLUGIN_DIR
cp $OPENVINO_IE_DIR/lib/$OPERATIVE_SYSTEM/intel64/libcpu_extension_avx2.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/lib/$OPERATIVE_SYSTEM/intel64/libclDNNPlugin.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/lib/$OPERATIVE_SYSTEM/intel64/libdliaPlugin.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/lib/$OPERATIVE_SYSTEM/intel64/libHeteroPlugin.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/lib/$OPERATIVE_SYSTEM/intel64/libinference_engine.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/lib/$OPERATIVE_SYSTEM/intel64/libMKLDNNPlugin.so $PLUGIN_DIR
cp $OPENVINO_IE_DIR/lib/$OPERATIVE_SYSTEM/intel64/libmyriadPlugin.so $PLUGIN_DIR
# openCV
cp /opt/intel/computer_vision_sdk/opencv/lib/libopencv_core.so.4.0.0 $PLUGIN_DIR/libopencv_core.so.4.0
cp /opt/intel/computer_vision_sdk/opencv/lib/libopencv_imgcodecs.so.4.0.0 $PLUGIN_DIR/libopencv_imgcodecs.so.4.0
cp /opt/intel/computer_vision_sdk/opencv/lib/libopencv_imgproc.so.4.0.0 $PLUGIN_DIR/libopencv_imgproc.so.4.0

if [ ! -e $PLUGIN_DIR ]; then
   printf "\n\nTarget folder ${PLUGIN_DIR} does not exists.\n"
   exit 1
else
   cp -r ${PLUGIN_DIR}/* ${AIXPRT_PLUGIN/}
fi

printf "${DASHES}"

#------------------ Remove temp directories
cd ${CUR_PATH}
if [ -e "remove_temp_directories.sh" ]
   then
	chmod +x "remove_temp_directories.sh"
	printf "Removing temp folders\n\n"
	bash remove_temp_directories.sh
fi
printf "Setup completed successfully.\n\n"
exit 1

