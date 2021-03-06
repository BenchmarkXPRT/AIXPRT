#!/bin/bash
DLDT_BUILD=$1
AIXPRT_DIR=$2
INSTALL_DIR=/opt/intel/ov_custom/

if [ ! -d ${INSTALL_DIR} ]; then
    sudo mkdir -p ${INSTALL_DIR}
fi

# =============== Create necessary directories ===================
# 1. deployment_tools directory
sudo mkdir -p ${INSTALL_DIR}/deployment_tools/inference_engine
sudo mkdir -p ${INSTALL_DIR}/deployment_tools/model_optimizer
sudo mkdir -p ${INSTALL_DIR}/deployment_tools/tools
sudo mkdir -p ${INSTALL_DIR}/deployment_tools/demo
sudo cp -r ${DLDT_BUILD}/model-optimizer/. ${INSTALL_DIR}/deployment_tools/model_optimizer                             # mo_optimizer

# 1.1 inference_engine subdirectory
sudo mkdir -p ${INSTALL_DIR}/deployment_tools/inference_engine/external
sudo mkdir -p ${INSTALL_DIR}/deployment_tools/inference_engine/src
sudo mkdir -p ${INSTALL_DIR}/deployment_tools/inference_engine/lib/intel64

sudo cp -r ${DLDT_BUILD}/inference-engine/build/share ${INSTALL_DIR}/deployment_tools/inference_engine/                             # Share

#sudo cp -r /opt/intel/openvino/deployment_tools/inference_engine/share ${INSTALL_DIR}/deployment_tools/inference_engine/

sudo cp -r ${DLDT_BUILD}/inference-engine/include ${INSTALL_DIR}/deployment_tools/inference_engine/                                 # include
sudo cp -r ${DLDT_BUILD}/inference-engine/src/extension ${INSTALL_DIR}/deployment_tools/inference_engine/src/                       # src
sudo cp -r ${DLDT_BUILD}/inference-engine/samples ${INSTALL_DIR}/deployment_tools/inference_engine/                                 # samples

sudo cp -r ${DLDT_BUILD}/inference-engine/bin/intel64/Release/lib/*.so ${INSTALL_DIR}/deployment_tools/inference_engine/lib/intel64/        # lib
sudo cp -r ${DLDT_BUILD}/inference-engine/bin/intel64/Release/lib/MvN* ${INSTALL_DIR}/deployment_tools/inference_engine/lib/intel64/        # lib
sudo cp -r ${DLDT_BUILD}/inference-engine/bin/intel64/Release/lib/plugins.xml ${INSTALL_DIR}/deployment_tools/inference_engine/lib/intel64/        # lib

if [ -d "${DLDT_BUILD}/inference-engine/temp/vpu/hddl" ]; then
    sudo cp -r ${DLDT_BUILD}/inference-engine/temp/vpu/hddl ${INSTALL_DIR}/deployment_tools/inference_engine/external/                  # hddl
fi
# sudo cp -r ${DLDT_BUILD}/inference-engine/temp/mkltiny_lnx* ${INSTALL_DIR}/deployment_tools/inference_engine/external/mkltiny_lnx   # mkl
if [ -d "${DLDT_BUILD}/inference-engine/temp/tbb" ]; then
    sudo cp -r ${DLDT_BUILD}/inference-engine/temp/tbb ${INSTALL_DIR}/deployment_tools/inference_engine/external/ #tbb
fi
if [ -d "${DLDT_BUILD}/inference-engine/temp/omp" ]; then
    sudo cp -r ${DLDT_BUILD}/inference-engine/temp/omp ${INSTALL_DIR}/deployment_tools/inference_engine/external/  # omp
fi    


# Create link to deployment_tools/inference_engine
cd ${INSTALL_DIR}
if [ -d "/opt/intel/ov_custom/inference_engine" ] 
then
    sudo rm -rf /opt/intel/ov_custom/inference_engine
fi
sudo ln -s deployment_tools/inference_engine inference_engine

# 1.2 tools subdirectory

if [ -d "${INSTALL_DIR}/deployment_tools/tools/open_model_zoo" ]; then
    sudo rm -rf ${INSTALL_DIR}/deployment_tools/tools/open_model_zoo
fi

if [ -d ${DLDT_BUILD}/open_model_zoo ]; then 
# pull and revert to Oct 28th commit e249977dc5d9979beda7da97a20f25795c2f82ea to ensure consistency
    cd ${DLDT_BUILD}/open_model_zoo && git pull
else
    cd ${DLDT_BUILD}
    git clone https://github.com/opencv/open_model_zoo.git
fi

sudo cp -r ${DLDT_BUILD}/open_model_zoo ${INSTALL_DIR}/deployment_tools/tools/

# 2. OpenCV
sudo cp -r ${DLDT_BUILD}/inference-engine/temp/opencv* ${INSTALL_DIR}/opencv # opencv

# 3. Python
sudo mkdir -p ${INSTALL_DIR}/python
sudo cp -r ${DLDT_BUILD}/inference-engine/bin/intel64/Release/lib/python_api/* ${INSTALL_DIR}/python/ # python
sudo cp -r ${DLDT_BUILD}/inference-engine/ie_bridges/python/sample ${INSTALL_DIR}/deployment_tools/inference_engine/samples/python_samples

# 4. Bin
if [ ! -d "${INSTALL_DIR}/bin" ]; then 
    sudo mkdir -p ${INSTALL_DIR}/bin/
fi

find $DLDT_BUILD/inference-engine/bin/intel64/Release/lib -type f -name '*.mvcmd*' -exec sudo cp '{}' ${AIXPRT_DIR}/Modules/Deep-Learning/packages/plugin ';'

DNAME=$( dirname "${DLDT_BUILD}" )
sudo cp ${AIXPRT_DIR}/install/setupvars.sh ${INSTALL_DIR}/bin/


