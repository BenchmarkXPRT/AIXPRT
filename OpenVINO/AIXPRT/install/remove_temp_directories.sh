#!/bin/bash
CUR_PATH=$PWD
aixprt_compiled="$PWD/aixprt_compiled"
aixprt_sources="$PWD/aixprt_sources"
plugin="$PWD/plugin"
resnet50="$PWD/resnet-50"
models="$PWD/models"
resnet50_model="$PWD/resnet-50_model"

if [ -d "${aixprt_compiled}" ]; then rm -Rf ${aixprt_compiled}; fi
if [ -d "${aixprt_sources}" ]; then rm -Rf ${aixprt_sources}; fi
if [ -d "${plugin}" ]; then rm -Rf ${plugin}; fi
if [ -d "${resnet50}" ]; then rm -Rf ${resnet50}; fi
if [ -d "${resnet50_model}" ]; then rm -Rf ${resnet50_model}; fi
if [ -d "${models}" ]; then rm -Rf ${models}; fi
