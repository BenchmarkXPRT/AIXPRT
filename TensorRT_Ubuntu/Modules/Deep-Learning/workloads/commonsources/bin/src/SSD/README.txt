The sampleUffSSD example is based on the following paper, SSD: Single Shot MultiBox
Detector (https://arxiv.org/abs/1512.02325). The SSD network performs the
task of object detection and localization in a single forward pass of the network.
The tensorflow SSD network was trained on the InceptionV2 architecture using
the MSCOCO dataset.

The sample makes use of TensorRT plugins to run the SSD network. To use these
plugins the TensorFlow graph needs to be preprocessed.

Steps to generate UFF file:
    0. Make sure you have the UFF converter installed. For installation instructions, see:
        https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/#python and click on the 'TensorRT Python API' link.

    1. Get the pre-trained Tensorflow model (ssd_inception_v2_coco) from:
        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

    2. Call the UFF converter with the preprocessing flag set (-p [config_file]).
        The config.py script specifies the preprocessing operations necessary for SSD TF graph.
        It must be copied to the working directory for the file to be imported properly.

        'convert-to-uff tensorflow --input-file frozen_inference_graph.pb -O NMS -p config.py'

This script saves the converted .uff file in the same directory as the input with
the name frozen_inference_graph.pb.uff. Copy this converted .uff file to the
data directory as sample_ssd.uff <TensorRT Install>/data/ssd/sample_ssd.uff

The sample also requires a labels .txt file with a list of all labels used to
train the model. Current example for this network is <TensorRT Install>/data/ssd/ssd_coco_labels.txt

Steps to run the network:
    1. To run the network in FP32 mode, ./sample_uff_ssd
    2. To run the network in INT8 mode, ./sample_uff_ssd --int8

To run the network in INT8 mode, refer to BatchStreamPPM.h for details on how
calibration can be performed. Currently we require a file (list.txt) with
a list of all PPM images for calibration in the <TensorRT Install>/data/ssd/ folder.
The PPM images to be used for calibration can also reside in the same folder.

NOTE - There might be some precision loss when running the network in INT8
mode causing some objects to go undetected. Our general observation is that
>500 images is a good number for calibration purposes.
