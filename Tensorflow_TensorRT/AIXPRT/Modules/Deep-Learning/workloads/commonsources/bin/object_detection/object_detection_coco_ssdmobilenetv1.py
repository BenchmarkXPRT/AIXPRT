# ============================================================================
# Copyright (C) 2018 BenchmarkXPRT Development Community
# Licensed under the BENCHMARKXPRT DEVELOPMENT COMMUNITY MEMBER LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License by contacting Principled Technologies, Inc.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing grants and
# restrictions under the License.
#============================================================================



# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import time
import argparse
import PIL
import subprocess

#from distutils.version import StrictVersion
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
import ops as utils_ops

#if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
#  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


_LOG_FILE = "log_mobilenet_v1_ssd.txt"
_WARMUP_NUM_LOOPS = 10

sys.path.insert(1, os.path.join(os.environ['APP_HOME'], 'Harness'))
import resultsapi
import utils

model_dir=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","models")
saved_model_dir=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","models","saved_model")
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = os.path.join(saved_model_dir, 'frozen_inference_graph.pb')  #'model/ssd_mobilenet_v1_coco_2018_graph.pb'

from graph_utils import force_nms_cpu as f_force_nms_cpu
from graph_utils import remove_assert as f_remove_assert

INPUT_NAME='image_tensor'
BOXES_NAME='detection_boxes'
CLASSES_NAME='detection_classes'
SCORES_NAME='detection_scores'
NUM_DETECTIONS_NAME='num_detections'
PIPELINE_CONFIG_NAME='pipeline.config'
CHECKPOINT_PREFIX='model.ckpt'

def optimize_graph(
    frozen_graph_path,
    optimized_path,
    force_nms_cpu=False,
    remove_assert=False,
    use_trt=True,
    precision_mode='FP32',
    minimum_segment_size=2,
    batch_size=1):

    modifiers = []
    if force_nms_cpu:
        modifiers.append(f_force_nms_cpu)
    if remove_assert:
        modifiers.append(f_remove_assert)


    #tf_config = tf.ConfigProto()
    #tf_config.gpu_options.allow_growth = True

    with tf.Session() as tf_sess:
        frozen_graph = tf.GraphDef()
        with open(frozen_graph_path, 'rb') as f:
            frozen_graph.ParseFromString(f.read())

        for m in modifiers:
            frozen_graph = m(frozen_graph)

        if use_trt:
            frozen_graph = trt.create_inference_graph(
                input_graph_def=frozen_graph,
                outputs=[NUM_DETECTIONS_NAME, BOXES_NAME, CLASSES_NAME, SCORES_NAME],
                precision_mode=precision_mode,
                minimum_segment_size=minimum_segment_size,
                max_batch_size=batch_size
            )
        #subprocess.call(['mkdir', '-p', os.path.dirname("./data")])
        with open(optimized_path, 'wb') as f:
            f.write(frozen_graph.SerializeToString())


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def batch_from_image(file_name, batch_size, batch_data, output_height=224, output_width=224,
                     num_channels=3):
  """Produce a batch of data from the passed image file.

  Args:
    file_name: string, path to file containing a JPEG image
    batch_size: int, the size of the desired batch of data
    output_height: int, final height of data
    output_width: int, final width of data
    num_channels: int, depth of input data

  Returns:
    Float array representing copies of the image with shape
      [batch_size, output_height, output_width, num_channels]
  """

  img = np.array(PIL.Image.open(file_name).resize((300, 300))).astype(np.float) / 128 - 1
  if(img.size!=270000):
    img =  np.stack((img,)*3, -1)
  batch_data.append(img)
  return batch_data

def log_stats(log_buffer, timings, batch_size):
  """Write stats to the passed log_buffer.

  Args:
    graph_name: string, name of the graph to be used for reporting.
    log_buffer: filehandle, log file opened for appending.
    timings: list of floats, times produced for multiple runs that will be
      used for statistic calculation
    batch_size: int, number of examples per batch
  """
  times = np.array(timings)
  steps = len(times)
  speeds = batch_size / times
  time_mean = np.mean(times)
  time_med = np.median(times)
  #time_99th = np.percentile(times, 99)
  #time_99th_uncertainty = np.abs(np.percentile(times[0::2], 99) -
  #                               np.percentile(times[1::2], 99))
  speed_mean = np.mean(speeds)
  speed_med = np.median(speeds)
  speed_uncertainty = np.std(speeds, ddof=1) / np.sqrt(float(steps))
  speed_jitter = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))

  msg = ("\n==========================\n"
         "\t batchsize %d, steps %d\n"
         "  fps \tmedian: %.1f, \tmean: %.1f"  # pylint: disable=line-too-long
         "  latency \tmedian: %.5f, \tmean: %.5f\n"  # pylint: disable=line-too-long
        ) % ( batch_size, steps,
             speed_med, speed_mean,
             time_med, time_mean)
  print(msg)
  time_mean = time_mean*1000
  labelstr = "Batch "+ str(FLAGS.batch_size)
  additional_info_details = {}
  additional_info_details["concurrent_instances"] = 1
  additional_info_details["total_requests"] = FLAGS.iterations
  accelerator_lib_details = {}
  if (FLAGS.aarch.lower()=="cpu"):
      accelerator_lib_details["cpu_accelerator_lib"] = ""
  else:
      accelerator_lib_details["gpu_accelerator_lib"] = ""
  workloadInput={
	"architecture":FLAGS.aarch,
	"precision":FLAGS.prec,
    "accelerator_lib": [accelerator_lib_details],
    "framework":utils.getTensorflowInfo()
	}
  results=[
	{
	"label":labelstr,
	"system_latency":time_mean,
	"system_latency_units":"milliseconds",
	"system_throughput":speed_mean,
	"system_throughput_units":"imgs/sec",
	"additional info":[additional_info_details]
	}
	]
  print(workloadInput)
  print(results)

  resultsapi.createResultJson("SSD-MobileNet-v1", workloadInput, results)
  log_buffer.write(msg)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'input_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4) ]

def run_inference_for_single_image(image, frozen_graph_path):
  timings = []
  # load frozen graph from file
  with open(frozen_graph_path, 'rb') as f:
    frozen_graph = tf.GraphDef()
    frozen_graph.ParseFromString(f.read())

  tf_config = tf.ConfigProto()
  tf_config.gpu_options.allow_growth=False

  with tf.Graph().as_default() as tf_graph:
    with tf.Session(config=tf_config) as sess:

      # import frozen graph and get relevant tensors
      tf.import_graph_def(frozen_graph, name='')
      tf_input = tf_graph.get_tensor_by_name(INPUT_NAME + ':0')
      tf_boxes = tf_graph.get_tensor_by_name(BOXES_NAME + ':0')
      tf_classes = tf_graph.get_tensor_by_name(CLASSES_NAME + ':0')
      tf_scores = tf_graph.get_tensor_by_name(SCORES_NAME + ':0')
      tf_num_detections = tf_graph.get_tensor_by_name(NUM_DETECTIONS_NAME + ':0')
      # Run inference
      tf.logging.info("Starting execution")
      tf.logging.info("Starting Warmup cycle")
      for _ in range(_WARMUP_NUM_LOOPS):
        boxes, classes, scores, num_detections = sess.run([ tf_boxes, tf_classes, tf_scores, tf_num_detections ],
                                feed_dict={tf_input: image})
      iteration = 1
      while(iteration<=FLAGS.iterations):
            print("Processing Iteration:",iteration)
            tf.logging.info("Starting timing.")
            tstart = time.time()
            boxes, classes, scores, num_detections = sess.run([ tf_boxes, tf_classes, tf_scores, tf_num_detections ],
                             feed_dict={tf_input: image})
            timings.append(time.time() - tstart)
            iteration+=1
            print("Time:",timings)
            tf.logging.info("Timing loop done!")

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict = {}
      output_dict['num_detections'] = num_detections
      output_dict['detection_classes'] = classes
      output_dict['detection_boxes'] = boxes
      output_dict['detection_scores'] = scores
      #if 'detection_masks' in output_dict:
        #output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict, timings

def ssd_config_override(box_score_threshold):
    return '''
      model {
       ssd {
        post_processing {
         batch_non_max_suppression {
          score_threshold: %f
         }
        }
        feature_extractor {
         override_base_feature_extractor_hyperparams: true
        }
       }
      }
    ''' % box_score_threshold

def main(_):
  device = "/"+FLAGS.aarch+":0" if len(FLAGS.aarch) > 1 else "/gpu:0"
  with tf.device(device):
        files_dir = PATH_TO_TEST_IMAGES_DIR
        data_dir=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages", files_dir)
        files = os.listdir(data_dir)
        log_buffer = open(os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",'ssdmobilenetv1','result','output','ssdmobilenetv1'+'_'+'_batch'+str(FLAGS.batch_size)+'_'+FLAGS.aarch+'.txt'),"a")

        batch_data=[]
        files = files[:FLAGS.batch_size]

        if os.path.isdir(saved_model_dir):
            print('Existing configuration found, removing it.')
            subprocess.call(['rm', '-rf', saved_model_dir])
        nms_score_threshold=0.3
        config_override = ssd_config_override(nms_score_threshold)
        input_pipeline_config_path = os.path.join(model_dir, PIPELINE_CONFIG_NAME)
        input_checkpoint_prefix = os.path.join(model_dir, CHECKPOINT_PREFIX)

        subprocess.call(['python', '-m', 'object_detection.export_inference_graph',
        '--input_type', 'image_tensor',
        '--input_shape', '%d,-1,-1,3' % FLAGS.batch_size,
        '--pipeline_config_path', input_pipeline_config_path,
        '--output_directory', saved_model_dir,
        '--trained_checkpoint_prefix', input_checkpoint_prefix,
        '--config_override', config_override])

        optimized_graph_path="TRT_"+FLAGS.prec+"_batch"+str(FLAGS.batch_size)
        optimize_graph(
          PATH_TO_FROZEN_GRAPH,
          optimized_graph_path,
          force_nms_cpu=False,
          remove_assert=True,
          use_trt=True,
          precision_mode=FLAGS.prec,
          minimum_segment_size=15,
          batch_size=FLAGS.batch_size
        )
        cur_size=0
        while cur_size < FLAGS.batch_size:
          for f in files:
              if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                  image_path = data_dir + '/' + f
              if cur_size == FLAGS.batch_size:
                  break 
              batch_data = batch_from_image(image_path, FLAGS.batch_size, batch_data)
              cur_size +=1
              print(image_path)
              print(cur_size)
        output_dict, timing = run_inference_for_single_image(batch_data, optimized_graph_path)

        log_stats(log_buffer, timing, FLAGS.batch_size)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--frozen_graph',
      type=str,
      default='',
      help='Frozen Graph '
  )
  parser.add_argument(
    "--batch_size", "-bs", type=int, default=1,
    help="[default: %(default)s] Batch size for inference. If an "
    "image file is passed, it will be copied batch_size times to "
    "imitate a batch.",
    metavar="<BS>"
    )

  parser.add_argument(
    "--iterations", "-i", type=int, default=100,
    help="[default: %(default)s] iterations for inference.",
    metavar="<BS>"
    )

  parser.add_argument(
      '--aarch',
      type=str,
      default='',
      help=''
  )
  parser.add_argument(
      '--prec',
      type=str,
      default='',
      help=''
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
