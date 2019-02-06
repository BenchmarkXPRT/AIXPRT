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

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Simple image classification.

Run image classification with Resnet50 trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import imghdr
import imagenet_preprocessing
import os

FLAGS = None
_WARMUP_NUM_LOOPS = 10


sys.path.insert(1, os.path.join(os.environ['APP_HOME'], 'Harness'))
import resultsapi
import utils

# def id_to_string(self, node_id):
#     if node_id not in self.node_lookup:
#       return ''
#     return self.node_lookup[node_id]


def preprocess_image(file_name, output_height=224, output_width=224,
                     num_channels=3):
  """Run standard ImageNet preprocessing on the passed image file.

  Args:
    file_name: string, path to file containing a JPEG image
    output_height: int, final height of image
    output_width: int, final width of image
    num_channels: int, depth of input image

  Returns:
    Float array representing processed image with shape
      [output_height, output_width, num_channels]

  Raises:
    ValueError: if image is not a JPEG.
  """
  # if imghdr.what(file_name) != "jpeg":
  #   raise ValueError("At this time, only JPEG images are supported. "
  #                    "Please try another image.")
  with tf.device('/cpu:0'):
      image_buffer = tf.read_file(file_name)
      normalized = imagenet_preprocessing.preprocess_image(
          image_buffer=image_buffer,
          bbox=None,
          output_height=output_height,
          output_width=output_width,
          num_channels=num_channels,
          is_training=False)

  with tf.Session() as sess:
    result = sess.run([normalized])

  return result[0]

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
  img_path =  os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages",file_name)
  image_array = preprocess_image(
      img_path, output_height, output_width, num_channels)

  image_array = image_array.astype(np.uint8)
  batch_data.append(image_array)
  return batch_data

def create_graph(model_dir, graph_file):
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      model_dir, graph_file), 'rb') as f:
    print("graph", os.path.join(
      model_dir, graph_file))
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    #_ = tf.import_graph_def(graph_def, name='')
    return graph_def

def log_stats(graph_name, log_buffer, timings, batch_size):
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
  speed_mean = np.mean(speeds)
  speed_med = np.median(speeds)

  msg = ("\n==========================\n"
         "\t batchsize %d, steps %d\n"
         "  fps \tmedian: %.1f, \tmean: %.1f"
         "  latency \tmedian: %.5f, \tmean: %.5f\n"
        ) % ( batch_size, steps,
             speed_med, speed_mean,
             time_med, time_mean)
  time_mean = time_mean*1000
  labelstr = "Batch "+ str(FLAGS.batch_size)
  additional_info_details = {}
  additional_info_details["total_requests"] = FLAGS.iterations
  additional_info_details["concurrent_instances"] = 1
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
  resultsapi.createResultJson("ResNet-50", workloadInput, results)

  log_buffer.write(msg)

def run_inference(img, frozen_graph_def):
  """Runs inference on an image.

  Args:
    image: Image file name.
    frozen_graph_def: model graph

  Returns:
    timings: performance timing
  """
  tf.logging.info("Starting execution")


  #inp, preds = tf.import_graph_def(frozen_graph_def,  return_elements = ['input:0', 'resnet_v1_50/SpatialSqueeze:0'])
  inp, preds = tf.import_graph_def(frozen_graph_def,  return_elements = ['input:0', 'resnet_v1_50/SpatialSqueeze:0'])
  #tf_config = tf.ConfigProto()
  #tf_config.gpu_options.allow_growth=True
  timings = []
  with tf.Session(graph=inp.graph) as sess:
    # Some useful tensors:
    # 'resnet_v1_50/SpatialSqueeze:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'input:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the 'resnet_v1_50/SpatialSqueeze' tensor by feeding the image_data as input to the graph.

    tf.logging.info("Starting Warmup cycle")
    for _ in range(_WARMUP_NUM_LOOPS):
      predictions = preds.eval(feed_dict={inp: img})
    iteration = 1
    tf.logging.info("Starting timing.")

    while(iteration<=FLAGS.iterations):
      print("Processing Iteration:",iteration)
      tstart = time.time()
      predictions = preds.eval(feed_dict={inp: img})
      timings.append(time.time() - tstart)
      print("Time:",timings)
      iteration+=1
    tf.logging.info("Timing loop done!")
  return timings

def get_frozen_graph(
    model,
    use_trt=False,
    use_dynamic_op=False,
    precision='fp32',
    batch_size=8,
    minimum_segment_size=2,
    calib_data_dir=None,
    num_calib_inputs=None,
    use_synthetic=False,
    cache=False,
    download_dir='./data'):
    """Retreives a frozen GraphDef from model definitions in classification.py and applies TF-TRT

    model: str, the model name (see NETS table in classification.py)
    use_trt: bool, if true, use TensorRT
    precision: str, floating point precision (fp32, fp16, or int8)
    batch_size: int, batch size for TensorRT optimizations
    returns: tensorflow.GraphDef, the TensorRT compatible frozen graph
    """
    num_nodes = {}
    times = {}

    # Load from pb file if frozen graph was already created and cached
    if cache:
        # Graph must match the model, TRT mode, precision, and batch size
        prebuilt_graph_path = "graphs/frozen_graph_%s_%d_%s_%d.pb" % (model, int(use_trt), precision, batch_size)
        if os.path.isfile(prebuilt_graph_path):
            print('Loading cached frozen graph from \'%s\'' % prebuilt_graph_path)
            start_time = time.time()
            with tf.gfile.GFile(prebuilt_graph_path, "rb") as f:
                frozen_graph = tf.GraphDef()
                frozen_graph.ParseFromString(f.read())
            times['loading_frozen_graph'] = time.time() - start_time
            num_nodes['loaded_frozen_graph'] = len(frozen_graph.node)
            num_nodes['trt_only'] = len([1 for n in frozen_graph.node if str(n.op)=='TRTEngineOp'])
            return frozen_graph, num_nodes, times

    # Build graph and load weights
    #frozen_graph = build_classification_graph(model, download_dir)
    model_dir=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","models")
    frozen_graph = create_graph(model_dir, FLAGS.frozen_graph)
    num_nodes['native_tf'] = len(frozen_graph.node)

    # Convert to TensorRT graph
    if use_trt:
        start_time = time.time()
        frozen_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=['resnet_v1_50/SpatialSqueeze:0'],
            max_batch_size=batch_size,
            max_workspace_size_bytes=(4096<<20)-1000,
            precision_mode=precision,
            minimum_segment_size=minimum_segment_size,
            is_dynamic_op=use_dynamic_op
        )
        times['trt_conversion'] = time.time() - start_time
        num_nodes['tftrt_total'] = len(frozen_graph.node)
        num_nodes['trt_only'] = len([1 for n in frozen_graph.node if str(n.op)=='TRTEngineOp'])

        if precision == 'int8':
            calib_graph = frozen_graph
            # INT8 calibration step
            print('Calibrating INT8...')
            batch_data=[]
            files_dir = 'input_images'
            data_dir=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages", files_dir)
            files = os.listdir(data_dir)
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                   image_path = files_dir + '/' + f
                batch_data = batch_from_image(image_path, FLAGS.batch_size, batch_data)
                print(image_path)
            run_inference(batch_data, calib_graph)
            #run(calib_graph, model, calib_data_dir, batch_size,
            #    num_calib_inputs // batch_size, 0, False)
            frozen_graph = trt.calib_graph_to_infer_graph(calib_graph)
            del calib_graph
            print('INT8 graph created.')

    # Cache graph to avoid long conversions each time
    if cache:
        if not os.path.exists(os.path.dirname(prebuilt_graph_path)):
            try:
                os.makedirs(os.path.dirname(prebuilt_graph_path))
            except Exception as e:
                raise e
        start_time = time.time()
        with tf.gfile.GFile(prebuilt_graph_path, "wb") as f:
            f.write(frozen_graph.SerializeToString())
        times['saving_frozen_graph'] = time.time() - start_time

    return frozen_graph, num_nodes, times

def main(_):
  #ss
  device = "/"+FLAGS.aarch+":0" if len(FLAGS.aarch) > 1 else "/gpu:0"
  with tf.device(device):
      #model_dir=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","models")

      if FLAGS.frozen_graph:
        #frozen_graph_def = create_graph(model_dir, FLAGS.frozen_graph)
        frozen_graph_def, num_nodes, times  = get_frozen_graph(
        model="resnet_v1_50",
        use_trt=True,
        use_dynamic_op=False,
        precision=FLAGS.prec,
        batch_size=FLAGS.batch_size,
        minimum_segment_size=2,
        calib_data_dir=None,
        num_calib_inputs=None,
        use_synthetic=False,
        cache=True,
        download_dir='./data')
      else:
        raise ValueError(
            "Either a Frozen Graph file or a SavedModel must be provided.")

        # search for files in 'images' dir
      files_dir = 'input_images'
      data_dir=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages", files_dir)
      files = os.listdir(data_dir)
      print ("Batch Size:", FLAGS.batch_size)
      log_buffer = open(os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",'resnet50_2016','result','output','resnet50'+'_batch'+str(FLAGS.batch_size)+'_'+FLAGS.aarch+'.txt'),"a")
      batch_data=[]
      files = files[:FLAGS.batch_size]
      times = []
      cur_size=0
      while cur_size < FLAGS.batch_size:
        for f in files:
          if f.lower().endswith(('.png', '.jpg', '.jpeg')):
               image_path = files_dir + '/' + f
          if cur_size == FLAGS.batch_size:
               break 
          batch_data = batch_from_image(image_path, FLAGS.batch_size, batch_data)
          cur_size +=1
          print(image_path)
          print(cur_size)
      timing = run_inference(batch_data, frozen_graph_def)
      times.append(timing)
      log_stats(frozen_graph_def, log_buffer, times, FLAGS.batch_size)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--frozen_graph',
      type=str,
      default='',
      help='Frozen Graph '
  )
  parser.add_argument(
    "--batch_size", "-bs", type=int, default=128,
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
