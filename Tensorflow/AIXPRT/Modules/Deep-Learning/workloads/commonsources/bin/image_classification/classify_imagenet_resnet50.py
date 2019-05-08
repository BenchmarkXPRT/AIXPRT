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

import time
import numpy as np
import os
import argparse
import json
import imagenet_preprocessing
import tensorflow as tf
import tensorflow.tools.graph_transforms as graph_transforms
import csv
import random

INPUTS = 'input'
OUTPUTS = 'predict'
OPTIMIZATION = 'strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'
_WARMUP_NUM_LOOPS = 10
RESNET_IMAGE_SIZE = 224

def read_inputs():
  tfConfigParams = {}
  workload_details = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",FLAGS.workload_dir,'workload_details.json')
  with open(workload_details) as data_file:
    data = json.load(data_file)
  if not (data.get("requested_config",None)==None):
    requested_config = data["requested_config"]
    if not (requested_config.get("framework_config",None)==None):
      if not (requested_config["framework_config"].get("enableConfig",None)==None):
        if requested_config["framework_config"]["enableConfig"]:
          return requested_config["framework_config"]
  return tfConfigParams

def preprocess_image(file_name, output_height=RESNET_IMAGE_SIZE, output_width=RESNET_IMAGE_SIZE,
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

def batch_from_image(file_name, batch_size, batch_data, output_height=RESNET_IMAGE_SIZE, output_width=RESNET_IMAGE_SIZE,
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
  img_path = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages", file_name)

  image_array = preprocess_image(
      img_path, output_height, output_width, num_channels)


  batch_data.append(image_array)
  return batch_data


def run_inference(tfConfigParams, images,image_path):
  model_dir=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","models")
  if FLAGS.precision=='int8':
    INPUTS = 'input'
    OUTPUTS = 'predict'
  if FLAGS.precision=='fp32':
    INPUTS = 'input'
    OUTPUTS = 'resnet_v1_50/SpatialSqueeze'
  device = "/"+FLAGS.aarch+":0" if len(FLAGS.aarch) > 1 else "/cpu:0"
  # open the device to run on
  with tf.device(device):
    timing_csv_file = open(FLAGS.csv_file_path,"a")
    # prepare the config
    with tf.Graph().as_default() as graph:
      config = tf.ConfigProto()
      for key , value in tfConfigParams.items():
        if(key == "inter_op_parallelism_threads"):
          config.inter_op_parallelism_threads = value
        if(key == "intra_op_parallelism_threads"):
          config.intra_op_parallelism_threads = value
        if(key == "allow_soft_placement"):
          config.allow_soft_placement = value
          # open a tensorflow session and load the graph
      with tf.Session(config=config) as sess:
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_dir+"/"+FLAGS.frozen_graph, 'rb') as input_file:
          input_graph_content = input_file.read()
          graph_def.ParseFromString(input_graph_content)
        output_graph = graph_transforms.TransformGraph(graph_def,[INPUTS], [OUTPUTS], [OPTIMIZATION])
        sess.graph.as_default()
        tf.import_graph_def(output_graph, name='')

        # Definite input and output Tensors for detection_graph
        input_tensor = graph.get_tensor_by_name(INPUTS+':0')
        output_tensor = graph.get_tensor_by_name(OUTPUTS+':0')
        tf.global_variables_initializer()

        #start inference
        tf.logging.info("Starting Warmup cycle")
        for _ in range(_WARMUP_NUM_LOOPS):
          predicts = sess.run([output_tensor], feed_dict={input_tensor: images})
        for iter in range(FLAGS.iterations):
          tf.logging.info("Starting timing.")
          tstart = time.time()
          predicts = sess.run([output_tensor], feed_dict={input_tensor: images})
          tend = time.time()
          print("Time:",(tend - tstart))
          predictions = np.squeeze(predicts)
          tf.logging.info("Timing loop done!")
          if(os.environ["DEMO"] == "True"):
              imageName = os.path.basename(image_path)
              predictionsList = predictions.argsort()[-5:][::-1]
              scoreList = []
              for node_id in predictionsList:
                  scoreList.append(predictions[node_id])
              row = [str(tstart),str(tend),imageName,str(predictionsList[0]),str(scoreList[0]),str(predictionsList[1]),str(scoreList[1]),str(predictionsList[2]),str(scoreList[2]),str(predictionsList[3]),str(scoreList[3]),str(predictionsList[4]),str(scoreList[4])]
              with open(FLAGS.csv_file_path, 'a') as csvFile:
                  writer = csv.writer(csvFile)
                  writer.writerow(row)
              csvFile.close()
          else:
              timing_csv_buffer_data = np.hstack((tstart, tend))
              np.savetxt(timing_csv_file, timing_csv_buffer_data[np.newaxis], delimiter=",", fmt='%f')
  return

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
       '--instance',
       type=str,
       default='',
       help=''
   )

  parser.add_argument(
      '--workload_dir',
      type=str,
      default='',
      help=''
  )
  parser.add_argument(
       '--csv_file_path',
       type=str,
       default='',
       help=''
   )
  parser.add_argument(
       '--precision',
       type=str,
       default='',
       help=''
   )
  FLAGS, unparsed = parser.parse_known_args()

# Read user flags
tf_config_params = read_inputs()

# prepare image input
files_dir = 'input_images'
data_dir=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages", files_dir)
files = os.listdir(data_dir)
included_extenstions = ['jpg', 'jpeg']
file_names = [fn for fn in os.listdir(data_dir)
  if any(fn.endswith(ext) for ext in included_extenstions)]
batch_data=[]
random.shuffle(file_names)
files = file_names[:FLAGS.batch_size]
cur_size=0
images_map = {}
while cur_size < FLAGS.batch_size:
    for f in files:
        image_path = data_dir + '/' + f
        if cur_size == FLAGS.batch_size:
            break
        cur_size +=1
        batch_data = batch_from_image(image_path, FLAGS.batch_size, batch_data)
        images_map
print(np.shape(batch_data))
# run inference
run_inference(tf_config_params,batch_data,image_path)
