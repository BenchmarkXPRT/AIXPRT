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
import time
import argparse
import PIL
import json
from distutils.version import StrictVersion
from PIL import Image
import tensorflow.tools.graph_transforms as graph_transforms


FLAGS = None

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


_LOG_FILE = "log_mobilenet_v1_ssd.txt"
_WARMUP_NUM_LOOPS = 10



# def create_graph(model_dir, graph_file):
#   """Creates a graph from saved GraphDef file and returns a saver."""
#   # Creates graph from saved graph_def.pb.
#   with tf.gfile.GFile(os.path.join(
#       model_dir, graph_file), 'rb') as f:
#     print("graph", os.path.join(
#       model_dir, graph_file))
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     #tf.import_graph_def(graph_def, name='')
#     return graph_def

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



PATH_TO_TEST_IMAGES_DIR = 'input_images'

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
def run_inference(image, data, model_dir, graph_file):
  timing_csv_file = open(FLAGS.csv_file_path,"a")
  detection_graph = tf.Graph()
  tfConfigParams = read_inputs()
  with detection_graph.as_default():
      config = tf.ConfigProto()
      for key , value in tfConfigParams.items():
        if(key == "inter_op_parallelism_threads"):
          config.inter_op_parallelism_threads = value
        if(key == "intra_op_parallelism_threads"):
          config.intra_op_parallelism_threads = value
        if(key == "allow_soft_placement"):
          config.allow_soft_placement = value
      timings = []
      with tf.Session(config=config) as sess:
          with tf.gfile.GFile(os.path.join(model_dir, graph_file), 'rb') as f:
            print("graph", os.path.join(model_dir, graph_file))
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
            ]:
              tensor_name = key + ':0'
              if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
            if 'detection_masks' in tensor_dict:
              # The following processing is only for single image
              detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
              detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
              # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
              real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
              detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
              detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
              detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  detection_masks, detection_boxes, image.shape[0], image.shape[1])
              detection_masks_reframed = tf.cast(
                  tf.greater(detection_masks_reframed, 0.5), tf.uint8)
              # Follow the convention by adding back the batch dimension
              tensor_dict['detection_masks'] = tf.expand_dims(
                  detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            # Run inference
            tf.logging.info("Starting execution")
            tf.logging.info("Starting Warmup cycle")
            for _ in range(_WARMUP_NUM_LOOPS):
              output_dict = sess.run(tensor_dict,
                                      feed_dict={image_tensor: image})

            tf.logging.info("Starting timing.")
            for iter in range(FLAGS.iterations):
                tstart = time.time()
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: image})
                tend = time.time()
                timings.append(tend - tstart)
                print("Time:",timings)
                tf.logging.info("Timing loop done!")

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                  output_dict['detection_masks'] = output_dict['detection_masks'][0]
                timing_csv_buffer_data = np.hstack((tstart, tend))
                np.savetxt(timing_csv_file, timing_csv_buffer_data[np.newaxis], delimiter=",", fmt='%f')
  return output_dict, timings



def main(_):
  device = "/"+FLAGS.aarch+":0" if len(FLAGS.aarch) > 1 else "/cpu:0"
  with tf.device(device):
        model_dir=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages","models")
        workload_details = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",FLAGS.workload_dir,'workload_details.json')
        with open(workload_details) as data_file:
            config_data = json.load(data_file)
        files_dir = PATH_TO_TEST_IMAGES_DIR
        data_dir=os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","packages", files_dir)
        files = os.listdir(data_dir)
        print ("Batch Size:", FLAGS.batch_size)
        log_buffer = open(os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",'ssdmobilenetv1','result','output','ssdmobilenetv1_final'+'_'+'batch'+str(FLAGS.batch_size)+'_'+FLAGS.aarch+'.txt'),"a")
        included_extenstions = ['jpg', 'jpeg']
        file_names = [fn for fn in os.listdir(data_dir)
              if any(fn.endswith(ext) for ext in included_extenstions)]

        batch_data=[]
        files = file_names[:FLAGS.batch_size]

        print(files)
        for f in files:
            image_path = data_dir + '/' + f
            batch_data = batch_from_image(image_path, FLAGS.batch_size, batch_data)
        print(np.shape(batch_data))
        if FLAGS.frozen_graph:
            output_dict, timing = run_inference(batch_data, config_data, model_dir, FLAGS.frozen_graph)
        else:
            raise ValueError("Either a Frozen Graph file or a SavedModel must be provided.")
        


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
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)