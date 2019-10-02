import sys
import os

dir_name="ssd-mobilenet_v1"
sys.path.insert(0, os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","bin"))
from run import object_detection_ssd
object_detection_ssd(dir_name)
