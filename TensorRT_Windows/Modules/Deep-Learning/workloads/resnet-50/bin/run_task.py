import sys
import os

model_name= "resnet-50" 
dir_name="resnet-50"
sys.path.insert(0, os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","bin"))
from run import image_classification
image_classification(model_name,dir_name)
