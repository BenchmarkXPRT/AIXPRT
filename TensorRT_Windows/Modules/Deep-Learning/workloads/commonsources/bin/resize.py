import sys
import os
import glob
from PIL import Image
import numpy as np
import shutil

size = sys.argv[1]
resize_size=256
if int(size) > 256:
	resize_size=325
path = '../../../packages/input_images/*'
files = glob.glob(path)
image_folder = '../../../packages/input_images_'+str(size)+'/'
if os.path.exists(image_folder):
	list_files = os.listdir(image_folder)
	if (len(list_files) == 5000):
		exit()
	else:
		shutil.rmtree(image_folder)

os.makedirs(image_folder)
for file in files:
	filename, fileext = os.path.splitext(file)
	i = Image.open(file)
	j = i.resize((int(resize_size), int(resize_size)), Image.ANTIALIAS)
	size_crop=(resize_size-int(size))/2
	size_crop_btm=(resize_size+int(size))/2
	crop_img = j.crop((size_crop, size_crop, size_crop_btm, size_crop_btm))
	image_out = image_folder +os.path.basename(filename)+''+os.path.basename(fileext)
	crop_img.save(image_out)
