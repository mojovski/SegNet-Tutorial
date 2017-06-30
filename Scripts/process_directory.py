import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
from os import listdir
from os.path import isfile, join
from skimage import img_as_ubyte

import sys
import time
import skimage.io

"""
This script loads all images from a directory and
classifies them.
The results are stored in a directory set by the user params.
See a  list of some cool pretrained models for segnet from different datasets:
https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/segnet_model_zoo.md

"""


sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that caffe is on the python path:
caffe_root = '/home/eugen/projects/caffee/caffe-segnet/'
sys.path.insert(0, caffe_root + 'python')
import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--colours', type=str, required=True)
parser.add_argument('--indir', type=str, required=True)
parser.add_argument('--outdir', type=str, required=True)

args = parser.parse_args()

net = caffe.Net(args.model,
				args.weights,
				caffe.TEST)

#caffe.set_mode_gpu()

input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['argmax'].data.shape

label_colours = cv2.imread(args.colours).astype(np.uint8)

cv2.namedWindow('Input')
cv2.namedWindow('SegNet')

all_input_files= [f for f in listdir(args.indir) if isfile(join(args.indir, f))]



for fi in all_input_files:
	
	filename, file_extension = os.path.splitext(fi)

	start = time.time()
	
	if (file_extension==".tiff" or file_extension==".tif"): #if tif
		print "Trying to read tiff"
		im = skimage.io.imread(args.indir+"/"+fi, plugin='tifffile')
		frame=img_as_ubyte(im)
	else:
		frame=cv2.imread(args.indir+"/"+fi)
		
	if (len(frame.shape)==2): #if grey
		frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)

	if frame is None: # in case there is a bad image
		print "Skipping image "+fi 
		continue

	end = time.time()
	print '%30s' % 'Read image frame in ', str((end - start)*1000), 'ms'

	start = time.time()
	frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
	
	#stop here for debugging
	#time.sleep(20)



	input_image = frame.transpose((2,0,1))
	input_image = input_image[(2,1,0),:,:]
	input_image = np.asarray([input_image])
	end = time.time()
	print '%30s' % 'Resized image in ', str((end - start)*1000), 'ms'

	start = time.time()
	out = net.forward_all(data=input_image)
	end = time.time()
	print '%30s' % 'Executed SegNet in ', str((end - start)*1000), 'ms'

	start = time.time()
	segmentation_ind = np.squeeze(net.blobs['argmax'].data)
	segmentation_ind_3ch = np.resize(segmentation_ind,(3,input_shape[2],input_shape[3]))
	segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
	segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

	cv2.LUT(segmentation_ind_3ch,label_colours,segmentation_rgb)
	#segmentation_rgb = segmentation_rgb.astype(float)/255

	end = time.time()
	print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms\n'

	cv2.imshow("Input", frame)
	cv2.imshow("SegNet", segmentation_rgb)
	ext=os.path.basename(fi).split(".")[-1]
	filename=os.path.basename(fi).split(".")[0]
	#resize to the original image size
	fx=float(frame.shape[1])/float(segmentation_rgb.shape[1])
	fy=float(frame.shape[0])/float(segmentation_rgb.shape[0])
	print "fx: "+str(fx)+", fy: "+str(fy)

	imgS_resize=cv2.resize(segmentation_rgb,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
	cv2.imwrite(args.outdir+"/"+filename+"_class."+ext, imgS_resize)
	
	key = cv2.waitKey(1)
	if key == 27: # exit on ESC
		break
cv2.destroyAllWindows()

