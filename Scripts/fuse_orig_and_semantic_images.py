"""
overlay original images and semantic classification images
"""
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
import glob
import sys
import time
import skimage.io



indir_orig=sys.argv[1]
indir_sem=sys.argv[2]
outdir=sys.argv[3]
alpha=float(sys.argv[4])

img_format=".jpg"

def findSemToOrig(orig_name, sem_images):
	""" searches the list of semantic images to match the original file name"""
	res=None
	numo=int(orig_name.split(".")[0])
	for si in sem_images:
		num=int(si.split("/")[-1].split("_")[1].split(".")[0])
		if num==numo:
			res=si
			return res
	return res


orig_images=glob.glob(indir_orig+"/*"+img_format)
orig_images.sort()
#print orig_images
sem_images=glob.glob(indir_sem+"/*"+img_format)
sem_images.sort()
#print sem_images
idx=-1
for img_path in orig_images:
	
	idx+=1
	imgO_grey=cv2.imread(img_path,0)
	imgO = cv2.cvtColor(imgO_grey,cv2.COLOR_GRAY2RGB)
	imgSpath=findSemToOrig(img_path.split("/")[-1], sem_images)
	imgS=cv2.imread(imgSpath)
	print "Processing original: \n"+str(img_path)
	print "and semantic image: \n"+str(imgSpath)
	#resize the semantic image
	fx=float(imgO.shape[1])/float(imgS.shape[1])
	fy=float(imgO.shape[0])/float(imgS.shape[0])
	print "fx: "+str(fx)+", fy: "+str(fy)

	imgS_resize=cv2.resize(imgS,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)

	print "new shape:"+str(imgS_resize.shape)
	print "orig shape: "+str(imgO.shape)

	imgOut=imgO.copy()
	cv2.addWeighted(imgO, alpha, imgS_resize, 1 - alpha, 0, imgOut)
	cv2.imwrite(outdir+"/"+str("{0:0=2d}".format(idx))+"_out.png", imgOut)


