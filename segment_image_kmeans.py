#!/usr/bin/env python
## Segmentation Script: Implements K-means clustering to segment regions from images ##

## Using OpenComputerVision for python image processing and k-means skeleton
import sys
import argparse
import numpy as np
import cv2

# Handle command line input
def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("imageFile")
	parser.add_argument("segmentedImage")
	args = parser.parse_args()
	return args

# Read in image as an array in numpy
def readImageAsArray(imageFile):
	imageArr = cv2.imread(args.imageFile)
	type(imageArr)
	## >> <.. numpy.ndarray>
	imageArr.shape, imageArr.dtype
	## >> ((dimensions, unit8?))

	Z = img.reshape((-1,3))
	Z = np.float32(Z)
	return Z

def operateKClusters():
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 4
	ret,label,center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	
	cv2.imshow('res2', res2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
