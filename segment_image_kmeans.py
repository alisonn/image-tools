#!/Users/alisonn/.virtualenvs/cv/bin/python
## Segmentation Script: Implements K-means clustering to segment regions from images ##
## Using OpenComputerVision for python image processing and k-means skeleton

import sys
import argparse
import numpy as np
import cv2

# Handle command line input
def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("fileList")
	parser.add_argument("imageFile")
	parser.add_argument"K_VALUE")
	args = parser.parse_args()
	return args

# DefCentroids: finds the centroids with information on ALL of the data you want
# @param: list of image files in a txt file separated by a newline, your desired K value
# @return: a set of centroids (i.e. BGR) in numpy array
def getCentroids(fileList, K):
	f = open(fileList, 'r')
	line = f.readline()
	masterArray = None #Initialize the master-keeping array

	while line:
		addArray = cv2.imread(line)		
		np.concatenate((masterArray, addArray), axis = 0)
		line = f.readline()
	f.close() # done with reading the text file

	Z = masterArray.reshape((-1,3))
	Z = np.float32(Z)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

	## Find the clusters based on data and given K
	ret,label,center = cv2.kmeans(Z,K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center) # center == centroids
	return center

def identifyCluster(color):
	cluster = raw_input("Which number is the " + color + " pigment? ")
	return cluster ## returns the label for whichever color cluster

def processImages(imageFile, center):
	# Read in the images, one at a time
	
	## CALCULATE EUCLIDEAN DISTANCE FOR EACH OF YOUR IMAGE'S PIXELS
	imageArr = cv2.imread(imageFile)
	Z = imageArr.reshape((-1,3))
	Z = np.float32(Z)
	
	# You have IMAGEARR and CENTER(array) so compute euclidean distance quickly
	#
	#	for each centroid:
	#		compute distances for every element in image to that centroid
	#		record the distances in a new array element whose row indices corresp to the same pixel
	#
	#	the col indices of this new array should corresp to the centroid cluster number
	# Calculate euclidean distances between pixel and cluster using faster method
	distances = np.sqrt(np.sum(((z-x)**2), axis = 1))

	# Find the minimum value of each row (i.e. the cluster to which the pixel belongs)
	# Return the column index of the minimum value per row --> assign as the cluster for that pixel

	res = center[label.flatten()]
	res2 = res.reshape((imageArr.shape))
	print label[(label == 0)]
	print label[(label == 1)]
	print label[(label == 2)]
	print label[(label >= 0)]
	##cv2.imshow('res2', res2)
	##cv2.waitKey(0)
	##cv2.destroyAllWindows()

## TODO: Draw the centroids or write them out

def main():
	args = getArgs()
	center = getCentroids(args.fileList, args.K_VALUE)
	processImage(args.imageFile, center)

main()
