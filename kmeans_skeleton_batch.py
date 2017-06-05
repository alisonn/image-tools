#!/Users/alisonn/.virtualenvs/cv/bin/python
# Batch Image K-means Quantization

import numpy as np
import cv2
import sys
import os
from glob import glob
import argparse
from decimal import *

# handle positional arguments from command line and assign to variables
# @return: arguments - image_path_list, main_dir, K
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("main_dir", help = 'main-dir should be in context of main-dir/pev-crossN/images')
	parser.add_argument("K", help = 'should be num_clusters + 1 (for background)')
	parser.add_argument("calculation_file", help = 'file where results are written', default = "pev_results.tsv")
	args = parser.parse_args()
	return args 

# collects all image paths into array
# @param: name of output text file, path to dir with pev subdirectories
# @return: list with paths to individual images = [image1 path, image2 path, ..., imageN path]
## TESTING: PASS  6/5/2017 AHN ##
def get_images_from_dirs(main_dir):
	sub_dirs = glob(main_dir)
	image_paths = []
	for dir in sub_dirs:
		for file in os.listdir(dir):
			if file.endswith(".png"):
				image_paths.append(os.path.join(dir, file))
				print os.path.join(dir, file)
	return image_paths

# function handles reading the images one at a time into a master array so it may take a while
# @return: tuple - data set array and master_image_array whose elements are the image arrays themselves
# master image array = [image1 as array, image2 as array, ..., imageN as array]
## TESTING: OK PASS 5/30/2017 ##
def batch_read_images(image_path_list):
	master_image_arr = []
	image_ct = 0

	for img in image_path_list:
		image_arr = cv2.imread(img)
		master_image_arr.append(image_arr)
		image_Z = image_arr.reshape((-1,3))
		image_rbg = np.float32(image_Z)

		if image_ct == 0:
			data_set = image_rbg

		else:
			data_set = np.vstack((data_set,image_rbg))

		image_ct += 1

		## an aside ##
		# type(master_image_arr) >> <..numpy.ndarray>
		# master_image_arr.shape, master_image_arr.dtype >> ((dimensions, unit8?))

	print str(image_ct) + " images have been read into a master image array \n"
	return data_set,master_image_arr

##############

# define criteria, number of clusters(K) and apply kmeans()
# @param: data_set is ndarray with rbg data points, K is number of clusters
# @return:
## TESTING: PASS 5/30/2017  ###
def run_kmeans(data_set, K):
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(data_set,int(K),None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center) ## returns to conventional RBG 0-255 values

	return ret,label,center

###############

# this function identifies red and white clusters from user input
# @param: center from cv2.kmeans outpout
# @return: tuple of ints (red,white) - the ints corresp to arbitrary cluster labels
## TESTING: PASS 5/31/2017  ###
############ TODO: MAYBE ADD A HEURISTIC BGR CUTOFF BASED ON ADDITIVE COLOR THEORY ##
def choose_clusters(center):

	font = cv2.FONT_HERSHEY_SIMPLEX
	colorA = tuple(map(int, center[0]))
	colorB = tuple(map(int, center[1]))
	colorC = tuple(map(int, center[2]))

	print "OPENCV Colors are in BGR format"
	print "Cluster 0:  " + str(colorA)
	print "Cluster 1:  " + str(colorB)
	print "Cluster 2:  " + str(colorC)

	red_cluster = raw_input("Which cluster is RED? ")
	white_cluster = raw_input("Which cluster is WHITE? ")

	return red_cluster,white_cluster

# two subroutines to streamline the batch_process_kmeans function
# @return: a string
def get_new_img_path(img_path, suffix):
	parsed = img_path.split(".jpg")
	result = parsed[0] + suffix
	return result

def calculate_pev(label_subset, red_cluster, white_cluster):
	num_red = (label_subset == int(red_cluster)).sum()
	num_white = (label_subset == int(white_cluster)).sum()
	proportion_red = Decimal(num_red)/Decimal(num_red + num_white)		
	result = "\t" + str(num_red) + "\t" + str(num_white) + "\t" + str(proportion_red) + "\n"
	return result

#################

# this function should be two-fold: make new images and keep track of their %red as to avoid running through the script again
# @param: labels for every data point, centroid colors, array of all images, paths to original images
# @return: new images made, file with images and corresp pev scores
## TESTING: PASS 6/2/2017 ### 
def batch_process_kmeans_images(label, center, master_image, image_paths, calculation_file):
	f = open(calculation_file, 'w')
	f.write("image	red	white	red%" + "\n")
	red_cluster,white_cluster = choose_clusters(center)
	labels = label.flatten()
	start_index = 0
	end_index = 0
	img_ct = 0

	for img in master_image: 
		# get the array l x w --> length of the label interval to subset
		# bc python indexing, end index is where the iterator stops but does not access data
		img_label_len = img.shape[0] * img.shape[1]

		if img_ct == 0:
			end_index = (img_label_len)
			print "First Image Length, Dimensions"

		else:
			start_index = (end_index)
			end_index = (start_index + img_label_len)

		## Calculate PEV for every image and write to file
		label_subset = labels[start_index:end_index]
		pev_results = calculate_pev(label_subset, red_cluster, white_cluster)
		to_write = str(image_paths[img_ct]) + pev_results
		f.write(to_write)

		## Generate K-means image and save for record
		current_path = image_paths[img_ct]
		suffix = "_kmeans.jpg"
		new_img_path = get_new_img_path(current_path, suffix)
		res = center[label_subset.flatten()]
		final_img = res.reshape((img.shape))
		cv2.imwrite(new_img_path, final_img)

		img_ct += 1		

	f.close()
	print "We have processed this many images:  " + str(img_ct)

########################

def main():
	args = get_args()

	image_paths = get_images_from_dirs(args.main_dir)
	print "obtained paths to images"

	data_set,master_image = batch_read_images(image_paths)
	print "batch read images done"

	ret,label,center = run_kmeans(data_set, args.K)
	print "finished running k-means on data set"

	batch_process_kmeans_images(label, center, master_image, image_paths, args.calculation_file)

	print "wrote all images and corresponding pev percentages"

##########

main()	
