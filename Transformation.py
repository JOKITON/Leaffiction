import sys
import os
import cv2 as cv
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import numpy as np
from utils.data import get_files
from utils.dir import check_dir
from utils.img_transformation import check_img_ext, enchance_img, gaussian_blur, apply_mask, roi_obj, analize_img, pseudo_img

MAX_TRANSFORM = 50

def	handle_img(path_img, path_to_save=None, plt=True):
	if check_img_ext(path_img) is False:
		print(f"Error: The given argument is not an image: {path_img}")
		return
	if str.find(path_img, ').JPG') == -1:
		print(f"Error: Image needs to be the original one: {path_img}")
		return

	img = cv.imread(path_img)
	if img is None:
		print(f"Error: Could not load image {path_img}")
		return

	img = enchance_img(img)
	if plt is True:
		pcv.plot_image(img, plt)

	gaussian_blur(img, plt, str.replace(str(path_to_save), '.JPG', '_GaussianBlur.JPG'))
	apply_mask(img, plt, str.replace(str(path_to_save), '.JPG', '_Masked.JPG'))
	roi_obj(img, plt, str.replace(str(path_to_save), '.JPG', '_RoiObj.JPG'))
	analize_img(img, plt, str.replace(str(path_to_save), '.JPG', '_AnalysisObj.JPG'))
	pseudo_img(img, plt, str.replace(str(path_to_save), '.JPG', '_PseudoMarks.JPG'))
	if path_to_save is not None:
		print(path_to_save, "has been correctly transformed...")

def	handle_dir(src_dir, out_dir):
	src_dir = check_dir(src_dir)
	out_dir = check_dir(out_dir)

	files, subdirs = get_files(src_dir, skip_current=False)
	if len(subdirs) > 0:
		raise(ValueError("error: The given source directory does not contain any images:", src_dir))
	for file in files:
		for it, img in enumerate(file):
			if it == MAX_TRANSFORM:
				print("Alert: The maximum number of transformations have been reached...")
				return
			img_path = src_dir + img
			# Check if its the original image
			if str.find(img_path, ').JPG') == -1:
				continue
			# Check if img is valid
			handle_img(img_path, path_to_save=out_dir + img, plt=False)

def main():

	if len(sys.argv) == 2:
		handle_img(sys.argv[1], plt=True)
	elif len(sys.argv) == 3:
		handle_dir(sys.argv[1], sys.argv[2])
	else:
		raise(ValueError("error: Incorrect number of arguments. Please provide one or two arguments."))


main()