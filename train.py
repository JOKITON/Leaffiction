import sys
import os
import json
import cv2 as cv
from utils.data import get_files, zip_folder, save_json
from utils.dir import check_dir, delete_dir, create_folders
from utils.img_transformation import check_img_ext, enchance_img, gaussian_blur, apply_mask, roi_obj, analize_img, pseudo_img
import shutil
import numpy as np
from plantcv import plantcv as pcv
from matplotlib import pyplot as plt
from colorama import Back, Fore, Style

JSON_MAIN_PATH = "config.json"
MAX_TRAIN = 5000

with open(JSON_MAIN_PATH, "r") as f:
    config = json.load(f)

def extract_features(img_name, img, plt_data=False):
	"""Extracts color and texture features from the leaf."""

	hist_blue = cv.calcHist([img],[0],None,[256],[0,256]).flatten().tolist()
	hist_green = cv.calcHist([img],[1],None,[256],[0,256]).flatten().tolist()
	hist_red = cv.calcHist([img],[2],None,[256],[0,256]).flatten().tolist()

	hue_img = pcv.rgb2gray_hsv(img, channel='h')

	r = np.mean(img[:, :, 2]) # Red Channel Mean
	g = np.mean(img[:, :, 1])
	b = np.mean(img[:, :, 0])

	norm_r = r / (r + g + b)
	norm_g = g / (r + g + b)
	norm_b = b / (r + g + b)

	sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
	sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
	edges = cv.Canny(img, 750, 800, apertureSize=3)

	""" 
	print("Color Ratios:", r / g, r / b, g / b)
	print("Sobel X & Y:", np.mean(np.abs(sobelx)), np.mean(np.abs(sobely)))
	print("Edge density:", np.sum(edges > 0) / edges.size)
	pcv.plot_image(hue_img)
	"""

	if plt_data:
		fig, axes = plt.subplots(1, 3, figsize=(12, 4))

		axes[0].plot(hist_blue, color='b')
		axes[0].set_title('Blue Channel')
		axes[0].set_xlabel("Pixel Intensity")
		axes[0].set_ylabel("Pixel Count")
		axes[0].grid(True)

		axes[1].plot(hist_green, color='g')
		axes[1].set_title('Green Channel')
		axes[1].set_xlabel("Pixel Intensity")
		axes[1].grid(True)

		axes[2].plot(hist_red, color='r')
		axes[2].set_title('Red Channel')
		axes[2].set_xlabel("Pixel Intensity")
		axes[2].grid(True)

		plt.suptitle("Image Histogram for BGR Channels")
		plt.tight_layout()
		plt.show()

	return {
		# "filename": img_name,
		"edge_density" : np.sum(edges > 0) / edges.size,
		"sobel_x" : np.mean(np.abs(sobelx)),
		"sobel_y" : np.mean(np.abs(sobely)),

		"r_g_ratio": r / g if g > 0 else 0,
		"r_b_ratio": r / b if b > 0 else 0,
		"g_b_ratio": g / b if b > 0 else 0,

		"r_norm" : norm_r,
		"g_norm" : norm_g,
		"b_norm" : norm_b,

		"hist_blue": hist_blue,
		"hist_green": hist_green,
		"hist_red": hist_red
	}

def	augment_img(img_path, dir, img_name):
	if check_img_ext(img_path) is False:
		raise(ValueError(f"Error: The given argument is not an image: {img_path}"))

	path_to_save = check_dir(config["train_path"])
	path_to_save = check_dir(path_to_save + dir)
	path_to_save = path_to_save + img_name

	img = cv.imread(img_path)
	img = enchance_img(img)

	# Save the original image
	# cv.imwrite(path_to_save, img)
	# Save the masked image
	img = apply_mask(img, plt=False, path_to_save=path_to_save)

	# Create and save the features
	features = extract_features(img_name, img, plt_data=False) # Img is in BGR format
	path_feat = str.replace(path_to_save, '.JPG', '.json')
	save_json(features, path_feat)


def	manage_dir(src_dir):
	src_dir = check_dir(src_dir)
	out_dir = config["train_path"]

	_, subdirs = get_files(src_dir, skip_current=True)
	it = 0
	create_folders(out_dir, subdirs, exist_ok=False)
	for dir in subdirs:
		root_dir = check_dir(src_dir + dir)
		for subdir, dirs, files in os.walk(root_dir):
			print(Style.DIM + subdir + Style.RESET_ALL)
			it = 0
			for file in files:
				if it >= MAX_TRAIN:
					break
				it += 1
				augment_img(root_dir + file, dir, file)
	zip_folder(out_dir, config["zip_src"])
	# Delete the folder after zipping
	delete_dir(out_dir)

def main():
	if len(sys.argv) == 2:
		manage_dir(sys.argv[1])
	else:
		raise(ValueError("error: you need to provide a singular path as an argument:\n\t\tpython train.py <path_to_folder>"))

main()
