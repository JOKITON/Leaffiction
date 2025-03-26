import sys
import cv2 as cv
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import numpy as np

WHITE = [255, 255, 255]
GREEN_RGB = [0, 255, 100]

def	apply_gaus(img, width):
	for X, y in img:
		result = 1 / (np.pi * 2 * width) * np.exp((X**2 + y**2) / (2 * width**2))
		img[X, y] = result
	return img

def	enchance_img(img):
	clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
	lab[:,:,0] = clahe.apply(lab[:,:,0])
	enhanced_img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

	return enhanced_img

def	gaussian_blur(img):

	# Convert using the Saturation channel
	gray_img = pcv.rgb2gray_hsv(rgb_img=img, channel='h')

	# Threshold the hue to isolate green/brown values
	gray_img = cv.inRange(gray_img, 25, 90)

	s_gblur = pcv.gaussian_blur(img=gray_img, ksize=(3, 3),
								sigma_x=0, sigma_y=None)
	s_thresh = pcv.threshold.binary(
		gray_img=s_gblur, threshold=75, object_type="dark"
	)

	pcv.plot_image(s_thresh)

def apply_mask(img):
	# Convert using the Saturation channel
	gray_img = pcv.rgb2gray_hsv(rgb_img=img, channel='h')

	# Threshold the hue to isolate green/brown values
	hsv_img = cv.inRange(gray_img, 25, 90)

	# Threshold the saturation to isolate the leaf
	mask = cv.threshold(hsv_img, 60, 255, cv.THRESH_BINARY)[1]

	img_mask = pcv.apply_mask(img=img, mask=mask, mask_color='white')

	# Plot the masked image
	pcv.plot_image(img_mask)

def	roi_obj(img):
	gray_img = pcv.rgb2gray_hsv(rgb_img=img, channel='h')

	hsv_img = cv.inRange(gray_img, 25, 90)

	roi = pcv.roi.rectangle(img=hsv_img, x=25, y=25, h=225, w=200)

	filtered_mask = pcv.roi.filter(mask=hsv_img, roi=roi, roi_type='partial')

	# Convert original image to a copy where the leaf will be highlighted
	highlighted_img = img.copy()

	# Apply a greenish color to the leaf using the mask
	highlighted_img[np.where(filtered_mask == 255)] = GREEN_RGB

	coordinates = np.column_stack(np.where(filtered_mask == 255))

	# Get the minimum values of X & Y
	start_X = np.min(coordinates[:, 0])
	max_X = np.max(coordinates[:, 0])
	start_Y = np.min(coordinates[:, 1])
	max_Y = np.max(coordinates[:, 1])

	# PlantCV uses (row, col) format for images
	# start_Y, start_X: Top left corner of the rectangle
	# max_Y, max_X: Bottom right corner of the rectangle
	width = max_Y - start_Y
	height = max_X - start_X

	# **Draw the ROI rectangle on the image**
	cv.rectangle(highlighted_img, (start_Y, start_X), (start_Y + width, start_X + height), (0, 0, 255), 2)

	pcv.plot_image(highlighted_img)

def main():
	img_path = sys.argv[1]
	if img_path is None:
		print(f"Error: Could not find image {img_path}")
		return

	if str.find(img_path, ').JPG') == -1:
		print(f"Error: Image needs to be the original one: {img_path}")
		return

	img = cv.imread(img_path)
	if img is None:
		print(f"Error: Could not load image {img_path}")
		return

	pcv.plot_image(img)
	gaussian_blur(img)
	apply_mask(img)
	roi_obj(img)

main()