import cv2 as cv
from plantcv import plantcv as pcv
import numpy as np

WHITE_RGB = [255, 255, 255]
GREEN_RGB = [0, 255, 100]
LOWER_GREEN_HSV = [25, 20, 20]
HIGHER_GREEN_HSV = [100, 255, 255]

def	check_img_ext(img):
	valid_extensions = ('.JPG', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
	return img.lower().endswith(valid_extensions)

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

def	gaussian_blur(img, plt=True, path_to_save=None):

	# Convert to HSV system
	hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

	# Threshold the hue to isolate green/brown values
	lower_green = np.array(LOWER_GREEN_HSV)
	upper_green = np.array(HIGHER_GREEN_HSV)
	mask = cv.inRange(hsv_img, lower_green, upper_green)

	s_gblur = pcv.gaussian_blur(img=mask, ksize=(3, 3),
								sigma_x=0, sigma_y=None)
	s_thresh = pcv.threshold.binary(
		gray_img=s_gblur, threshold=75, object_type="dark"
	)

	# Plot the masked image
	if plt is True:
		pcv.plot_image(s_thresh)
	# Save the image
	if path_to_save is not None:
		cv.imwrite(path_to_save, s_thresh)

	return s_thresh

def apply_mask(img, plt=True, path_to_save=None):

	hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

	lower_green = np.array(LOWER_GREEN_HSV)
	upper_green = np.array(HIGHER_GREEN_HSV)
	mask = cv.inRange(hsv_img, lower_green, upper_green)

	img_mask = pcv.apply_mask(img=img, mask=mask, mask_color='white')

	if plt is True:
		pcv.plot_image(img_mask)
	if path_to_save is not None:
		cv.imwrite(path_to_save, img_mask)

	return img_mask

def	roi_obj(img, plt=True, path_to_save=None):
	hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

	lower_green = np.array(LOWER_GREEN_HSV)
	upper_green = np.array(HIGHER_GREEN_HSV)
	mask = cv.inRange(hsv_img, lower_green, upper_green)

	roi = pcv.roi.rectangle(img=mask, x=25, y=25, h=225, w=200)

	filtered_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type='partial')

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

	if plt is True:
		pcv.plot_image(highlighted_img)
	if path_to_save is not None:
		cv.imwrite(path_to_save, highlighted_img)

	return highlighted_img

def analize_img(img, plt=True, path_to_save=None):
	# Convert to HSV system
	hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

	# Threshold the hue to isolate green/brown values
	lower_green = np.array(LOWER_GREEN_HSV)
	upper_green = np.array(HIGHER_GREEN_HSV)
	mask = cv.inRange(hsv_img, lower_green, upper_green)

	analysis_image = pcv.analyze.size(img=img, labeled_mask=mask)

	if plt is True:
		pcv.plot_image(analysis_image)
	if path_to_save is not None:
		cv.imwrite(path_to_save, analysis_image)

	return analysis_image

def	pseudo_img(img, plt=True, path_to_save=None):
	# Convert to HSV system
	hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

	# Threshold the hue to isolate green/brown values
	lower_green = np.array(LOWER_GREEN_HSV)
	upper_green = np.array(HIGHER_GREEN_HSV)
	mask = cv.inRange(hsv_img, lower_green, upper_green)

	try:
		# Find the contours of the leaf
		homolog_pts, start_pts, stop_pts, ptvals, chain, max_dist = pcv.homology.acute(img=img, mask=mask, win=40, threshold=110)
	except:
		print("error: could not calculature the pseudo-marks. Skipping...")
		return hsv_img

	homolog_pts = sorted(homolog_pts, key=lambda pt: pt[0][0])  # Sort by X
	start_pts = sorted(start_pts, key=lambda pt: pt[0][0])
	stop_pts = sorted(stop_pts, key=lambda pt: pt[0][0])

	result_img = img.copy()
	for pt in homolog_pts:
		cv.circle(result_img, tuple(pt[0]), 5, (0, 0, 255), -1)  # Red points

	for pt in start_pts:
		cv.circle(result_img, tuple(pt[0]), 5, (255, 0, 0), -1)  # Blue (Start)

	for pt in stop_pts:
		cv.circle(result_img, tuple(pt[0]), 5, (0, 255, 0), -1)  # Green (End)

	if plt is True:
		pcv.plot_image(result_img)
	if path_to_save is not None:
		cv.imwrite(path_to_save, result_img)

	return result_img
