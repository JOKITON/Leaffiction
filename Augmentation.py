import sys
import json
import cv2 as cv
from utils.data import get_files, check_dir
import numpy as np

JSON_MAIN_PATH = "config.json"

with open(JSON_MAIN_PATH, "r") as f:
    config = json.load(f)

def	img_flip(img_path, img):
	img_flip = cv.flip(img, 1)
	img_path = str.replace(img_path, '.JPG', '_Flip.JPG')
	cv.imwrite(img_path, img_flip)

def	img_rotate(img_path, img, angle):
	img_flip = cv.rotate(img, angle)
	img_path = str.replace(img_path, '.JPG', '_Rotate.JPG')
	cv.imwrite(img_path, img_flip)

def	img_skew(img_path, img, angle, direction='x'):
	rows, cols = img.shape[:2]
	# Define shearing transformation matrix
	if direction == 'x':
		M = np.float32([[1, angle, 0], [0, 1, 0]])  # Horizontal shear
	elif direction == 'y':
		M = np.float32([[1, 0, 0], [angle, 1, 0]])  # Vertical shear
	else:
		print("Invalid direction! Use 'x' or 'y'.")
		return
	img_flip = cv.warpAffine(img, M, (cols, rows))
	img_path = str.replace(img_path, '.JPG', '_Skew.JPG')
	cv.imwrite(img_path, img_flip)

def	img_shear(img_path, img, shear_factor, direction='x'):
	rows, cols = img.shape[:2]
	# Define shearing transformation matrix
	if direction == 'x':
		M = np.float32([[1, shear_factor, 0], [0, 1, 0]])  # Horizontal shear
	elif direction == 'y':
		M = np.float32([[1, 0, 0], [shear_factor, 1, 0]])  # Vertical shear
	else:
		print("Invalid direction! Use 'x' or 'y'.")
		return
	img_flip = cv.warpAffine(img, M, (cols, rows))
	img_path = str.replace(img_path, '.JPG', '_Shear.JPG')
	cv.imwrite(img_path, img_flip)

def	img_crop(img_path, img, crop_rate):
	X, y, _ = img.shape
	img_crop = img[int(X*crop_rate):X, int(y*crop_rate):y]
	img_path = str.replace(img_path, '.JPG', '_Crop.JPG')
	cv.imwrite(img_path, img_crop)
 
def	img_distort(img_path, img, distortion_rate):

	A = img.shape[0] / 5.0 # Maximum pixel displacement
	w = 2.0 / img.shape[1] # Frequency of the sine wave

	shift = lambda x: A * np.sin(distortion_rate*np.pi*x * w)

	img_dist = np.zeros_like(img)

	for i in range(img.shape[1]):
		shift_val = int(shift(i))  # Compute the shift amount
		for c in range(3):  # Apply the shift to each color channel separately
			img_dist[:, i, c] = np.roll(img[:, i, c], shift_val)

	img_path = str.replace(img_path, '.JPG', '_Distortion.JPG')
	cv.imwrite(img_path, img_dist)

def	rand_aug(img_path, img):
	number = np.random.rand() * 6
	if number < 1:
		img_flip(img_path, img)
	elif number < 2:
		img_rotate(img_path, img, cv.ROTATE_90_CLOCKWISE)
	elif number < 3:
		img_skew(img_path, img, 0.3, direction='x')
	elif number < 4:
		img_shear(img_path, img, 0.3, direction='y')
	elif number < 5:
		img_crop(img_path, img, 0.1)
	else:
		img_distort(img_path, img, 1)

def	img_augmentation_folder(folder_path, subdir, times, rate):
	folder_path = check_dir(folder_path)
	folder_path = check_dir(folder_path + subdir)
	files, dirs = get_files(folder_path, skip_current=False)

	files = np.array(files).flatten()
	for i in range(times):
		i += 1
		for file in zip(files):
			for it, img in enumerate(file):
				img_path = folder_path + str(img)
				# Check if its the original image
				if str.find(img_path, ').JPG') == -1:
					continue
				# Check if img is valid
				img = cv.imread(img_path)
				if img is None:
					print(f"Error: Could not load image {img_path}")
					continue
				# Data augmentation
				elif it % rate == 0:
					rand_aug(img_path, img)

def	img_augmentation(img_path):
	if str.find(img_path, ').JPG') == -1:
		print(f"Error: Image needs to be the original one: {img_path}")
		return

	img = cv.imread(img_path)
	if img is None:
		print(f"Error: Could not load image {img_path}")
		return

	img_flip(img_path, img)
	img_rotate(img_path, img, cv.ROTATE_90_CLOCKWISE)
	img_skew(img_path, img, 0.3, direction='x')
	img_shear(img_path, img, 0.3, direction='y')
	img_crop(img_path, img, 0.1)
	img_distort(img_path, img, 1)

def main():
	img_path = sys.argv[1]
	if img_path is None:
		print(f"Error: Could not find image {img_path}")
		return

	if 'Apple' in img_path:
		img_augmentation_folder(img_path, 'Apple_rust', config["aug_apple_rust"], 1)
		img_augmentation_folder(img_path, 'Apple_scab', config["aug_apple_scab"], 1)
		img_augmentation_folder(img_path, 'Apple_Black_rot', config["aug_apple_Black_rot"], 1)
	elif 'Grape' in img_path:
		img_augmentation_folder(img_path, 'Grape_healthy', config["aug_grape_healthy"], 1)

main()