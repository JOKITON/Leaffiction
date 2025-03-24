import sys
import cv2 as cv
from utils.data import get_files
import numpy as np

def	img_flip(img_path):
	img = cv.imread(img_path)
	img_flip = cv.flip(img, 1)
	img_path = str.replace(img_path, '.JPG', '_Flip.JPG')
	cv.imwrite(img_path, img_flip)

def	img_rotate(img_path, angle):
	img = cv.imread(img_path)
	img_flip = cv.rotate(img, angle)
	img_path = str.replace(img_path, '.JPG', '_Rotate.JPG')
	cv.imwrite(img_path, img_flip)

def	img_skew(img_path, angle):
	img = cv.imread(img_path)
	rows, cols = img.shape[:2]
	M = np.float32([[1 - angle, angle, 0], [0, 1, 0]])
	img_flip = cv.warpAffine(img, M, (cols, rows))
	img_path = str.replace(img_path, '.JPG', '_Skew.JPG')
	cv.imwrite(img_path, img_flip)

def	data_augmentation():
	folder_path = sys.argv[1]
	files, dirs = get_files(folder_path)

	for file, dir in zip(files, dirs):
		for img in file:
			img_path = folder_path + '/' + dir + '/' + img
			if str.find(img_path, ').JPG') == -1:
				continue
			elif dir == 'Apple_rust':
				img_flip(img_path)
				img_rotate(img_path, cv.ROTATE_90_CLOCKWISE)
				img_skew(img_path, 0.25)

def main():
	data_augmentation()

main()