import os
import cv2 as cv

def get_files(folder_path, skip_current=True):
	arr_files = []
	arr_dirs = []
	for subdir, dirs, files in os.walk(folder_path):
		if subdir == folder_path:
			arr_dirs = dirs
			if skip_current is True:
				continue	
		arr_files.append(files)
	return arr_files, arr_dirs

def save_img(img, path):
	try:
		cv.imwrite(path, img)
	except:
		raise(MemoryError("error: the image could not be written:", path))
