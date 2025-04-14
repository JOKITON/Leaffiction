import os
import cv2 as cv
import shutil
import json
from colorama import Back, Fore, Style
from typing import Callable
import numpy as np
from utils.img_transformation import enchance_img

def get_files_mod(folder_path, function: Callable, skip_current=True):
	arr_files = []
	arr_dirs = []
	for subdir, dirs, files in os.walk(folder_path):
		if subdir == folder_path:
			arr_dirs = dirs
			if skip_current is True:
				continue
		for file in files:
			arr_files.append(function(file))
	return arr_files, arr_dirs

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

def	get_rand_img(folder_path, skip_current=True):

	files, subdirs = get_files(folder_path)
	len_files = len(files[0]) + len(files[1]) + len(files[2]) + len(files[3])
	rand_num = np.round(np.random.rand() * len_files)
	it = 0
	for dir in subdirs:
		# Parse directory & check if it exists
		subdir_path = folder_path + dir
		for subdir, dirs, files in os.walk(subdir_path):
			for file in files:
				if it == rand_num:
					return subdir_path + '/' + file
				it += 1
	

def save_json(data, filename):
	"""Save data to a JSON file."""
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	with open(filename, "w") as f:
		json.dump(data, f, indent=4)

def check_zip(zip_path):
	if os.path.isfile(zip_path) and zip_path.endswith('.zip'):
		return True
	else:
		return False

def zip_folder(folder_path, output_path):
	shutil.make_archive(output_path, 'zip', folder_path)
	print(Style.BRIGHT + Fore.GREEN + f"Zip successfully created : {output_path}.zip" + Style.RESET_ALL)

def save_img(img, path):
	try:
		cv.imwrite(path, img)
	except:
		raise(MemoryError("error: the image could not be written:", path))

def get_img_group(config, img_path):
	if str.find(img_path, 'Apple') != -1:
		if str.find(img_path, config["apple_path"]) == -1:
			img_path = str(config["apple_path"] + img_path)
		group = "Apple"
	elif str.find(img_path, 'Grape') != -1:
		if str.find(img_path, config["grape_path"]) == -1:
			img_path = str(config["grape_path"] + img_path)
		group = "Grape"
	else:
		raise(ValueError("error: unknown image group"))

	return group

def conv_class_int(group, class_type):
	if group == "Apple":
		if class_type == 0:
			return "Apple_scab"
		if class_type == 1:
			return "Apple_healthy"
		if class_type == 2:
			return "Apple_rust"
		if class_type == 3:
			return "Apple_Black_rot"
	elif group == "Grape":
		if class_type == 0:
			return "Grape_Esca"
		if class_type == 1:
			return "Grape_healthy"
		if class_type == 2:
			return "Grape_Black_rot"
		if class_type == 3:
			return "Grape_spot"
	else:
		raise(ValueError("error: unknown class type"))

def conv_class_str(group, class_type):
	if group == "Apple":
		if str.find(class_type, 'Apple_scab') != -1:
			return "Apple_scab"
		if str.find(class_type, 'Apple_healthy') != -1:
			return "Apple_healthy"
		if str.find(class_type, 'Apple_rust') != -1:
			return "Apple_rust"
		if str.find(class_type, 'Apple_Black_rot') != -1:
			return "Apple_Black_rot"
	elif group == "Grape":
		if str.find(class_type, 'Grape_Esca') != -1:
			return "Grape_Esca"
		if str.find(class_type, 'Grape_healthy') != -1:
			return "Grape_healthy"
		if str.find(class_type, 'Grape_Black_rot') != -1:
			return "Grape_Black_rot"
		if str.find(class_type, 'Grape_spot') != -1:
			return "Grape_spot"
	else:
		raise(ValueError("error: unknown class type"))
