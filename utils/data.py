import os
import cv2 as cv
import shutil
import json
from colorama import Back, Fore, Style

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

def save_json(data, filename):
	"""Save data to a JSON file."""
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	with open(filename, "w") as f:
		json.dump(data, f, indent=4)

def zip_folder(folder_path, output_path):
	shutil.make_archive(output_path, 'zip', folder_path)
	print(Style.BRIGHT + Fore.GREEN + f"Zip successfully created : {output_path}.zip" + Style.RESET_ALL)

def save_img(img, path):
	try:
		cv.imwrite(path, img)
	except:
		raise(MemoryError("error: the image could not be written:", path))

def get_img_path(config, img_path):
	if str.find(img_path, 'Apple') != -1:
		img_path = str(config["apple_path"] + img_path)
		group = "Apple"
	elif str.find(img_path, 'Grape') != -1:
		img_path = str(config["grape_path"] + img_path)
		group = "Grape"
	else:
		raise(ValueError("error: unknown image group"))
	img = cv.imread(img_path)
	if img is None:
		raise(ValueError("error: image not found"))

	return img, group

def conv_class(group, class_type):
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
			return "Grape_spot"
		if class_type == 1:
			return "Grape_healthy"
		if class_type == 2:
			return "Grape_Black_rot"
		if class_type == 3:
			return "Grape_Esca"
	else:
		raise(ValueError("error: unknown class type"))
