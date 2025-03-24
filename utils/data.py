import os

def get_files(folder_path):
	arr_files = []
	arr_dirs = []
	for subdir, dirs, files in os.walk(folder_path):
		if subdir == folder_path:
			arr_dirs = dirs
			continue
		arr_files.append(files)
	return arr_files, arr_dirs