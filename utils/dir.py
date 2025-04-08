import os
import zipfile

def	check_dir(dir):
	if not os.path.isdir(dir):
		raise ValueError(f"Error: The given directory is not valid: {dir}")
	if dir[-1] != '/':
		dir += '/'
	return dir

def	unzip(src_path, out_path):
	with zipfile.ZipFile(src_path, 'r') as zip_ref:
		zip_ref.extractall(out_path)

	return zip_ref

def	create_folders(base_dir, subdirs, exist_ok=False):
	os.makedirs(base_dir)

	for subdir in subdirs:
		subdir_path = os.path.join(base_dir, subdir)
		os.makedirs(subdir_path)

	return base_dir

def delete_dir(dir_path):
	if os.path.isdir(dir_path):
		for root, dirs, files in os.walk(dir_path, topdown=False):
			for file in files:
				os.remove(os.path.join(root, file))
			for dir in dirs:
				os.rmdir(os.path.join(root, dir))
		os.rmdir(dir_path)
	else:
		raise ValueError(f"Error: The given path is not a directory: {dir_path}")
