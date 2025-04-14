import numpy as np
import json
from colorama import Back, Fore, Style
import os
import plantcv as pcv
import cv2 as cv
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from utils.data import get_files, get_img_group, conv_class_int, conv_class_str, save_json, zip_folder, check_zip, get_rand_img
from utils.dir import check_dir, delete_dir, create_folder, create_folders, unzip
from utils.img_transformation import check_img_ext, enchance_img, gaussian_blur, apply_mask, roi_obj, analize_img, pseudo_img
from utils.model import save_model, load_model, find_best_params
from utils.plt import plot_img_pred

class ImgClassifier:

	def __init__(self, config, script_path, verbose=False):
		"""Initialize EEGData object with file paths and optional filter settings."""
		self.IS_TRAINED = False
		self.MAX_TRAIN = 10000
		self.DIR_EXIST_OK = False
		self.DEL_DIR = True

		self.IS_DATASET_APPLE = False
		self.IS_DATASET_GRAPE = False

		self.IS_DATA_SAVED = False
		self.IS_PIPELINE_SAVED = False

		self.X_train: list = []
		self.y_train: list = []
		self.X_test: list = []
		self.y_test: list = []

		self.config : dict = config
		self.script_path = script_path
		self.src_dir : str = ""
		self.pipeline : Pipeline = None
		self.rand_img_ar : list = []

	def	plot_features(self, hist_blue, hist_green, hist_red):
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

	def crt_feature(self, img) -> dict:
		"""Extracts color and texture features from the leaf."""

		hist_blue = cv.calcHist([img],[0],None,[256],[0,256]).flatten().tolist()
		hist_green = cv.calcHist([img],[1],None,[256],[0,256]).flatten().tolist()
		hist_red = cv.calcHist([img],[2],None,[256],[0,256]).flatten().tolist()

		r = np.mean(img[:, :, 2]) # Red Channel Mean
		g = np.mean(img[:, :, 1])
		b = np.mean(img[:, :, 0])

		norm_r = r / (r + g + b)
		norm_g = g / (r + g + b)
		norm_b = b / (r + g + b)

		sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
		sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
		edges = cv.Canny(img, 750, 800, apertureSize=3)

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

	def crt_img(self, img_path, dir, img_name):
		if check_img_ext(img_path) is False:
			raise(ValueError(f"Error: The given argument is not an image: {img_path}"))

		# Create path to save the features
		path_to_save = check_dir(self.config["train_path"])
		path_to_save = check_dir(path_to_save + dir)
		path_to_save = path_to_save + img_name

		img = cv.imread(img_path)
		img = enchance_img(img)

		# Save the masked image
		# img = apply_mask(img, plt=False, path_to_save=path_to_save)

		return img, path_to_save

	def parse_features(self, json_path=None, features=None):
		json_features = []

		if json_path is not None:
			with open(json_path, "r") as f:
				features = json.load(f)
		elif features is None:
			raise(ValueError("error: you need to provide a json file or features"))

		for key in features:
			if type(features[key]) == list:
				for data in features[key]:
					json_features.append(data)
			else:
				json_features.append(features[key])

		return json_features

	def	extract_features(self, src_dir):
		src_dir = check_dir(src_dir)
		features = []
		y_true = []

		y_index = 0

		_, subdirs = get_files(src_dir, skip_current=True)
		for dir in subdirs:
			self.set_dataset_type(dir)
			root_dir = check_dir(src_dir + dir)
			for subdir, dirs, files in os.walk(root_dir):
				print(subdir)
				print(len(files))
				for file in files:
					if str.find(file, '.json') != -1:
						features.append(self.parse_features(root_dir + file))
						y_true.append(y_index)
				if len(files) > 0:
					y_index += 1

		if len(features) != len(y_true):
			raise(ValueError("Files are not the same length: "
				+ str(len(features)) + " " + str(len(y_true))))

		return features, y_true

	def	set_dataset_type(self, dir_to_check=None):
		if dir_to_check is None:
			dir_to_check = self.src_dir
		if str.find(dir_to_check, "Apple") != -1:
			self.IS_DATASET_APPLE = True
			self.IS_DATASET_GRAPE = False
		elif str.find(dir_to_check, "Grape") != -1:
			self.IS_DATASET_APPLE = False
			self.IS_DATASET_GRAPE = True
		else:
			raise(ValueError(f"error: The given directory does not correspond to any dataset type: {dir_to_check}"))

	def	crt_features(self, src_dir):
		# Check given directories & classify dataset
		self.src_dir = check_dir(src_dir)
		out_dir = self.config["train_path"]
		self.set_dataset_type(self.src_dir)

		# Initialize loop, get subdirs & create folders
		files, subdirs = get_files(self.src_dir, skip_current=True)

		create_folders(out_dir, subdirs, exist_ok=self.DIR_EXIST_OK)
		for dir in subdirs:
			# Parse directory & check if it exists
			root_dir = check_dir(self.src_dir + dir)
			for subdir, dirs, files in os.walk(root_dir):
				print(Style.DIM + subdir + Style.RESET_ALL)
				for file in files:
					# Read and extract features from each image, then create a JSON file
					img, path = self.crt_img(root_dir + file, dir, file)
					features = self.crt_feature(img)
					path_feat = str.replace(path, '.JPG', '.json')
					save_json(features, path_feat)


	def train(self, dir, zip_out=True):
		"""Train the model using the provided training data."""

		self.crt_features(dir)

		if zip_out is True:
			zip_folder(self.config["train_path"], self.config["zip_src"])
			print()

		if self.DEL_DIR is True:
			delete_dir(self.script_path + self.config["zip_out"])

		self.IS_TRAINED = True

	def	crt_pipeline(self, type=None):
		self.pipeline = Pipeline([
			('scaler', StandardScaler()),
			('mlp', MLPClassifier(hidden_layer_sizes=[256, 512, 512], max_iter=100, learning_rate_init=0.001, alpha=0.001, random_state=42, learning_rate='adaptive'))
		])

	def	manage_rand_img(self, len):
		for _ in range(len):
			if self.IS_DATASET_APPLE is True:
				self.rand_img_ar.append(get_rand_img(self.config["apple_path"]))
			elif self.IS_DATASET_GRAPE is True:
				self.rand_img_ar.append(get_rand_img(self.config["grape_path"]))
			else:
				raise(ValueError("error: The given "))

	def	check_img_proc(self, img_path):
		if self.IS_DATASET_APPLE is True and str.find(img_path, 'Grape') != -1:
			print(Style.BRIGHT + Fore.LIGHTRED_EX
				+ "The given image does not belong to the Apple dataset. Use another image..." + Style.RESET_ALL + Fore.RESET)
			return False
		elif self.IS_DATASET_GRAPE is True and str.find(img_path, 'Apple') != -1:
			print(Style.BRIGHT + Fore.LIGHTRED_EX
				+ "The given image does not belong to the Grape dataset. Use another image..." + Style.RESET_ALL + Fore.RESET)
			return False
		return True

	def	make_rand_pred(self, img_path):
		# Create features out of the given image to predict
		group = get_img_group(self.config, img_path)
		img = cv.imread(img_path)
		if img is None:
			raise(ValueError("error: image not found"))
		enh_img = enchance_img(img)
		to_test = self.crt_feature(enh_img)
		to_test = self.parse_features(features=to_test)

		pred_img = self.pipeline.predict([to_test])
		print("Image to predict:", img_path)
		print("Predicted class:", conv_class_int(group, pred_img[0]))
  
		plot_img_pred(img, img_path, conv_class_str(group, img_path), conv_class_int(group, pred_img[0]))

	def classify_img(self):

		# Check if img belongs to dataset
		for img in self.rand_img_ar:
			if self.check_img_proc(img) is False:
				return

		print()
		print("Training set size:", len(self.X_train), len(self.y_train))
		print("Test set size:", len(self.X_test), len(self.y_test))
  
		param_grid = {
			"mlp__max_iter": [100, 125, 150],
			"mlp__hidden_layer_sizes": [[128, 256, 512], [256, 512, 512], [256, 512, 1024], [64, 128, 256, 512]],
		}

		# Using the whole dataset to find the best params
		""" X = np.concatenate([self.X_train, self.X_test])
		y = np.concatenate([self.y_train, self.y_test])
		find_best_params(X, y, param_grid, self.pipeline) """

		# Fit the pipeline to the data
		self.pipeline.fit(self.X_train, self.y_train)

		""" pipeline = load_model("data/pipeline.pkl")
		save_model("data/pipeline.pkl", pipeline) """

		train_score = self.pipeline.score(self.X_train, self.y_train)
		test_score = self.pipeline.score(self.X_test, self.y_test)
		print("<------------------------->")
		print("Train set accuracy:", train_score)
		print("Test set accuracy(Untouched):", test_score)
		print()

		return

	def split_data(self, features, y_true, test_size=0.2):
		X_train, X_test, y_train, y_test = train_test_split(
			features, y_true, test_size=test_size, random_state=42, stratify=y_true)
		self.X_train, self.X_test = X_train, X_test
		self.y_train, self.y_test = y_train, y_test

	def	predict(self, n_preds=50):
		if check_zip(self.config["zip_src"] + '.zip') is False:
			self.IS_TRAINED = True
			print(Style.BRIGHT + Fore.LIGHTRED_EX
				+ "Model not trained yet. Please train the model before prediction." + Style.RESET_ALL + Fore.RESET)
			return

		if self.config["load_pipeline"] == "True":
			self.pipeline = load_model(self.config["pipeline_path"])
		else:
			self.crt_pipeline()
		
		unzip(self.config["zip_src"] + '.zip', self.config["zip_out"])

		features, y_true = self.extract_features(self.config["zip_out"])
		self.split_data(features, y_true)

		self.manage_rand_img(n_preds)

		self.classify_img()

		for img in self.rand_img_ar:
			self.make_rand_pred(img)

		if self.DEL_DIR is True:
			delete_dir(self.script_path + self.config["zip_out"])

	def	train_predict(self, dir, to_predict, zip_out=False):
		return
