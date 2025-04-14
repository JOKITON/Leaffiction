import os
import sys
import json
import cv2 as cv
from plantcv import plantcv as pcv
from utils.data import get_files, get_img_path, conv_class
from utils.dir import check_dir, unzip, delete_dir
from utils.img_transformation import check_img_ext, enchance_img, gaussian_blur, apply_mask, roi_obj, analize_img, pseudo_img
from sklearn.model_selection import train_test_split
import numpy as np
from utils.model import save_model, load_model

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LearningCurveDisplay, learning_curve
from sklearn.model_selection import GridSearchCV

JSON_MAIN = "config.json"
script_path = check_dir(os.path.dirname(__file__))
json_path = script_path + JSON_MAIN

with open(json_path, "r") as f:
	config = json.load(f)
 
def extract_features(img):
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

def parse_features(json_path=None, features=None):
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

def	get_names(src_dir):
	src_dir = check_dir(src_dir)
	images = []
	features = []
	y_true = []

	y_index = 0
	_, subdirs = get_files(src_dir, skip_current=True)
	for dir in subdirs:
		root_dir = check_dir(src_dir + dir)
		for subdir, dirs, files in os.walk(root_dir):
			print(subdir)
			print(int(len(files) / 2))
			for file in files:
				file_name = dir + '/' + file
				if str.find(file, '.json') != -1:
					features.append(parse_features(root_dir + file))
					y_true.append(y_index)
			if len(files) > 0:
				y_index += 1

	if len(images) != len(features) != len(y_true):
		raise(ValueError("Files are not the same length: "
			+ str(len(images)) + " " + str(len(features)) + " " + str(len(y_true))))

	return images, features, y_true

def get_data(crt_zip=True, del_data=True, random_seed=42):
	src = script_path + config["zip_src"] + '.zip'
	out = script_path + config["zip_out"]

	if crt_zip is True:
		unzip(src, out)

	images, features, y_true = get_names(out)

	if del_data is True:
		delete_dir(out)

	X_train, X_test, y_train, y_test = train_test_split(
		features, y_true, test_size=0.2, random_state=random_seed)

	print("Training set size:", len(X_train), len(y_train))
	print("Test set size:", len(X_test), len(y_test))

	return X_train, X_test, y_train, y_test


def find_best_params(X, y, pipeline):
	param_grid = {
		"mlp__max_iter": [100, 125],
		"mlp__hidden_layer_sizes": [[128, 256, 512], [8, 16, 32, 64, 128, 256]],
	}

	grid_search = GridSearchCV(pipeline, param_grid, cv=None, scoring='accuracy', n_jobs=-1, error_score='raise')
	grid_search.fit(X, y)

	print("Best parameters:", grid_search.best_params_)
	print("Best CV accuracy:", grid_search.best_score_)
	return grid_search.best_estimator_

def classify_img(X_train, y_train, X_test, y_test, img_path):

	img, group = get_img_path(config, img_path)
	img = enchance_img(img)
	img = apply_mask(img, plt=False, path_to_save=None)

	to_test = extract_features(img)
	to_test = parse_features(features=to_test)

	pipeline = load_model("data/pipeline.pkl")
	""" pipeline = Pipeline([
		('scaler', StandardScaler()),
		('mlp', MLPClassifier(hidden_layer_sizes=[128, 256, 512], max_iter=100, learning_rate_init=0.001, alpha=0.001, random_state=42, learning_rate='adaptive'))
	]) """
	# Using the whole dataset to find the best params
	# find_best_params(X_train + X_test, y_train + y_test, pipeline)

	# Fit the pipeline to the data
	# pipeline.fit(X_train, y_train)

	# save_model("data/pipeline.pkl", pipeline)

	train_score = pipeline.score(X_train, y_train)
	test_score = pipeline.score(X_test, y_test)
	print("<------------------------->")
	print("Train set accuracy:", train_score)
	print("Test set accuracy(Untouched):", test_score)
	print()

	pred_img = pipeline.predict([to_test])
	print("Given image:", img_path)
	print("Predicted class:", conv_class(group, pred_img[0]))

	return test_score

def main():
	if len(sys.argv) == 2:
		img_name = sys.argv[1]
		X_train, X_test, y_train, y_test = get_data(crt_zip=False, del_data=False)
		test_score = classify_img(X_train, y_train, X_test, y_test, img_name)
	else:
		raise(ValueError("error: you need to provide a singular path as an argument:\n\t\tpython train.py <path_to_folder>"))

main()
