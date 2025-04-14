import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def save_model(path, alg):

	if alg is not None and path is not None:
		with open(path, 'wb') as file:
			pickle.dump(alg, file)

def load_model(path) -> Pipeline:

	if path is not None:
		with open(path, 'rb') as file:
			ret_alg = pickle.load(file)
		return ret_alg
	else:
		raise(ValueError(f"error: pipeline could not be found on: {path}. Change config.json accordingly..."))

def find_best_params(X, y, param_grid, pipeline):

	grid_search = GridSearchCV(pipeline, param_grid, cv=None, scoring='accuracy', n_jobs=-1, error_score='raise')
	grid_search.fit(X, y)

	print("Best parameters:", grid_search.best_params_)
	print("Best CV accuracy:", grid_search.best_score_)
	return grid_search.best_estimator_
