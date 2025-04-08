import pickle

def save_model(path, alg):

	if alg is not None and path is not None:
		with open(path, 'wb') as file:
			pickle.dump(alg, file)

def load_model(path):

	if path is not None:
		with open(path, 'rb') as file:
			ret_alg = pickle.load(file)
		return ret_alg
