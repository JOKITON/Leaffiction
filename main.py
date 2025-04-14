import json
import os
from utils.dir import check_dir
from img_classifier import ImgClassifier
from colorama import Style, Fore, Back

JSON_MAIN_PATH = "config.json"
RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL

YES_NO = Fore.GREEN + Style.BRIGHT + " (yes" + RESET_ALL
YES_NO += " / " + Fore.RED + Style.BRIGHT + "no): " + RESET_ALL + Style.BRIGHT

script_path = check_dir(os.path.dirname(__file__))

with open(JSON_MAIN_PATH, "r") as f:
    config = json.load(f)

def	manage_train(obj):
	print("\n└─> Choose a dataset (" + Fore.LIGHTWHITE_EX + Style.BRIGHT
		+ "only numbers" + RESET_ALL + "): ")
	print("\t[1]" + Style.BRIGHT + "🍎 Apple " + RESET_ALL + "dataset" + RESET_ALL)
	print("\t[2]" + Style.BRIGHT + "🍐 Grape " + RESET_ALL + "dataset" + RESET_ALL)
	print("\t[3]" + Fore.MAGENTA + " Go back " + RESET_ALL)
	while 1:
		str = input(Style.BRIGHT + Fore.LIGHTCYAN_EX + "└─> " + RESET_ALL)
		if str.isdigit() is True:
			if int(str) < 1 and int(str) > 3:
				return int(str)
			elif int(str) == 3:
				return int(str)
			elif int(str) == 1:
				obj.train("data/Apple")
				return int(str)
			elif int(str) == 2:
				obj.train("data/Grape")
				return int(str)

def	manage_predict(obj):
    obj.predict()

def	manage_options(obj):
	while 1:
		print("\n└─> Choose an option (" + Fore.LIGHTWHITE_EX + Style.BRIGHT
			+ "only numbers" + RESET_ALL + "): ")
		print("\t[1]🧬 " + Style.BRIGHT + Back.LIGHTCYAN_EX + "Train" + RESET_ALL)
		print("\n\t[2]📊 " + Style.BRIGHT + Back.LIGHTRED_EX + "Predict" + RESET_ALL)
		print("\n\t[3]⏩ " + Style.BRIGHT + Back.LIGHTMAGENTA_EX + "Train & Predict" + RESET_ALL)
		while 1:
			str1 = input(Style.BRIGHT + Fore.LIGHTCYAN_EX + "└─> " + Fore.RESET)
			print(RESET_ALL)
			if str1.isdigit() is True:
				option = int(str1)

				if option == 1:
					ret = manage_train(obj)
					break
					
				elif option == 2:
					manage_predict(obj)
					break
				elif option == 3:
					ret = manage_train(obj)
					if ret == 3:
						break
					manage_predict(obj)
					break
				else:
					print(
						Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
			else:
				print(
					Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
			continue

def	main():
    obj = ImgClassifier(config, script_path, verbose=False)
    manage_options(obj)
    # obj.train("data/Apple")
    # obj.predict("data/Apple/Apple_rust/image (64).JPG")
    
main()
