import os
import sys
from utils.plt import plot_bar_dis, plot_pie_dis

# Part 1: Analysis of the Data Set
def	data_analysis():
	folder_path = sys.argv[1]
	it_len = -1

	# Measure number of directories
	for subdir, dirs, files in os.walk(folder_path):
		it_len += 1

	ar_len = [0] * it_len
	it = 0
	ar_dirs = []
	for subdir, dirs, files in os.walk(folder_path):
		if subdir == folder_path:
			ar_dirs = dirs
			continue
		ar_len[it] += len(files)
		it += 1

	plot_bar_dis(ar_len, ar_dirs)
	plot_pie_dis(ar_len, ar_dirs)

def main():
	data_analysis()

main()