from matplotlib import pyplot as plt
from utils.img_transformation import apply_mask

def	plot_pie_dis(ar_len, ar_dirs):
	fig, ax = plt.subplots()
	colors = ['lightgreen', 'green', 'brown', 'olive']
	ax.pie(ar_len, labels=ar_dirs, autopct='%1.2f%%', colors=colors)
	ax.set_xlabel('Pie Class distribution', loc='left')
	plt.show()


def	plot_bar_dis(ar_len, ar_dirs):
	fig, ax = plt.subplots()
	colors = ['lightgreen', 'green', 'brown', 'olive']
	ax.bar(ar_dirs, ar_len, color=colors)
	ax.set_xlabel('Different types')
	ax.set_ylabel('Number of images')
 
	plt.show()

def	plot_img_pred(img, img_path, true, pred):
	if true == pred:
		facecolor = '#6aed6e'
	else:
		facecolor = '#fb4d27'

	img_masked = apply_mask(img, plt=False, path_to_save=None)
	fig, axs = plt.subplots(1, 2, figsize=(20, 6), facecolor=facecolor)

	axs[0].imshow(img)
	axs[1].imshow(img_masked)
	axs[0].set_ylabel(f'Img: {img_path}', fontsize='medium', fontweight='bold')
	axs[0].set_xlabel(f'True class: {true}', fontsize='large', fontweight='bold')
	axs[1].set_xlabel(f'Predicted class: {pred}', fontsize='large', fontweight='bold')

	plt.show()

