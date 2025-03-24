from matplotlib import pyplot as plt

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
