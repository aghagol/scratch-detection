import numpy as np
import cv2
 
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage import morphology

def build_filters():
	filters = []
	ksize = 51
	sigma = 4.2
	lamda = 10.0
	gamma = 0.3
	psi = 0
	for theta in np.arange(0, np.pi, np.pi / 16):
		kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
		kern /= 1.5*kern.sum()
		filters.append(kern)
	return filters

def process(img, filters):
	accum = np.zeros_like(img)
	for idx, kern in enumerate(filters):
		fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
		# cv2.imwrite('out/gabor_%02d.png'%idx, fimg)
		np.maximum(accum, fimg, accum)
	return accum

if __name__ == '__main__':
	import os, sys

	filters = build_filters()

	img_list = os.listdir('input')

	for img_fn in img_list:

		img = cv2.cvtColor(cv2.imread('input/' + img_fn), cv2.COLOR_BGR2GRAY)

		res1 = process(img, filters)

		cv2.imwrite('out_gabor/' + img_fn, res1)

		# edges = cv2.Canny(res1,res1.max()*.8,res1.max()*.9)
		# edges = cv2.Canny(res1, 50, 200)

		hxx, hxy, hyy = hessian_matrix(res1, sigma=3)
		i1, i2 = hessian_matrix_eigvals(hxx,hxy,hyy)
		# ridges = np.maximum(np.abs(i1),np.abs(i2))
		ridges = np.abs(i2)
		ridges = (ridges - ridges.min()) / (ridges.max() - ridges.min())
		ridges = morphology.remove_small_objects(ridges>.2, 100)
		ridges = (ridges * 255).astype(np.uint8)

		cv2.imwrite('out_ridges/' + img_fn, ridges)
		# cv2.imwrite('out/' + img_fn, edges)

		# cv2.imshow('result', res1)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()