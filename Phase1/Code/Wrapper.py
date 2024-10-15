#!/usr/bin/env python3

"""
RBE/CS549 Spring 2024: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def save_image_grid(images, row_num, col_num, path, cmap='gray'):
    # Create a figure
	fig, axes = plt.subplots(row_num, col_num, figsize=(20, 6))
	axes = axes.flatten()

	# Plot each filter
	for i, ax in enumerate(axes):
		ax.imshow(images[i], cmap=cmap)
		ax.axis('off')  # Turn off axis labels

	plt.tight_layout()
	plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
	plt.show()

def rotate_image(image, angle):
  row, col = image.shape[0], image.shape[1]
  rot_mat = cv2.getRotationMatrix2D((col/2, row/2), angle, 1.0)
  rot_mat[0, 2] += (rot_mat[0, 0] + rot_mat[0, 1] - 1) / 2
  rot_mat[1, 2] += (rot_mat[1, 0] + rot_mat[1, 1] - 1) / 2
  result = cv2.warpAffine(image, rot_mat, (col, row))
  return result

def Gauss2d(filter_size, sigma):
	y, x = np.mgrid[-filter_size//2 + 1:filter_size//2 + 1, -filter_size//2 + 1:filter_size//2 + 1]
	
	kernel = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
	kernel /= np.sum(np.abs(kernel))
	# kernel = kernel * (0.5/(np.pi*sigma* sigma))
	return kernel

def DoG_filterbank(scales, num_orients, filter_size):
	filterbank = []
	orients = np.arange(0, 360, 360/num_orients)
	
	sobel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
	sobel_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
	
	for i in range(len(scales)):
		for j in range(orients.shape[0]):
			Gauss_kernel = Gauss2d(filter_size, scales[i])
			
			Gx = cv2.filter2D(Gauss_kernel,-1, sobel_x)
			Gy = cv2.filter2D(Gauss_kernel,-1, sobel_y)
			
			filterbank.append(Gx*np.cos(np.deg2rad(orients[j])) + Gy*np.sin(np.deg2rad(orients[j])))

	return filterbank

def first_derivative_gaussian(sigma, theta, e, filter_size):
    if filter_size % 2 == 0:
        filter_size += 1  # Ensure odd filter_size for symmetry
    
    half_size = filter_size // 2
    y, x = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
    # Rotate coordinates
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    # Apply elongation
    sigma_x = sigma
    sigma_y = sigma * e
    # Gaussian function
    gauss = np.exp(-(x_theta**2 / (2 * sigma_x**2) + y_theta**2 / (2 * sigma_y**2)))
    # Derivative in x direction (aligned with theta)
    derivative = -x_theta / (sigma_x**2) - y_theta / (sigma_y**2) 
    # Combine Gaussian and derivative
    kernel = derivative * gauss
    # Normalize the kernel
    kernel /= np.sum(np.abs(kernel))
    return kernel

def second_derivative_gaussian(sigma, theta, e, filter_size):
    if filter_size % 2 == 0:
        filter_size += 1  # Ensure odd filter_size for symmetry
    
    half_size = filter_size // 2
    y, x = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
    # Rotate coordinates
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    # Apply elongation
    sigma_x = sigma
    sigma_y = sigma * e
    # Gaussian function
    gauss = np.exp(-(x_theta**2 / (2 * sigma_x**2) + y_theta**2 / (2 * sigma_y**2)))
    # Derivative
    derivative = ((x_theta**2-sigma_x**2) / (sigma_x**4) + (y_theta**2-sigma_y**2) / (sigma_y**4))
    # Combine Gaussian and derivative
    kernel = derivative * gauss
    # Normalize the kernel
    kernel /= np.sum(np.abs(kernel))
    return kernel

def LoG(sigma, filter_size):
	if filter_size % 2 == 0:
		filter_size += 1  # Ensure odd filter_size for symmetry
	
	half_size = filter_size // 2
	x, y = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
	# Gaussian function
	gauss = np.exp(-(x**2 / (2 * sigma**2) + y**2 / (2 * sigma**2)))
    # Derivative 
	derivative = ((x**2-sigma**2) / (sigma**4) + (y**2-sigma**2) / (sigma**4))
    # Combine Gaussian and derivative
	kernel = derivative * gauss
    # Normalize the kernel
	kernel /= np.sum(np.abs(kernel))
	return kernel

def gabor_filter(sigma, theta, gamma, Lambda, psi, filter_size):
    if filter_size % 2 == 0:
        filter_size += 1  # Ensure odd filter_size for symmetry
    
    half_size = filter_size // 2
    y, x = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
    # Rotate coordinates
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    # Apply elongation
    sigma_x = sigma
    sigma_y = sigma / gamma
    # Gaussian envelope
    gaussian = np.exp(-(x_theta**2 / (2 * sigma_x**2) + y_theta**2 / (2 * sigma_y**2)))
    # Sinusoidal carrier
    sinusoid = np.cos(2*np.pi/Lambda * x_theta + psi)
    # Combine Gaussian envelope and sinusoidal carrier
    kernel = gaussian * sinusoid
    return kernel

def LM_bank(scales, num_orients, filter_size):
	filter_bank = []
	orients = np.deg2rad(np.arange(0, 360, 360/num_orients))
	for i in range(3): # First and Second derivative of Gaussian (first 3 scales)
		for j in range(orients.shape[0]):
			filter_bank.append(first_derivative_gaussian(scales[i],orients[j],3,filter_size))
		for k in range(orients.shape[0]):
			filter_bank.append(second_derivative_gaussian(scales[i],orients[k],3,filter_size))	
	# Laplacian of Gaussian
	for i in range(len(scales)):  
		filter_bank.append(LoG(scales[i],filter_size))
	for i in range(len(scales)):
		filter_bank.append(LoG(3*scales[i],filter_size))
	# Gaussian
	for i in range(len(scales)):
		filter_bank.append(Gauss2d(filter_size,scales[i]))
	return filter_bank

def gabor_bank(scales, num_orients, gamma, Lambda, psi, filter_size):
	filter_bank = []
	orients = np.deg2rad(np.arange(0, 360, 360/num_orients))

	for i in range(len(scales)):
		for j in range(orients.shape[0]):
			for k in range(len(gamma)):
				filter_bank.append(gabor_filter(scales[i], orients[j], gamma[k], Lambda, psi, filter_size))
	return filter_bank

#########Half Disc Masks######################################

def half_disc(mask_size, theta):
	half_size = mask_size // 2
	y, x = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]

	kernel = np.zeros((mask_size,mask_size))
	for col in range(mask_size):
		if col < mask_size // 2:
			for row in range(mask_size):
				if x[row][col]**2 + y[row][col]**2 <= (mask_size/2)*(mask_size/2):
					kernel[row][col] = 1
	
	rotated_kernel = rotate_image(kernel, theta)
	rotated_kernel[rotated_kernel>0] = 1.0
	return rotated_kernel

def half_disc_mask_bank(scales, num_orients):
	filter_bank = []
	orients = np.arange(0, 180, 180/num_orients)
	
	for i in range(len(scales)):
		for j in range(orients.shape[0]):
			filter_bank.append(half_disc(scales[i], orients[j]))
			filter_bank.append(half_disc(scales[i], 180+orients[j]))
	return filter_bank

############Maps############################################

def texton_map(filter_bank, image, num_clusters, num_init):
	maps = []
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	for n in range(len(filter_bank)):
		filtered_image = cv2.filter2D(gray_image, -1, filter_bank[n])
		maps.append(filtered_image)
	
	maps = np.array(maps)
	maps = maps.reshape(maps.shape[0], image.shape[0]*image.shape[1]).transpose()
	# maps = maps.reshape(image.shape[0]*image.shape[1], maps.shape[0])
	km = KMeans(n_clusters=num_clusters, n_init=num_init).fit(maps)
	t_map = km.labels_.reshape(image.shape[0], image.shape[1])
	return t_map

def brightness_map(image, num_clusters, num_init):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	maps = gray_image.reshape(1, image.shape[0]*image.shape[1]).transpose()
	km = KMeans(n_clusters=num_clusters, n_init=num_init).fit(maps)
	b_map = km.labels_.reshape(gray_image.shape[0], gray_image.shape[1])
	return b_map

def color_map(image, num_clusters, num_init):
	maps = image.reshape(image.shape[0]*image.shape[1], image.shape[2]) 
	km = KMeans(n_clusters=num_clusters, n_init=num_init).fit(maps)
	c_map = km.labels_.reshape(image.shape[0], image.shape[1])
	return c_map

def chisquareDist(map_input,num_bins, disc_mask_bank):
	x2_dists = []
	N = len(disc_mask_bank)
	n = 0
	while n < N:
		left_mask = disc_mask_bank[n]
		right_mask = disc_mask_bank[n+1]
		chi_sqr_dist = np.zeros(map_input.shape)
		tmp = np.zeros(map_input.shape)
		smallest_bin = np.min(map_input)

		for i in range(num_bins):
			tmp[map_input == smallest_bin + i] = 1
			g_i = cv2.filter2D(tmp, -1, left_mask)
			h_i = cv2.filter2D(tmp, -1, right_mask)
			chi_sqr_dist += (g_i - h_i)**2 / (g_i + h_i + np.exp(-6))
		chi_sqr_dist /= 2

		x2_dists.append(chi_sqr_dist)
		n = n + 2
	
	return x2_dists

def main():
	curr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

	num_imgs = 10
	input_images = []
	canny_baselines = []
	sobel_baselines = []

	for i in range(num_imgs):
		input_images.append(cv2.imread(os.path.join(curr_dir,"BSDS500/Images/" + str(i+1) + ".jpg")))
		canny_baselines.append(cv2.cvtColor(cv2.imread(os.path.join(curr_dir,"BSDS500/CannyBaseline/" + str(i+1) + ".png")), cv2.COLOR_BGR2GRAY))
		sobel_baselines.append(cv2.cvtColor(cv2.imread(os.path.join(curr_dir,"BSDS500/SobelBaseline/" + str(i+1) + ".png")), cv2.COLOR_BGR2GRAY))

	
	# Project parameters
	DoG_filter_path = os.path.join(curr_dir, "results/filter_banks/DoG.png")
	DoG_scales = [2, 3]
	DoG_num_orients = 16
	DoG_filter_size = 49

	LM_small_path = os.path.join(curr_dir, "results/filter_banks/LMS.png")
	LM_small_scales = [1, np.sqrt(2), 2, 2*np.sqrt(2)]
	LM_small_num_orients = 6
	LM_small_filtersize = 49

	LM_large_path = os.path.join(curr_dir, "results/filter_banks/LML.png")
	LM_large_scales = [np.sqrt(2), 2, 2*np.sqrt(2), 4]
	LM_large_num_orients = 6
	LM_large_filtersize = 49

	gabor_path = os.path.join(curr_dir, "results/filter_banks/gabor.png")
	gabor_scales = [10, 24]
	gabor_num_orients = 6
	gabor_gamma = [2, 3, 4]
	gabor_Lambda = 30
	gabor_psi = 0
	gabor_filter_size = 49

	hd_path = os.path.join(curr_dir, "results/half_disc_mask/hd_masks.png")
	hd_scales = [2,5,10,20,30]
	hd_num_orients = 8

	t_map_path = os.path.join(curr_dir, "results/t_map/")
	t_map_num_clusters = 64
	t_map_num_init = 2

	b_map_path = os.path.join(curr_dir, "results/b_map/")
	b_map_num_clusters = 16
	b_map_num_init = 2

	c_map_path = os.path.join(curr_dir, "results/c_map/")
	c_map_num_clusters = 16
	c_map_num_init = 4

	T_g_path = os.path.join(curr_dir, "results/T_g/")
	B_g_path = os.path.join(curr_dir, "results/B_g/")
	C_g_path = os.path.join(curr_dir, "results/C_g/")

	pb_path = os.path.join(curr_dir, "results/pb_lite_output/")
	weights = [0.5, 0.5] # pb lite weights

	# Create filters for the filter bank
	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	"""
	DoG_bank = DoG_filterbank(DoG_scales, DoG_num_orients, DoG_filter_size)
	save_image_grid(DoG_bank, 2, 16, DoG_filter_path)

	"""
	Generate Leung-Malik Filter Bank: (LM)
	"""
	LM_small_bank = LM_bank(LM_small_scales, LM_small_num_orients, LM_small_filtersize)
	LM_large_bank = LM_bank(LM_large_scales, LM_large_num_orients, LM_large_filtersize)
	save_image_grid(LM_small_bank, 4, 12, LM_small_path)
	save_image_grid(LM_large_bank, 4, 12, LM_large_path)

	"""
	Generate Gabor Filter Bank: (Gabor)
	"""
	gabor_filter_bank = gabor_bank(gabor_scales, gabor_num_orients, gabor_gamma, gabor_Lambda, gabor_psi, gabor_filter_size)
	save_image_grid(gabor_filter_bank, 4, 9, gabor_path)

	"""
	Generate Half-disk masks
	"""
	hd_masks = half_disc_mask_bank(hd_scales, hd_num_orients)
	save_image_grid(hd_masks, 6, 8, hd_path) # Not saving the full image

	# Generating maps and gradients on all images
	combined_filter_bank = DoG_bank + LM_small_bank + LM_large_bank + gabor_filter_bank 
	for i in range(len(input_images)):
		
		t_map = texton_map(combined_filter_bank, input_images[i], t_map_num_clusters, t_map_num_init)
		plt.imsave(t_map_path + str(i+1) + ".png", t_map)
		# plt.imshow(t_map)
		# plt.show()

		b_map = brightness_map(input_images[i], b_map_num_clusters, b_map_num_init)
		plt.imsave(b_map_path + str(i+1) + ".png", b_map)
		# plt.imshow(b_map)
		# plt.show()

		c_map = brightness_map(input_images[i], c_map_num_clusters, c_map_num_init)
		plt.imsave(c_map_path + str(i+1) + ".png", c_map)
		# plt.imshow(c_map)
		# plt.show()

		# Gradients
		T_g = chisquareDist(t_map, t_map_num_clusters, hd_masks)
		T_g = np.array(T_g)
		T_g = np.mean(T_g, axis=0)
		plt.imsave(T_g_path + str(i+1) + ".png", T_g)
		# plt.imshow(T_g)
		# plt.show()

		B_g = chisquareDist(b_map, b_map_num_clusters, hd_masks)
		B_g = np.array(B_g)
		B_g = np.mean(B_g, axis=0)
		plt.imsave(B_g_path + str(i+1) + ".png", B_g)
		# plt.imshow(B_g)
		# plt.show()
		
		C_g = chisquareDist(c_map, c_map_num_clusters, hd_masks)
		C_g = np.array(C_g)
		C_g = np.mean(C_g, axis=0)
		plt.imsave(C_g_path + str(i+1) + ".png", C_g)
		# plt.imshow(C_g)
		# plt.show()

		# Final pb lite output
		mean_feature_vec = (T_g + B_g + C_g)/3
		baseline = weights[0]*canny_baselines[i] + weights[1]*sobel_baselines[i]
		pb_lite_output = np.multiply(mean_feature_vec, baseline)
		plt.imsave(pb_path + str(i+1) + ".png", pb_lite_output, cmap = "gray")
		# plt.imshow(pb_lite_output, cmap = "gray")
		# plt.show()

    
if __name__ == '__main__':
    main()
 


