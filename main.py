import sys
import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
from PIL import Image
from skimage.metrics import structural_similarity
import matplotlib.colors as mcolors

ref_filepath = sys.argv[1]
test_filepath = sys.argv[2]
max_shift = int(sys.argv[3])
interval = int(sys.argv[4])
n_roi = int(sys.argv[5])
roi_type = sys.argv[6]
roi_halfwidth = int(sys.argv[7])
do_nps = sys.argv[8]
px_size = float(sys.argv[9])
save_filepath = sys.argv[10]

if roi_type == "circle" or roi_type == "Circle" or roi_type == "circ":
	roi = "circle"
elif roi_type == "square" or roi_type == "Square" or roi_type == "sq":
	roi = "square"
else:
	print("Invalid ROI type (options are square or circle).")
	exit()	
	
class projection:
	def __init__(self, dataset, index):
		self.projectionData = dataset[index]
		self.sliceName = (dataset[index].split("/")[-1]).split(".")[0]
		self.displayText = "Image"
		self.scaleDisp = 1

	def read(self):
		self.data = cv2.imread(self.projectionData, flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
		self.data16 = self.data.astype('uint16')
		self.n_rows, self.n_cols = np.shape(self.data)
		
		pos_img = self.data - np.amin(self.data)
		scale_img = (pos_img / float(np.amax(pos_img))) * 255.

		self.disp_rows = int(self.scaleDisp*self.n_rows)
		self.disp_cols = int(self.scaleDisp*self.n_cols)
		
		self.disp_img = (scale_img).astype('uint8')
		self.disp_img = cv2.cvtColor(self.disp_img, cv2.COLOR_GRAY2RGB)
		self.disp_img = cv2.resize(self.disp_img, (self.disp_cols, self.disp_rows))
		
		self.mkup_img = self.disp_img
		
	def image_markup(self, label, pos, halfwidth, line_color, line_thickness):
		if roi == "circle":
			self.mkup_img = cv2.circle(self.mkup_img, pos, radius=halfwidth, color=line_color, thickness=line_thickness) 
		elif roi == "square":
			startpoint = (pos[0]-halfwidth, pos[1]-halfwidth)
			endpoint = (pos[0]+halfwidth, pos[1]+halfwidth)
			self.mkup_img = cv2.rectangle(self.mkup_img, startpoint, endpoint, color=line_color, thickness=line_thickness)

			
		self.mkup_img = cv2.putText(self.mkup_img, label, (pos[0]+halfwidth, pos[1]+halfwidth), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=line_color, thickness=line_thickness)

	def display(self):
		cv2.imshow(self.displayText, self.disp_img)
		
def find_shift(fixed, moving):
	print("Calculating shift between the slices in reference and test images.")
	max_score = 0
	shift = 0
	ref = projection(fixed, 0)
	ref.read()
	for i in range(max_shift):
		test = projection(moving, i)
		test.read()
		score, diff = structural_similarity(ref.data, test.data, full=True)
		if score > max_score:
			max_score = score
			shift = i
	
	print("A shift of " + str(shift) + " slices was found." )
	return shift
	

class dataFrame:
	def __init__(self):
		self.columnNames = ["refSlice", "testSlice", "roiID", "roiCentre", "refMean", "refSD", "testmean", "testSD"]
		self.table = pd.DataFrame(columns=self.columnNames)
		self.idx = 0
		
	
	def write_line(self, data_as_list):
		new_line = pd.DataFrame(data=[data_as_list], columns=self.columnNames, index=[self.idx])
		
		self.table = pd.concat([self.table, new_line])
		
		
	def save_as_excel(self, name):
		self.table.to_excel(name+".xlsx")


def select_pixel(event, x, y, flags, params):
	global c1, c2, c3
	if event == cv2.EVENT_LBUTTONDOWN:
		roi_x.append(x)
		roi_y.append(y)
		
		c1 = randint(0, 255)
		c2 = randint(0, 255)
		c3 = randint(0, 255)
		
def nps_calc(roi_raw, roi_mean, r_sp_freq):
	# For each ROI:
	# Get mean value
	# Subtract mean value from raw values
	# DFT the result
	# Square the result
	# Bin the data according to radial spatial frequency
	# Average the data in each bin
	# Multiply by ratio of pixel area to ROI area
	# Then (outside this function):
	# Average over all ROIs
			
	g = roi_raw - roi_mean
	fft_g = np.fft.fft2(g)
	fft_g_sq = np.power(np.absolute(fft_g), 2)
	
	dim = roi_raw.shape[0]
	nbins = int(dim/2)
	
	bin_sum, bin_edges = np.histogram(r_sp_freq, bins=nbins, density=False, weights=fft_g_sq)
	bin_counts, bin_edges = np.histogram(r_sp_freq, bins=nbins, density=False, weights=None)
	bin_ave = bin_sum / bin_counts
	bin_centres = [(bin_edges[x] + bin_edges[x+1])/2 for x in range(len(bin_edges)-1)]
	bin_nps = (px_size**2 / dim**2) * bin_ave # units: rad^2 nm^2
	
	return bin_centres, bin_nps
			
	

"""
======================== MAIN PROGRAM ==================================
"""		

if not os.path.exists(save_filepath):
	os.makedirs(save_filepath)
		
ref_shift = sorted(glob.glob(ref_filepath + "*"))
test_shift = sorted(glob.glob(test_filepath + "*"))
shift = find_shift(ref_shift, test_shift)

ref_images = sorted(glob.glob(ref_filepath + "*"))[0::interval]
test_images = sorted(glob.glob(test_filepath + "*"))[0+shift::interval]

scale_display = 0.75

if len(ref_images) != len(test_images):
	print("ERROR: number of test images must be the same as the number of reference images.")
	exit()
	
else:
	n_slice = len(ref_images)
	
list_of_colours = list(mcolors.TABLEAU_COLORS.keys())
	
results = dataFrame()

for n in range(n_slice):
	roi_x = []
	roi_y = []
	
	ref_img = projection(ref_images, n)
	ref_img.scaleDisp = scale_display
	ref_img.read()
	ref_img.displayText = "Select " + str(n_roi) + " ROI centres."
	
	test_img = projection(test_images, n)
	test_img.scaleDisp = scale_display
	test_img.read()
	
	displayFlag = True
	
	while displayFlag == True:
		if len(roi_x) > 0:
			rgb = (np.asarray(mcolors.to_rgb(mcolors.TABLEAU_COLORS[list_of_colours[len(roi_x)-1]]))*255).astype(int)
			ref_img.image_markup(str(len(roi_x)-1), (roi_x[-1],roi_y[-1]), int(roi_halfwidth*scale_display), (int(rgb[2]), int(rgb[1]), int(rgb[0])), 2)
			test_img.image_markup(str(len(roi_x)-1), (roi_x[-1],roi_y[-1]), int(roi_halfwidth*scale_display), (int(rgb[2]), int(rgb[1]), int(rgb[0])), 2)
		ref_img.display()
		cv2.setMouseCallback(ref_img.displayText, select_pixel)
		cv2.waitKey(1)
			
		if len(roi_x) == n_roi:
			cv2.destroyAllWindows()
			displayFlag = False
			
	rgb = (np.asarray(mcolors.to_rgb(mcolors.TABLEAU_COLORS[list_of_colours[len(roi_x)-1]]))*255).astype(int)
	ref_img.image_markup(str(len(roi_x)-1), (roi_x[-1],roi_y[-1]), int(roi_halfwidth*scale_display), (int(rgb[2]), int(rgb[1]), int(rgb[0])), 2)
	test_img.image_markup(str(len(roi_x)-1), (roi_x[-1],roi_y[-1]), int(roi_halfwidth*scale_display), (int(rgb[2]), int(rgb[1]), int(rgb[0])), 2)
	
	joined_disp = np.concatenate((ref_img.mkup_img, test_img.mkup_img), axis=1)
	joined_disp = cv2.resize(joined_disp, (2*ref_img.disp_cols, ref_img.disp_rows))
	cv2.imshow("Displaying ROIs on reference image (left) and test image (right). Press <<ENTER>> to confirm.", joined_disp)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	save_figure_path = save_filepath + "/figures/"
	
	if not os.path.exists(save_figure_path):
		os.makedirs(save_figure_path)
	
	save_img_name = save_figure_path + "ref_" + ref_img.sliceName + "_test_" + test_img.sliceName + ".tiff"
	cv2.imwrite(save_img_name, joined_disp)
	
	roi_x = [int(x/scale_display) for x in roi_x]
	roi_y = [int(y/scale_display) for y in roi_y]
	
	for idx in range(n_roi):
		x0 = roi_x[idx]
		y0 = roi_y[idx]
		
		if roi == "circle":
			ref_values = []
			test_values = []
			
			radius_roi = roi_halfwidth
		
			for i in range(y0 - radius_roi, y0 + radius_roi):
				for j in range(x0 - radius_roi, x0 + radius_roi):
					if np.sqrt((j - x0)**2 + (i - y0)**2) <= radius_roi:
						ref_values.append(ref_img.data[i, j])
						test_values.append(test_img.data[i, j])
		elif roi == "square":
			ref_values = ref_img.data[y0-roi_halfwidth:y0+roi_halfwidth, x0-roi_halfwidth:x0+roi_halfwidth]
			test_values = test_img.data[y0-roi_halfwidth:y0+roi_halfwidth, x0-roi_halfwidth:x0+roi_halfwidth]
			
					
		ref_mean = np.mean(ref_values)
		ref_sd = np.std(ref_values)
		test_mean = np.mean(test_values)
		test_sd = np.std(test_values)
		
		if do_nps:
			if idx==0:
				print("Calculating NPS.")
				fig=plt.figure()
				ref_plt = fig.gca()
				test_plt = fig.gca()
				
				sp_freq = np.fft.fftfreq(ref_values.shape[0], px_size)
				df = sp_freq[1] - sp_freq[0]
				sp_freq = sp_freq + df/2
	
				r = np.zeros(ref_values.shape)
	
				for i in range(len(sp_freq)):
					for j in range(len(sp_freq)):
						yy = sp_freq[i]
						xx = sp_freq[j]
						r[i, j] = np.sqrt(xx**2 + yy**2)
		
			ref_sp_freq, ref_noise_power = nps_calc(ref_values, ref_mean, r)
			test_sp_freq, test_noise_power = nps_calc(test_values, test_mean, r)
			
			roi_colour = list_of_colours[idx]
			
			ref_plt.plot(ref_sp_freq, ref_noise_power, color=roi_colour, linestyle='--', label="Ref ROI "+str(idx))
			test_plt.plot(test_sp_freq, test_noise_power, color=roi_colour, linestyle='-', label="Test ROI "+str(idx))
			
			if idx==n_roi-1:
				plt.xlabel(r"Spatial frequency (nm$^{-1}$)")
				plt.ylabel(r"Noise power (rad$^{2}$ nm$^{2}$)")
				plt.savefig(save_filepath + "/figures/nps_" + "ref_" + ref_img.sliceName + "_test_" + test_img.sliceName + ".png")
		
		data_to_write = [ref_img.sliceName, test_img.sliceName, idx, (roi_x[idx], roi_y[idx]), ref_mean, ref_sd, test_mean, test_sd]
		results.idx = (n * n_roi) + idx
		results.write_line(data_to_write)
		
results.save_as_excel(save_filepath+"/results")


