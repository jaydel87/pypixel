import sys
import os
import glob
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from random import randint
from PIL import Image
from skimage.metrics import structural_similarity
from itertools import product

parser = argparse.ArgumentParser(description="Quantitative ROI-based comparison of two similar images")
parser.add_argument('ref_filepath', help="Required: Path to the reference image.")
parser.add_argument('test_filepath', help="Required: Path to the test image.")
parser.add_argument('save_filepath', help="Required: Path where data is to be saved.")
parser.add_argument('--no_alignZ', action='store_true', default=False, help="Skip z-alignment (orthogonal to slice plane) of test dataset to reference dataset.")
parser.add_argument('--no_alignXY', action='store_true', default=False, help="Skip xy-alignment (in slice place) of test dataset to reference dataset.")
parser.add_argument('--max_shift', default=[30, 5], nargs=2, type=int, help="Maximum shift allowed in (slice, xy plane) for image alignment. Default value is (30, 10).")
parser.add_argument('--interval', default=1, type=int, help="For interval=x, image comparison will be performed for every x slices in the dataset. Default value is 10.")
parser.add_argument('--slices', nargs='*', action='store', type=int, help="Optional. Provide specific slices to analyse. Overrides --intervals if used.")
parser.add_argument('--n_roi', default=5, type=int, help="Number of regions of interest to be selected. Default value is 1 i.e., every image will be treated.")
parser.add_argument('--roi_shape', choices=['square', 'circle'], default='square', help="The shape of the region of interest to be selected. Available options: square or circle.")
parser.add_argument('--roi_halfwidth', default=10, type=int, help="The half-width or radius of the region of interest in pixels. Default value is 10.")
parser.add_argument('--no_nps', action='store_true', default=False, help="Skip calculation of noise power spectrum.")
parser.add_argument('--px_size', default=90, type=float, help="The size, in nm, of the image pixels. Default value is 90 nm. Value is the same for both images.")
params = parser.parse_args()

if params.roi_shape == "circle" or params.roi_shape == "Circle" or params.roi_shape == "c" or params.roi_shape=="C":
	roi = "circle"
elif params.roi_shape == "square" or params.roi_shape == "Square" or params.roi_shape == "s" or params.roi_shape=="S":
	roi = "square"
else:
	print("Invalid ROI type (options are square or circle).")
	exit()	
	
if params.slices != None:
	slice_idx = [x-1 for x in params.slices]
	
class projection:
	def __init__(self, dataset, index):
		self.projectionData = dataset[index]
		self.sliceName = (dataset[index].split("/")[-1]).split(".")[0]
		self.displayText = "Image"
		self.scaleDisp = 1

	def read(self):
		self.data = cv2.imread(self.projectionData, flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
		self.n_rows, self.n_cols = np.shape(self.data)
		
	def pad(self, pad_size, pad_value):
		return cv2.copyMakeBorder(self.data, pad_size, pad_size, pad_size, pad_size, borderType=cv2.BORDER_CONSTANT, value=pad_value)
			
	def translate(self, px_horizontal, px_vertical):
		transl_mat = np.float32([[1, 0, px_horizontal], [0, 1, px_vertical]])
		return cv2.warpAffine(self.data, transl_mat, (self.data.shape[1], self.data.shape[0])) 
		
	def convertDisplay(self):	
		pos_img = self.data - np.amin(self.data)
		scale_img = (pos_img / float(np.amax(pos_img))) * 255.

		self.disp_rows = int(self.scaleDisp*self.n_rows)
		self.disp_cols = int(self.scaleDisp*self.n_cols)
		
		self.disp_img = (scale_img).astype('uint8')
		self.disp_img = cv2.cvtColor(self.disp_img, cv2.COLOR_GRAY2RGB)
		self.disp_img = cv2.resize(self.disp_img, (self.disp_cols, self.disp_rows))
		
		self.mkup_img = self.disp_img
		
	def draw_roi(self, label, pos, halfwidth, line_color, line_thickness):		
		if roi == "circle":
			self.mkup_img = cv2.circle(self.mkup_img, pos, radius=halfwidth, color=line_color, thickness=line_thickness) 
		elif roi == "square":
			startpoint = (pos[0]-halfwidth, pos[1]-halfwidth)
			endpoint = (pos[0]+halfwidth, pos[1]+halfwidth)
			self.mkup_img = cv2.rectangle(self.mkup_img, startpoint, endpoint, color=line_color, thickness=line_thickness)

			
		self.mkup_img = cv2.putText(self.mkup_img, label, (pos[0]+halfwidth, pos[1]+halfwidth), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=line_color, thickness=line_thickness)

	def display(self):
		cv2.imshow(self.displayText, self.disp_img)
		
def z_shift(fixed, moving):
	print("Calculating shift between the slices in reference and test images.")
	max_score = 0
	shift = 0
	ref = projection(fixed, 0)
	ref.read()
	for i in range(params.max_shift[0]):
		test = projection(moving, i)
		test.read()
		score, diff = structural_similarity(ref.data, test.data, full=True)
		if score > max_score:
			max_score = score
			shift = i
	
	print("A shift of " + str(shift) + " slices was found." )
	return shift
	

def xy_shift(fixed, moving):
	print("Aligning test image to reference image.")
	max_score = 0
	shift = 0
	
	for tr_horiz, tr_vert in product(range(-params.max_shift[1],params.max_shift[1]+1), range(-params.max_shift[1],params.max_shift[1]+1)):
		score, diff = structural_similarity(fixed.data, moving.translate(tr_horiz, tr_vert), full=True)
		if score > max_score:
			max_score=score
			shift = tr_horiz, tr_vert
			aligned_image = moving.translate(tr_horiz, tr_vert)
	
	print("A shift of " + str(shift) + " pixels was found.")			
	return aligned_image

	

class dataFrame:
	def __init__(self):
		if not params.no_nps:
			self.columnNames = ["refSlice", "testSlice", "roiID", "roiCentre", "refMean", "refSD", "testMean", "testSD", "radialSpFreq", "refNPS", "testNPS"]
		else:
			self.columnNames = ["refSlice", "testSlice", "roiID", "roiCentre", "refMean", "refSD", "testMean", "testSD"]
		self.table = pd.DataFrame(columns=self.columnNames)
		self.idx = 0
		
	
	def write_line(self, data_as_list):
		new_line = pd.DataFrame(data=[data_as_list], columns=self.columnNames, index=[self.idx])
		
		self.table = pd.concat([self.table, new_line])
		
		
	def save_as_excel(self, name):
		self.table.to_excel(name+".xlsx")


def select_pixel(event, x, y, flags, params):
	if event == cv2.EVENT_LBUTTONDOWN:
		roi_x.append(x)
		roi_y.append(y)
		
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
	bin_nps = (params.px_size**2 / dim**2) * bin_ave # units: rad^2 nm^2
	
	return bin_centres, bin_nps
			
	

"""
======================== MAIN PROGRAM ==================================
"""		

if not os.path.exists(params.save_filepath):
	os.makedirs(params.save_filepath)
		
ref_shift = sorted(glob.glob(params.ref_filepath + "*"))
test_shift = sorted(glob.glob(params.test_filepath + "*"))

if not params.no_alignZ:
	slice_shift = z_shift(ref_shift, test_shift)
	if params.slices == None:
		ref_images = sorted(glob.glob(params.ref_filepath + "*"))[0:-slice_shift:params.interval]
		test_images = sorted(glob.glob(params.test_filepath + "*"))[0+slice_shift::params.interval]
	else:
		shifted_idx = [x + slice_shift for x in slice_idx]
		ref_images = [sorted(glob.glob(params.ref_filepath + "*"))[s] for s in slice_idx]
		test_images = [sorted(glob.glob(params.test_filepath + "*"))[s] for s in shifted_idx]
	
else:
	if params.slices == None:
		ref_images = sorted(glob.glob(params.ref_filepath + "*"))[0::params.interval]
		test_images = sorted(glob.glob(params.test_filepath + "*"))[0::params.interval]
	else:
		ref_images = [sorted(glob.glob(params.ref_filepath + "*"))[s] for s in slice_idx]
		test_images = [sorted(glob.glob(params.test_filepath + "*"))[s] for s in slice_idx]

scale_display = 0.75

if len(ref_images) != len(test_images):
	print("ERROR: number of test images must be the same as the number of reference images.")
	exit()
	
else:
	n_slice = len(ref_images)
	
list_of_colours = list(mcolors.TABLEAU_COLORS.keys())
	
results = dataFrame()

for n in range(n_slice):	
	print("\nImage " + str(n+1) + " of " + str(n_slice) + ".")
	roi_x = []
	roi_y = []
	
	ref_img = projection(ref_images, n)
	ref_img.scaleDisp = scale_display
	ref_img.read()
	ref_img.convertDisplay()
	ref_img.displayText = "Select " + str(params.n_roi) + " ROI centres."
	
	test_img = projection(test_images, n)
	test_img.scaleDisp = scale_display
	test_img.read()
	
	if not params.no_alignXY:
		test_img.data = xy_shift(ref_img, test_img)
	test_img.convertDisplay()
	
	displayFlag = True
	
	while displayFlag == True:
		if len(roi_x) > 0:
			rgb = (np.asarray(mcolors.to_rgb(mcolors.TABLEAU_COLORS[list_of_colours[len(roi_x)-1]]))*255).astype(int)
			ref_img.draw_roi(str(len(roi_x)-1), (roi_x[-1],roi_y[-1]), int(params.roi_halfwidth*scale_display), (int(rgb[2]), int(rgb[1]), int(rgb[0])), 2)
			test_img.draw_roi(str(len(roi_x)-1), (roi_x[-1],roi_y[-1]), int(params.roi_halfwidth*scale_display), (int(rgb[2]), int(rgb[1]), int(rgb[0])), 2)
		ref_img.display()
		cv2.setMouseCallback(ref_img.displayText, select_pixel)
		cv2.waitKey(1)
			
		if len(roi_x) == params.n_roi:
			cv2.destroyAllWindows()
			displayFlag = False
			
	rgb = (np.asarray(mcolors.to_rgb(mcolors.TABLEAU_COLORS[list_of_colours[len(roi_x)-1]]))*255).astype(int)
	ref_img.draw_roi(str(len(roi_x)-1), (roi_x[-1],roi_y[-1]), int(params.roi_halfwidth*scale_display), (int(rgb[2]), int(rgb[1]), int(rgb[0])), 2)
	test_img.draw_roi(str(len(roi_x)-1), (roi_x[-1],roi_y[-1]), int(params.roi_halfwidth*scale_display), (int(rgb[2]), int(rgb[1]), int(rgb[0])), 2)
	
	joined_disp = np.concatenate((ref_img.mkup_img, test_img.mkup_img), axis=1)
	joined_disp = cv2.resize(joined_disp, (2*ref_img.disp_cols, ref_img.disp_rows))
	cv2.imshow("Displaying ROIs on reference image (left) and test image (right). Press <<ENTER>> to confirm.", joined_disp)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	save_figure_path = params.save_filepath + "/figures/"
	
	if not os.path.exists(save_figure_path):
		os.makedirs(save_figure_path)
	
	save_img_name = save_figure_path + "ref_" + ref_img.sliceName + "_test_" + test_img.sliceName + ".tiff"
	cv2.imwrite(save_img_name, joined_disp)
	
	roi_x = [int(x/scale_display) for x in roi_x]
	roi_y = [int(y/scale_display) for y in roi_y]
	
	for idx in range(params.n_roi):
		x0 = roi_x[idx]
		y0 = roi_y[idx]
		
		if roi == "circle":
			ref_values = []
			test_values = []
			
			radius_roi = params.roi_halfwidth
		
			for i, j in product(range(y0 - radius_roi, y0 + radius_roi), range(x0 - radius_roi, x0 + radius_roi)): 
				if np.sqrt((j - x0)**2 + (i - y0)**2) <= radius_roi:
					ref_values.append(ref_img.data[i, j])
					test_values.append(test_img.data[i, j])
		elif roi == "square":
			ref_values = ref_img.data[y0-params.roi_halfwidth:y0+params.roi_halfwidth, x0-params.roi_halfwidth:x0+params.roi_halfwidth]
			test_values = test_img.data[y0-params.roi_halfwidth:y0+params.roi_halfwidth, x0-params.roi_halfwidth:x0+params.roi_halfwidth]
			
					
		ref_mean = np.mean(ref_values)
		ref_sd = np.std(ref_values)
		test_mean = np.mean(test_values)
		test_sd = np.std(test_values)
		
		if not params.no_nps:
			if idx==0:
				print("Calculating NPS.")
				fig=plt.figure()
				ref_plt = fig.gca()
				test_plt = fig.gca()
				
				if n==0 and idx == 0:
					sp_freq = np.fft.fftfreq(ref_values.shape[0], params.px_size)
					df = sp_freq[1] - sp_freq[0]
					sp_freq = sp_freq + df/2
	
					r = np.zeros(ref_values.shape)
	
					for i, j in product(range(len(sp_freq)), range(len(sp_freq))):
						yy = sp_freq[i]
						xx = sp_freq[j]
						r[i, j] = np.sqrt(xx**2 + yy**2)
		
			sp_freq, ref_noise_power = nps_calc(ref_values, ref_mean, r)
			sp_freq, test_noise_power = nps_calc(test_values, test_mean, r)
			
			roi_colour = list_of_colours[idx]
			
			ref_plt.plot(sp_freq, ref_noise_power, color=roi_colour, linestyle='--', label="Ref ROI "+str(idx))
			test_plt.plot(sp_freq, test_noise_power, color=roi_colour, linestyle='-', label="Test ROI "+str(idx))
			
			if idx==params.n_roi-1:
				plt.xlabel(r"Spatial frequency (nm$^{-1}$)")
				plt.ylabel(r"Noise power (rad$^{2}$ nm$^{2}$)")
				plt.savefig(params.save_filepath + "/figures/nps_" + "ref_" + ref_img.sliceName + "_test_" + test_img.sliceName + ".png")
				
			data_to_write = [ref_img.sliceName, test_img.sliceName, idx, (roi_x[idx], roi_y[idx]), ref_mean, ref_sd, test_mean, test_sd, sp_freq, list(ref_noise_power), list(test_noise_power)]
		
		else:
			data_to_write = [ref_img.sliceName, test_img.sliceName, idx, (roi_x[idx], roi_y[idx]), ref_mean, ref_sd, test_mean, test_sd]
		
		results.idx = (n * params.n_roi) + idx
		results.write_line(data_to_write)
		
results.save_as_excel(params.save_filepath+"/results")


