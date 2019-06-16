# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2
import os
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--watermark", required=True,
	help="path to watermark image (assumed to be transparent PNG)")
ap.add_argument("-i", "--input", required=True,
	help="path to the input directory of images")
ap.add_argument("-o", "--output", required=True,
	help="path to the output directory")
ap.add_argument("-a", "--alpha", type=float, default=0.25,
	help="alpha transparency of the overlay (smaller is more transparent)")
ap.add_argument("-c", "--correct", type=int, default=1,
	help="flag used to handle if bug is displayed or not")
args = vars(ap.parse_args())