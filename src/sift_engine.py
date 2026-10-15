"""
SIFT Feature Extraction & Landmark Indexing Engine
Author: Bilal Butt (@TheBilalButt)
Course: Computer Vision - Lab 06
"""

import os
import glob
import cv2
import numpy as np

class SIFTEngine:
    def __init__(self, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma
        )
        self.database = {}

    def extract_features(self, image_bgr):
        """Converts image to grayscale and extracts SIFT keypoints and 128-D descriptors."""
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return gray, keypoints, descriptors

    def index_reference_database(self, landmarks_dir):
        """Task 1 & 2: Traverses landmark directories, extracts SIFT features, and indexes them."""
        self.database = {}
        landmark_folders = [d for d in os.listdir(landmarks_dir) 
                            if os.path.isdir(os.path.join(landmarks_dir, d)) and d != 'unknown_landmark']

        for lm_name in sorted(landmark_folders):
            lm_path = os.path.join(landmarks_dir, lm_name)
            img_files = sorted(glob.glob(os.path.join(lm_path, "*.jpg")) + glob.glob(os.path.join(lm_path, "*.png")))
            self.database[lm_name] = []
            
            for fpath in img_files:
                bgr = cv2.imread(fpath)
                if bgr is None:
                    continue
                gray, kps, descs = self.extract_features(bgr)
                self.database[lm_name].append({
                    'filename': os.path.basename(fpath),
                    'path': fpath,
                    'bgr': bgr,
                    'gray': gray,
                    'keypoints': kps,
                    'descriptors': descs,
                    'shape': gray.shape
                })
        return self.database

    def draw_keypoints(self, image_bgr, keypoints, rich=True):
        """Draws keypoints showing scale and orientation."""
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich else cv2.DRAW_MATCHES_FLAGS_DEFAULT
        return cv2.drawKeypoints(image_bgr, keypoints, None, color=(0, 255, 0), flags=flags)
