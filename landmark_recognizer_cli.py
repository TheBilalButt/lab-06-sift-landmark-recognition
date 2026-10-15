"""
Interactive Landmark Recognition CLI
Author: Bilal Butt (@TheBilalButt)
Usage: python landmark_recognizer_cli.py [query_image_path]
"""

import sys
import os
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.sift_engine import SIFTEngine
from src.feature_matcher import FeatureMatcher
from src.geometric_localizer import GeometricLocalizer

def main():
    if len(sys.argv) > 1:
        query_path = sys.argv[1]
    else:
        query_path = os.path.join("data", "landmarks", "eiffel_tower", "ref_1.jpg")

    if not os.path.exists(query_path):
        print(f"Error: File not found: {query_path}")
        return

    print(f"[CLI] Processing Query Image: {query_path}")
    q_bgr = cv2.imread(query_path)

    engine = SIFTEngine()
    database = engine.index_reference_database(os.path.join("data", "landmarks"))

    gray_q, kp_q, des_q = engine.extract_features(q_bgr)
    matcher = FeatureMatcher(ratio_threshold=0.75, min_match_threshold=20)
    rec = matcher.recognize_landmark(des_q, database)

    print(f"Decision: {'RECOGNIZED' if rec['is_recognized'] else 'REJECTED'}")
    print(f"Predicted Landmark: {rec['predicted_landmark']}")
    if rec['is_recognized']:
        top = rec['top_candidate']
        print(f"Reliable Matches: {top['best_match_count']}")
        localizer = GeometricLocalizer()
        loc = localizer.estimate_homography_and_localize(
            q_bgr, kp_q,
            top['best_ref_meta']['bgr'], top['best_ref_meta']['keypoints'],
            top['best_ref_matches']
        )
        print(f"RANSAC Inliers: {loc['inliers_count']} / {top['best_match_count']}")
        cv2.imwrite("results/cli_output.jpg", loc['annotated_image'])
        print("Annotated result saved to results/cli_output.jpg")
    else:
        print(f"Rejection Reason: {rec['rejection_reason']}")

if __name__ == '__main__':
    main()
