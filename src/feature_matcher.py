"""
Feature Matching & Landmark Classification Engine
Author: Bilal Butt (@TheBilalButt)
Course: Computer Vision - Lab 06
"""

import cv2
import numpy as np

class FeatureMatcher:
    def __init__(self, ratio_threshold=0.75, min_match_threshold=20, margin_ratio=1.35):
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.ratio_threshold = ratio_threshold
        self.min_match_threshold = min_match_threshold
        self.margin_ratio = margin_ratio

    def match_descriptors(self, query_descriptors, ref_descriptors):
        """Task 4: KNN matching (k=2) with Lowe's ratio test."""
        if query_descriptors is None or ref_descriptors is None:
            return []
        if len(query_descriptors) < 2 or len(ref_descriptors) < 2:
            return []

        knn_matches = self.bf.knnMatch(query_descriptors, ref_descriptors, k=2)
        good_matches = []
        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        return good_matches

    def recognize_landmark(self, query_descriptors, database):
        """Compares query against all landmark reference images, counts reliable matches, and ranks candidates."""
        candidate_scores = {}

        for lm_name, ref_images in database.items():
            best_ref_matches = []
            best_ref_idx = -1
            total_lm_matches = 0

            for idx, ref in enumerate(ref_images):
                matches = self.match_descriptors(query_descriptors, ref['descriptors'])
                total_lm_matches += len(matches)
                if len(matches) > len(best_ref_matches):
                    best_ref_matches = matches
                    best_ref_idx = idx

            candidate_scores[lm_name] = {
                'landmark': lm_name,
                'total_matches': total_lm_matches,
                'best_ref_idx': best_ref_idx,
                'best_ref_matches': best_ref_matches,
                'best_match_count': len(best_ref_matches),
                'best_ref_meta': ref_images[best_ref_idx] if best_ref_idx >= 0 else None
            }

        # Rank candidates by best reference match count
        ranked_candidates = sorted(
            candidate_scores.values(),
            key=lambda x: x['best_match_count'],
            reverse=True
        )

        top_cand = ranked_candidates[0] if ranked_candidates else None
        second_cand = ranked_candidates[1] if len(ranked_candidates) > 1 else None

        # Rejection logic: match threshold AND margin test
        is_recognized = False
        rejection_reason = ""

        if top_cand is None or top_cand['best_match_count'] < self.min_match_threshold:
            is_recognized = False
            rejection_reason = f"Below minimum match threshold ({top_cand['best_match_count'] if top_cand else 0} < {self.min_match_threshold})"
        elif second_cand is not None and second_cand['best_match_count'] > 0:
            margin = top_cand['best_match_count'] / second_cand['best_match_count']
            if margin < self.margin_ratio and top_cand['best_match_count'] < 35:
                is_recognized = False
                rejection_reason = f"Ambiguous matching: margin {margin:.2f} < {self.margin_ratio:.2f}"
            else:
                is_recognized = True
        else:
            is_recognized = True

        return {
            'is_recognized': is_recognized,
            'predicted_landmark': top_cand['landmark'] if is_recognized else 'Unknown / Rejected',
            'top_candidate': top_cand,
            'ranked_candidates': ranked_candidates,
            'rejection_reason': rejection_reason
        }
