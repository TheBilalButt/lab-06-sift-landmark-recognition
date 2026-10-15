"""
Geometric Verification & Landmark Localization Engine
Author: Bilal Butt (@TheBilalButt)
Course: Computer Vision - Lab 06
"""

import cv2
import numpy as np

class GeometricLocalizer:
    def __init__(self, ransac_threshold=5.0, min_inliers=12, min_inlier_ratio=0.20):
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio

    def estimate_homography_and_localize(self, query_bgr, query_kps, ref_bgr, ref_kps, good_matches):
        """Task 5: Estimates Homography using RANSAC, checks geometric validity, and projects corners."""
        h_q, w_q = query_bgr.shape[:2]
        h_r, w_r = ref_bgr.shape[:2]
        annotated_query = query_bgr.copy()

        if len(good_matches) < 4:
            return {
                'success': False,
                'reason': f'Insufficient matches ({len(good_matches)} < 4)',
                'H': None,
                'inliers_count': 0,
                'inlier_ratio': 0.0,
                'annotated_image': annotated_query,
                'corners': None
            }

        src_pts = np.float32([ref_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([query_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        if H is None or mask is None:
            return {
                'success': False,
                'reason': 'RANSAC homography estimation failed',
                'H': None,
                'inliers_count': 0,
                'inlier_ratio': 0.0,
                'annotated_image': annotated_query,
                'corners': None
            }

        inliers_count = int(mask.sum())
        inlier_ratio = inliers_count / len(good_matches) if len(good_matches) > 0 else 0

        # Check inlier threshold
        if inliers_count < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            self._render_warning_banner(annotated_query, f"GEOMETRY UNRELIABLE: Inliers {inliers_count} < {self.min_inliers}")
            return {
                'success': False,
                'reason': f'Insufficient inliers ({inliers_count}/{len(good_matches)} = {inlier_ratio*100:.1f}%)',
                'H': H,
                'inliers_count': inliers_count,
                'inlier_ratio': inlier_ratio,
                'annotated_image': annotated_query,
                'corners': None
            }

        # Check Homography determinant (must be positive to avoid self-intersecting reflection)
        det = np.linalg.det(H[:2, :2])
        if det <= 0:
            self._render_warning_banner(annotated_query, "GEOMETRY UNRELIABLE: Negative Jacobian Determinant")
            return {
                'success': False,
                'reason': f'Non-invertible / mirrored homography (det={det:.3f})',
                'H': H,
                'inliers_count': inliers_count,
                'inlier_ratio': inlier_ratio,
                'annotated_image': annotated_query,
                'corners': None
            }

        # Project 4 corners of reference image onto query image
        ref_corners = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]]).reshape(-1, 1, 2)
        projected_corners = cv2.perspectiveTransform(ref_corners, H)

        # Geometric sanity check: Convexity
        is_convex = cv2.isContourConvex(np.int32(projected_corners))
        if not is_convex:
            self._render_warning_banner(annotated_query, "GEOMETRY UNRELIABLE: Non-Convex Polygon Projection")
            return {
                'success': False,
                'reason': 'Projected bounding polygon is degenerate / non-convex',
                'H': H,
                'inliers_count': inliers_count,
                'inlier_ratio': inlier_ratio,
                'annotated_image': annotated_query,
                'corners': projected_corners
            }

        # Geometric sanity check: Area ratio
        poly_area = cv2.contourArea(np.int32(projected_corners))
        query_area = w_q * h_q
        if poly_area < 0.02 * query_area or poly_area > 3.0 * query_area:
            self._render_warning_banner(annotated_query, "GEOMETRY UNRELIABLE: Polygon Area Out of Bounds")
            return {
                'success': False,
                'reason': f'Unreasonable projected area ({poly_area:.0f} vs query {query_area})',
                'H': H,
                'inliers_count': inliers_count,
                'inlier_ratio': inlier_ratio,
                'annotated_image': annotated_query,
                'corners': projected_corners
            }

        # Valid Localization: Render polygon and telemetry
        pts_int = np.int32(projected_corners)
        cv2.polylines(annotated_query, [pts_int], True, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Corner markers
        for pt in pts_int:
            cv2.circle(annotated_query, tuple(pt[0]), 6, (0, 200, 255), -1, cv2.LINE_AA)

        # Header banner
        banner = annotated_query.copy()
        cv2.rectangle(banner, (0, 0), (w_q, 45), (15, 23, 42), -1)
        cv2.addWeighted(banner, 0.75, annotated_query, 0.25, 0, annotated_query)
        cv2.rectangle(annotated_query, (0, 0), (w_q, 45), (0, 255, 0), 2)
        txt = f"LOCALIZED: {inliers_count}/{len(good_matches)} inliers ({inlier_ratio*100:.1f}%)"
        cv2.putText(annotated_query, txt, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

        return {
            'success': True,
            'reason': 'Geometric verification passed',
            'H': H,
            'inliers_count': inliers_count,
            'inlier_ratio': inlier_ratio,
            'annotated_image': annotated_query,
            'corners': projected_corners
        }

    def _render_warning_banner(self, img, text):
        h, w = img.shape[:2]
        banner = img.copy()
        cv2.rectangle(banner, (0, 0), (w, 45), (15, 23, 42), -1)
        cv2.addWeighted(banner, 0.8, img, 0.2, 0, img)
        cv2.rectangle(img, (0, 0), (w, 45), (0, 0, 255), 2)
        cv2.putText(img, text, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 50, 255), 2)
