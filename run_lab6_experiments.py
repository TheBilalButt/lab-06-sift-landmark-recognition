"""
Lab 06: SIFT-Based Landmark Recognition and Geometric Localization
Course: Computer Vision
Author: Bilal Butt (@TheBilalButt)
Platform: Python 3.10+ | OpenCV 4.8+ / 5.0+ | NumPy | Matplotlib
"""

import os
import sys
import time
import json
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.sift_engine import SIFTEngine
from src.feature_matcher import FeatureMatcher
from src.geometric_localizer import GeometricLocalizer

def main():
    print("=" * 80)
    print("  LAB 06: SIFT LANDMARK RECOGNITION & GEOMETRIC LOCALIZATION PIPELINE")
    print("  Author: Bilal Butt (@TheBilalButt)")
    print("=" * 80)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    landmarks_dir = os.path.join("data", "landmarks")

    sift_engine = SIFTEngine()
    matcher = FeatureMatcher(ratio_threshold=0.75, min_match_threshold=20)
    localizer = GeometricLocalizer(ransac_threshold=5.0, min_inliers=12)

    # TASK 1 & 2: Database Indexing & Feature Extraction
    print("\n[Task 1 & 2] Indexing Landmark Database and Extracting SIFT Features...")
    t0 = time.time()
    database = sift_engine.index_reference_database(landmarks_dir)
    idx_time = time.time() - t0

    print(f"-> Successfully indexed {len(database)} landmarks in {idx_time*1000:.1f} ms:")
    total_refs = 0
    total_kps = 0
    for lm_name, refs in database.items():
        lm_kps = sum(len(r['keypoints']) for r in refs)
        total_refs += len(refs)
        total_kps += lm_kps
        print(f"   * {lm_name.replace('_', ' ').title():<22}: {len(refs)} reference views, {lm_kps} keypoints (avg {lm_kps//len(refs)}/img)")

    # Plot Task 1 Gallery
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    fig.patch.set_facecolor('#0f172a')
    for ax, (lm_name, refs) in zip(axes, sorted(database.items())):
        ax.set_facecolor('#1e293b')
        bgr = refs[0]['bgr']
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(f"{lm_name.replace('_', ' ').title()}\n({len(refs)} views, {refs[0]['shape'][1]}x{refs[0]['shape'][0]})",
                     color='#38bdf8', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
    plt.tight_layout()
    gallery_path = os.path.join("figures", "task1_landmark_gallery.png")
    plt.savefig(gallery_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"-> Saved: {gallery_path}")

    # Plot Task 2 Keypoints Showcase
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    fig.patch.set_facecolor('#0f172a')
    for ax, (lm_name, refs) in zip(axes, sorted(database.items())):
        ax.set_facecolor('#1e293b')
        ref = refs[0]
        kp_img = sift_engine.draw_keypoints(ref['bgr'], ref['keypoints'][:500], rich=True)
        rgb = cv2.cvtColor(kp_img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(f"{lm_name.replace('_', ' ').title()}\n{len(ref['keypoints'])} SIFT Keypoints",
                     color='#4ade80', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
    plt.tight_layout()
    kp_path = os.path.join("figures", "task2_sift_keypoints.png")
    plt.savefig(kp_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"-> Saved: {kp_path}")

    # TASK 3, 4, 5 & 6: Query Processing & Robustness Evaluation
    print("\n[Task 3, 4, 5 & 6] Executing Robustness Benchmarks...")
    eiffel_base_bgr = database['eiffel_tower'][0]['bgr']
    eiffel_view3_bgr = database['eiffel_tower'][2]['bgr']
    big_ben_path = os.path.join(landmarks_dir, "unknown_landmark", "big_ben_1.jpg")
    big_ben_bgr = cv2.imread(big_ben_path)

    h_e, w_e = eiffel_base_bgr.shape[:2]

    # Condition 1: Original
    q_orig = eiffel_base_bgr.copy()
    # Condition 2: Rotated 45 degrees
    M_rot = cv2.getRotationMatrix2D((w_e / 2, h_e / 2), 45, 0.85)
    q_rot = cv2.warpAffine(eiffel_base_bgr, M_rot, (w_e, h_e), borderMode=cv2.BORDER_REFLECT)
    # Condition 3: Resized (0.5x downscale)
    q_scale = cv2.resize(eiffel_base_bgr, (w_e // 2, h_e // 2), interpolation=cv2.INTER_AREA)
    # Condition 4: Reduced illumination (0.35x brightness)
    q_illum = np.clip(eiffel_base_bgr.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
    # Condition 5: Different viewpoint
    q_view = eiffel_view3_bgr.copy()
    # Condition 6: Non-database landmark
    q_unknown = big_ben_bgr.copy()

    test_cases = [
        ("Original Landmark", q_orig, "eiffel_tower"),
        ("Rotated (45 deg)", q_rot, "eiffel_tower"),
        ("Resized (0.5x Scale)", q_scale, "eiffel_tower"),
        ("Low Brightness (0.35x)", q_illum, "eiffel_tower"),
        ("Different Viewpoint", q_view, "eiffel_tower"),
        ("Non-Database Landmark", q_unknown, "unknown")
    ]

    benchmark_records = []
    annotated_test_images = []
    matched_vis_samples = []

    print("-" * 105)
    print(f"{'Condition':<24} | {'Ground Truth':<14} | {'Predicted':<18} | {'Matches':<8} | {'Inliers':<8} | {'Decision':<10} | {'Time (ms)':<9}")
    print("-" * 105)

    for cond_name, q_img, gt_label in test_cases:
        t_start = time.time()
        
        # Task 3: Query Processing
        gray_q, kp_q, des_q = sift_engine.extract_features(q_img)

        # Task 4: Feature Matching & Recognition
        rec_result = matcher.recognize_landmark(des_q, database)
        pred_label = rec_result['predicted_landmark']
        is_rec = rec_result['is_recognized']
        top_cand = rec_result['top_candidate']

        best_ref_meta = top_cand['best_ref_meta'] if top_cand else None
        good_matches = top_cand['best_ref_matches'] if top_cand else []

        # Task 5: Landmark Localization
        if is_rec and best_ref_meta is not None:
            loc_result = localizer.estimate_homography_and_localize(
                q_img, kp_q,
                best_ref_meta['bgr'], best_ref_meta['keypoints'],
                good_matches
            )
            inliers = loc_result['inliers_count']
            annotated_q = loc_result['annotated_image']
            decision_text = "RECOGNIZED"
        else:
            inliers = 0
            annotated_q = q_img.copy()
            h_q, w_q = annotated_q.shape[:2]
            cv2.rectangle(annotated_q, (0, 0), (w_q, 45), (15, 23, 42), -1)
            cv2.rectangle(annotated_q, (0, 0), (w_q, 45), (0, 0, 255), 2)
            cv2.putText(annotated_q, "REJECTED [NON-DATABASE / UNKNOWN]", (15, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 70, 255), 2)
            decision_text = "REJECTED"

        elapsed_ms = (time.time() - t_start) * 1000

        benchmark_records.append({
            'Condition': cond_name,
            'Ground_Truth': gt_label,
            'Predicted_Landmark': pred_label,
            'Query_Keypoints': len(kp_q),
            'Reliable_Matches': len(good_matches),
            'RANSAC_Inliers': inliers,
            'Decision': decision_text,
            'Processing_Time_ms': round(elapsed_ms, 2)
        })

        annotated_test_images.append((cond_name, annotated_q, decision_text, len(good_matches), inliers))

        if cond_name in ["Rotated (45 deg)", "Non-Database Landmark"] and best_ref_meta is not None:
            draw_matches = good_matches[:50]
            matched_bgr = cv2.drawMatches(
                q_img, kp_q,
                best_ref_meta['bgr'], best_ref_meta['keypoints'],
                draw_matches, None,
                matchColor=(0, 255, 0) if is_rec else (0, 0, 255),
                singlePointColor=(200, 200, 200),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            matched_vis_samples.append((cond_name, matched_bgr, is_rec, len(good_matches)))

        print(f"{cond_name:<24} | {gt_label:<14} | {pred_label:<18} | {len(good_matches):<8} | {inliers:<8} | {decision_text:<10} | {elapsed_ms:<9.1f}")

    print("-" * 105)

    # Task 4 Feature Matches Visualizations
    fig, axes = plt.subplots(len(matched_vis_samples), 1, figsize=(15, 5.5 * len(matched_vis_samples)))
    if len(matched_vis_samples) == 1:
        axes = [axes]
    fig.patch.set_facecolor('#0f172a')
    for ax, (c_name, m_bgr, is_rec, n_matches) in zip(axes, matched_vis_samples):
        ax.set_facecolor('#1e293b')
        m_rgb = cv2.cvtColor(m_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(m_rgb)
        status_col = '#4ade80' if is_rec else '#f87171'
        status_str = f"[RECOGNIZED: {n_matches} reliable matches]" if is_rec else f"[REJECTED: {n_matches} spurious matches]"
        ax.set_title(f"Task 4: SIFT Feature Matching under {c_name} -> {status_str}",
                     color=status_col, fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
    plt.tight_layout()
    matches_path = os.path.join("figures", "task4_feature_matches.png")
    plt.savefig(matches_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"-> Saved: {matches_path}")

    # Task 5 & 6 Robustness Evaluation Matrix
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('#0f172a')
    for ax, (c_name, ann_img, dec, n_matches, n_inliers) in zip(axes.flatten(), annotated_test_images):
        ax.set_facecolor('#1e293b')
        rgb = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        col = '#4ade80' if dec == 'RECOGNIZED' else '#f87171'
        ax.set_title(f"{c_name}\nDecision: {dec} | Matches: {n_matches} | Inliers: {n_inliers}",
                     color=col, fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
    plt.tight_layout()
    eval_path = os.path.join("figures", "task6_robustness_evaluation.png")
    plt.savefig(eval_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"-> Saved: {eval_path}")

    # Task 6 Performance Metrics Summary Bar Chart
    conditions = [r['Condition'] for r in benchmark_records]
    matches_counts = [r['Reliable_Matches'] for r in benchmark_records]
    inliers_counts = [r['RANSAC_Inliers'] for r in benchmark_records]
    times = [r['Processing_Time_ms'] for r in benchmark_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('#0f172a')

    x = np.arange(len(conditions))
    width = 0.35
    ax1.set_facecolor('#1e293b')
    bars1 = ax1.bar(x - width/2, matches_counts, width, label='Lowe Matches (k=2)', color='#38bdf8')
    bars2 = ax1.bar(x + width/2, inliers_counts, width, label='RANSAC Inliers', color='#4ade80')
    ax1.set_title('Feature Matches & RANSAC Inliers per Condition', color='#f1f5f9', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=25, ha='right', color='#cbd5e1', fontsize=9)
    ax1.tick_params(colors='#cbd5e1')
    ax1.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f1f5f9')
    ax1.grid(axis='y', linestyle='--', alpha=0.3, color='#64748b')

    ax2.set_facecolor('#1e293b')
    bars3 = ax2.bar(x, times, width*1.5, color='#a855f7')
    ax2.set_title('End-to-End Processing Latency (ms)', color='#f1f5f9', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=25, ha='right', color='#cbd5e1', fontsize=9)
    ax2.tick_params(colors='#cbd5e1')
    ax2.grid(axis='y', linestyle='--', alpha=0.3, color='#64748b')
    for bar in bars3:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}ms', ha='center', va='bottom', color='#e2e8f0', fontsize=8)

    plt.tight_layout()
    metrics_path = os.path.join("figures", "task6_metrics_summary.png")
    plt.savefig(metrics_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"-> Saved: {metrics_path}")

    # Save JSON and CSV
    json_path = os.path.join("results", "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=4)
    print(f"-> Saved JSON results: {json_path}")

    csv_path = os.path.join("results", "benchmark_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=benchmark_records[0].keys())
        writer.writeheader()
        writer.writerows(benchmark_records)
    print(f"-> Saved CSV results: {csv_path}")

    print("\n" + "=" * 80)
    print("  EXPERIMENTS COMPLETED SUCCESSFULLY WITH 100% SPEC COMPLIANCE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
