# Final Production Benchmark & Accuracy Evaluation on Annotated Clip 4

**Evaluation Date**: 2026-10-28  
**Test Clip**: `data/clip 4.mp4` (832x464 @ 29.97 FPS, 300 frames, 10.01s)  
**Ground Truth File**: `data/annotate for clip 4/annotations.xml` (6,300 bounding boxes across 21 students)  
**Production Checkpoint**: `classroom_attention_research/checkpoints/final_attention_model.joblib`  
**Diagnostic Output Video**: `web_app/static/outputs/annotated_clip_4.mp4`  

---

## 1. Executive Summary

With the final calibrated ensemble model deployed into `AttentionScorer`, the classroom attention pipeline achieves production-grade performance across all dimensions:

| Dimension | Metric | Score | Thesis Significance |
| :--- | :--- | :---: | :--- |
| **Census Tracking** | Student Discovery Rate | **21 / 21 (100.0%)** | All 21 seated students discovered & tracked |
| **Spatial Localization** | Mean Bounding Box IoU | **0.631** | Tight spatial localization across all rows |
| **Detection Quality** | Detection F1 (@ IoU 0.30) | **87.2%** | Recall = 91.1%, Precision = 83.5% |
| **3-Class Attention** | Overall Accuracy | **93.3%** | High vs Moderate vs Low across 5,943 frames |
| **Balanced Performance**| Balanced Accuracy | **94.5%** | Equal discriminative ability on all classes |
| **Macro F1-Score** | Multi-class Macro F1 | **0.922** | Robust multi-class F1 |
| **Binary Attention** | Binary Accuracy | **93.6%** | High Focus vs Disengaged |
| **Disengagement Recall**| Low Attention Recall | **97.2%** | Disengaged students captured (1,081 of 1,112) |
| **Attentive Recall** | High Attention Recall | **91.8%** | Attentive students captured (3,477 of 3,788) |
| **Moderate Recall** | Moderate Attention Recall | **94.6%** | Subtle engagement captured (987 of 1,043) |

---

## 2. Confusion Matrix

$$\begin{array}{r|ccc|c}
\text{Ground Truth} \backslash \text{Predicted} & \textbf{High} & \textbf{Moderate} & \textbf{Low} & \textbf{Total Support} \\
\hline
\textbf{High} & \mathbf{3,477} & 237 & 74 & 3,788 \\
\textbf{Moderate} & 42 & \mathbf{987} & 14 & 1,043 \\
\textbf{Low} & 27 & 4 & \mathbf{1,081} & 1,112 \\
\end{array}$$

---

## 3. Deployment Summary

The final model is fully integrated into:
- [`src/models/attention_scorer.py`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/src/models/attention_scorer.py)
- [`src/pipeline/video_pipeline.py`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/src/pipeline/video_pipeline.py)
- Web Server & Interactive Dashboard at `http://127.0.0.1:8000`
