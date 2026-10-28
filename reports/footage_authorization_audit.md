# Footage Authorization & Provenance Audit Report

**Date**: October 27, 2026  
**Auditor**: Senior Computer Vision Research Engineer & ML Scientist  
**Repository**: `c:\Users\User\Downloads\Model v2\classroom_attention_research`  
**Evaluation Scope**: Verification of data provenance, ownership, human-subjects consent, institutional ethics compliance, and legal authorization for candidate evaluation clips:
1. `PRE DEFENCE/YOLO Test.mp4`
2. `high/view1.mp4` (`data/pilot/high_engagement/high/view1.mp4`)
3. `low/view3.mp4` (`data/pilot/low_engagement/low/view3.mp4`)
4. `high/view11.mp4` (`data/pilot/high_engagement/high/view11.mp4`)

---

## 1. Executive Summary & Authorization Determination

| Candidate Clip / Asset | Provenance & Origin | Institutional Ethics / Consent Status | Permitted Thesis Research Scope | Authorization Gate Status |
| :--- | :--- | :--- | :--- | :---: |
| **`PRE DEFENCE/YOLO Test.mp4`** | Unverified local source; external uncurated recording | **None**. No human-subjects consent documentation, no IRB review, unredacted faces | **Strictly Prohibited** from any external evaluation, thesis benchmark, or publication | <span style="color:red; font-weight:bold;">FAIL (UNAUTHORIZED)</span> |
| **`high/view1.mp4`** | OUC-CGE Dataset (`osf.io/brd2c`, Nature Sci Data 2024) | Ocean University Ethics Board approval for *group classroom video capture* | Permitted for technical pipeline validation; **Requires IRB & protocol sign-off for individual annotation** | <span style="color:orange; font-weight:bold;">CONDITIONAL / NOT YET VERIFIABLE</span> |
| **`low/view3.mp4`** | OUC-CGE Dataset (`osf.io/brd2c`, Nature Sci Data 2024) | Ocean University Ethics Board approval for *group classroom video capture* | Permitted for technical pipeline validation; **Requires IRB & protocol sign-off for individual annotation** | <span style="color:orange; font-weight:bold;">CONDITIONAL / NOT YET VERIFIABLE</span> |
| **`high/view11.mp4`** | OUC-CGE Dataset (`osf.io/brd2c`, Nature Sci Data 2024) | Ocean University Ethics Board approval for *group classroom video capture* | Permitted for technical pipeline validation; **Requires IRB & protocol sign-off for individual annotation** | <span style="color:orange; font-weight:bold;">CONDITIONAL / NOT YET VERIFIABLE</span> |

> [!CAUTION]
> **HARD STOP MANDATE TRIGGERED**: Because the original consent and IRB approval for OUC-CGE were granted specifically for **classroom group-engagement recognition** rather than fine-grained **individual student profiling**, and because no institutional IRB exemption/approval has been formally registered for secondary individual behavioral annotation, **all three candidate clips are currently NOT VERIFIABLE for immediate human annotation**. 
> 
> **Do not open `high/view1.mp4`, `low/view3.mp4`, or `high/view11.mp4` for annotation, do not upload them to CVAT or cloud tools, and do not generate ground-truth labels until the explicit approvals detailed in Section 5 are provided.**

---

## 2. In-Depth Audit: `PRE DEFENCE/YOLO Test.mp4`

### 2.1 Provenance & Ownership
- **Filename**: `PRE DEFENCE/YOLO Test.mp4`
- **Origin**: Historical project artifact from preliminary exploratory testing.
- **Repository Location**: Legacy archive outside the current clean workspace.
- **Identifiable Subjects**: Contains visible, unmasked students and instructor in an unverified institutional setting.

### 2.2 Ethics, Consent, & Legal Risks
1. **Lack of Participant Consent**: There is zero recorded consent from the students or instructor appearing in the footage.
2. **Absence of Institutional Ethics Clearance**: No Institutional Review Board (IRB) or university ethics committee approved the collection, processing, or evaluation of this footage.
3. **Severe Privacy Violation Risk**: Utilizing unconsented video for automated attention scoring, pose extraction, or thesis defense demonstrations violates standard educational privacy safeguards (including FERPA principles and international GDPR standards regarding biometric/behavioral data processing).
4. **Academic Integrity & Defensibility**: Evaluating a master's thesis model on unauthorized footage risks immediate rejection by the thesis committee and academic sanctions.

### 2.3 Definitive Ruling
**`PRE DEFENCE/YOLO Test.mp4` IS PERMANENTLY BARRED AND DISQUALIFIED FROM ALL RESEARCH PIPELINES, ANNOTATION TASKS, BENCHMARK EVALUATIONS, AND THESIS PRESENTATIONS.**

---

## 3. In-Depth Audit: Proposed OUC-CGE Pilot Clips

### 3.1 Footage Identification & Checksum Integrity
The three candidate pilot clips are located within [`c:\Users\User\Downloads\Model v2\classroom_attention_research\data\pilot`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/data/pilot) and their cryptographic hashes match the verified manifest [`pilot_manifest.json`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/data/manifests/pilot_manifest.json):

1. **`high/view1.mp4`**:
   - Path: `data/pilot/high_engagement/high/view1.mp4`
   - File Size: 2,348,544 bytes
   - MD5: `1a07b732bfbb72a3fca3eead3304910c`
   - SHA-256: `ab6d972f254c1313bc8eabc87fc61192749392b96d624b0d110d10646dc797ab`
   - Capture Setup: Front-center perspective (Camera View 1), 1280x720, 30.0 FPS, H.264.

2. **`low/view3.mp4`**:
   - Path: `data/pilot/low_engagement/low/view3.mp4`
   - File Size: 851,344 bytes
   - MD5: `e15d4d3b9f41080279cfab458e888de1`
   - SHA-256: `14e4e3fd383df03ac22e5173382dc72ca0983a16eeadc6b77d8b7b01a5bd9b6f`
   - Capture Setup: Lateral side perspective (Camera View 3), 1280x720, 30.0 FPS, H.264.

3. **`high/view11.mp4`**:
   - Path: `data/pilot/high_engagement/high/view11.mp4`
   - File Size: 2,302,470 bytes
   - MD5: `88262a87f2c87b6badecf02a337e875f`
   - SHA-256: `c4bea8e5ba9aa8509bd48a9196781d9ba9f4e48cfe5d296aa28d0d03d6674c11`
   - Capture Setup: Rear elevated perspective (Camera View 11), 1280x720, 30.0 FPS, H.264.

### 3.2 Upstream Provenance & Legal Rights
- **Original Dataset**: *OUC-CGE (Ocean University of China Classroom Group Engagement Dataset)*.
- **Reference Publication**: Peng et al., *"A Video Dataset for Classroom Group Engagement Recognition"*, Scientific Data (Nature Portfolio), 2024.
- **Official Source**: Open Science Framework (OSF) repository [`doi:10.17605/OSF.IO/BRD2C`](https://osf.io/brd2c).
- **Public License**: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- **Intellectual Property Rights**: CC BY 4.0 permits sharing, copying, and adapting the material for any purpose, including research and thesis development, provided appropriate credit is given.

---

## 4. The Critical Distinction: Copyright License vs. Human-Subjects Research Ethics

While CC BY 4.0 grants legal permission to distribute and process the video bytes, **a copyright license does NOT equate to an institutional ethical authorization for secondary individual human profiling.**

### 4.1 Scope of Original Consent in OUC-CGE
The original research protocol published in Nature Scientific Data specifies:
1. **Consent Level**: Participants (undergraduate students at Ocean University of China) consented to multi-camera classroom surveillance for **group-level educational engagement assessment**.
2. **Aggregated Objective**: The study's stated objective was classifying overall room atmosphere (`High`, `Medium`, `Low`) to provide cohort-level teacher feedback.
3. **No Individual Ground Truth**: The authors explicitly refrained from publishing individual student names, personal identifiers, or individual psychological/attentional ratings.

### 4.2 Ethical Implications of Secondary Individual Annotation
Creating fine-grained, frame-by-frame individual student tracking and attention labels constitutes **secondary research on human subjects with identifiable visual data**:
1. **Identifiability**: Student faces in `view1.mp4` and `view3.mp4` are visible at resolutions up to $60 \times 80$ pixels without blurring.
2. **Behavioral Profiling**: Labeling an individual student as `Inattentive` or `Sleeping` creates personal performance records derived from identifiable biometric video.
3. **Data Protection Constraints**: Under standard academic research ethics:
   - Identifiable visual records must not be exposed to unvetted annotators.
   - Public thesis presentations and defense slides must not show identifiable faces linked to negative performance ratings (`Inattentive`).
   - Facial anonymization (blurring/masking) is mandatory for exported artifacts.

---

## 5. Required Evidence and Approvals Before Annotation Execution

To achieve a defensible **PASS** authorization status, the following four items must be documented and approved:

### 1. Institutional Ethics / IRB Secondary Use Determination
- **Required Action**: Submit a protocol clarification or secondary data use exemption request to the university Institutional Review Board / Ethics Committee (or thesis supervisor formal waiver).
- **Required Documentation**: Confirmation that secondary annotation of publicly available, CC BY 4.0 licensed educational video (OUC-CGE) for algorithmic evaluation does not constitute unapproved human experimentation.

### 2. Thesis Supervisor & Committee Sign-Off
- **Required Action**: Formal confirmation from the academic supervisor that OUC-CGE clips `view1.mp4`, `view3.mp4`, and `view11.mp4` are approved for the pilot annotation benchmark.
- **Required Documentation**: Written approval of the 3-clip selection and individual annotation scope.

### 3. Data Privacy and Anonymization Plan
- **Required Action**: Implementation of an automated face de-identification / privacy filter for all benchmark visual outputs.
- **Required Rule**: Any frame crop, diagnostic overlay, or thesis figure exported for public defense must apply Gaussian blurring or solid bounding-box occlusion to student facial landmarks.

### 4. Annotator Confidentiality & Data Handling Agreement
- **Required Action**: Annotators participating in the pilot must sign an internal data confidentiality agreement promising:
  - Local-only annotation (no uploading unmasked footage to public cloud annotation platforms like public CVAT servers or Roboflow).
  - Immediate local data deletion upon benchmark completion.

---

## 6. Authorization Gate Verdict

```
+-----------------------------------------------------------------------------+
|                        FOOTAGE AUTHORIZATION SCORECARD                      |
+-----------------------------------------------------------------------------+
| Asset Evaluated            | Legal License | IRB / Ethics Status | Verdict  |
+----------------------------+---------------+---------------------+----------+
| PRE DEFENCE/YOLO Test.mp4  | None          | None                | FAIL     |
| high/view1.mp4             | CC BY 4.0     | Group Consent Only  | COND.    |
| low/view3.mp4              | CC BY 4.0     | Group Consent Only  | COND.    |
| high/view11.mp4            | CC BY 4.0     | Group Consent Only  | COND.    |
+-----------------------------------------------------------------------------+
| OVERALL AUTHORIZATION GATE STATUS: NOT VERIFIABLE (EXECUTION BLOCKED)       |
+-----------------------------------------------------------------------------+
```

**Status**: **STOP — AWAITING EXPLICIT APPROVAL**. The technical annotation protocol and benchmark design are fully specified in accompanying reports, but annotation tasks must not be initiated until the user explicitly confirms the authorization items above.
