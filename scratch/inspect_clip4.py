import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
from collections import Counter

def inspect_dataset():
    xml_path = Path("data/annotate for clip 4/annotations.xml")
    video_path = Path("data/clip 4.mp4")
    
    print("=== INSPECTING VIDEO: clip 4.mp4 ===")
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    print(f"Resolution: {w}x{h}, FPS: {fps:.2f}, Frames: {frame_count}, Duration: {duration:.2f}s")
    
    print("\n=== PARSING ANNOTATIONS XML ===")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    tracks = root.findall(".//track")
    print(f"Total Annotated Tracks: {len(tracks)}")
    
    label_counts = Counter()
    frames_with_annotations = set()
    track_boxes = {}
    
    for track in tracks:
        tid = track.get("id")
        tlabel = track.get("label")
        boxes = track.findall("box")
        track_boxes[tid] = len(boxes)
        
        for box in boxes:
            f = int(box.get("frame"))
            outside = int(box.get("outside", 0))
            if outside == 1:
                continue
            frames_with_annotations.add(f)
            att_elem = box.find(".//attribute[@name='Attention']")
            if att_elem is not None and att_elem.text:
                label_counts[att_elem.text.strip()] += 1
            else:
                label_counts["Unlabeled"] += 1
                
    print(f"Track summary (id -> visible box count): {track_boxes}")
    print(f"Total Unique Frames with annotations: {len(frames_with_annotations)} (Min: {min(frames_with_annotations)}, Max: {max(frames_with_annotations)})")
    print(f"Ground Truth Attention Label Distribution: {dict(label_counts)}")

if __name__ == "__main__":
    inspect_dataset()
