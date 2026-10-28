import urllib.request
import urllib.parse
import json
import time

def test_api():
    url = "http://127.0.0.1:8000/api/analyze"
    data = urllib.parse.urlencode({
        "sample_id": "low_view3",
        "max_duration_sec": "5.0",
        "sample_fps": "10.0",
        "drop_threshold": "20.0"
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, method="POST")
    print("Sending POST request to /api/analyze for low_view3 (5.0s)...")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            status = resp.status
            content = resp.read().decode("utf-8")
            t_total = time.time() - t0
            print(f"Status: {status} (completed in {t_total:.1f}s)")
            res = json.loads(content)
            print("Response Keys:", list(res.keys()))
            print(f"Census Student Count: {res.get('census_student_count')}")
            print(f"Total Students Tracked: {res.get('total_students_tracked')}")
            print(f"Exited Camera Count: {res.get('exited_camera_count')}")
            print(f"Vanished Students: {res.get('vanished_students_count')}")
            print(f"Last Box Event: {res.get('last_box_event')}")
            print(f"Output Video URL: {res.get('output_video_url')}")
            print(f"Drop Events Count: {res.get('drop_events_count')}")
            print(f"Class Average Attention: {res.get('class_average_attention')}%")
            print("SUCCESS! Pipeline executed cleanly without crashes or missing attributes.")
    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    test_api()
