import xml.etree.ElementTree as ET

tree = ET.parse("data/annotate for clip 4/annotations.xml")
root = tree.getroot()

print("Tracks and initial boxes at frame 0:")
for track in root.findall(".//track"):
    tid = track.get("id")
    box0 = track.find(".//box[@frame='0']")
    if box0 is not None:
        att = box0.find(".//attribute[@name='Attention']")
        att_val = att.text if att is not None else "None"
        xtl = float(box0.get("xtl"))
        ytl = float(box0.get("ytl"))
        xbr = float(box0.get("xbr"))
        ybr = float(box0.get("ybr"))
        print(f"Track {tid:2s} | Box: ({xtl:6.1f}, {ytl:6.1f}, {xbr:6.1f}, {ybr:6.1f}) | W: {xbr-xtl:5.1f}, H: {ybr-ytl:5.1f} | Att: {att_val}")
