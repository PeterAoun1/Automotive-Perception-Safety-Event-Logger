

import argparse
import time
import csv
import cv2
import numpy as np
from collections import OrderedDict, deque
from math import hypot

class CentroidTracker:
    def __init__(self, max_lost=30, max_distance=120):
        self.next_object_id = 0
        self.objects = OrderedDict()  
        self.lost = OrderedDict()     
        self.tracks = OrderedDict()   
        
        self.max_lost = max_lost
        self.max_distance = max_distance

    def register(self, centroid, timestamp):
        oid = self.next_object_id
        self.next_object_id += 1
        self.objects[oid] = centroid
        self.lost[oid] = 0
        self.tracks[oid] = deque(maxlen=60)  
        self.tracks[oid].append((timestamp, centroid[0], centroid[1]))
        return oid

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.lost:
            del self.lost[object_id]
        if object_id in self.tracks:
            del self.tracks[object_id]

    def update(self, detections, timestamp):
       
        centroids = []
        for (x1,y1,x2,y2) in detections:
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            centroids.append((cx, cy))

        if len(self.objects) == 0:
            for c in centroids:
                self.register(c, timestamp)
            return self.objects, self.tracks

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.zeros((len(object_centroids), len(centroids)), dtype=np.float32)
        for i, oc in enumerate(object_centroids):
            for j, nc in enumerate(centroids):
                D[i, j] = hypot(oc[0]-nc[0], oc[1]-nc[1])

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()

        for row, col in zip(rows, cols):
            if row in assigned_rows or col in assigned_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            self.objects[oid] = centroids[col]
            self.lost[oid] = 0
            self.tracks[oid].append((timestamp, centroids[col][0], centroids[col][1]))
            assigned_rows.add(row)
            assigned_cols.add(col)

        
        for i, oid in enumerate(object_ids):
            if i not in assigned_rows:
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    self.deregister(oid)

        
        for j, c in enumerate(centroids):
            if j not in assigned_cols:
                self.register(c, timestamp)

        return self.objects, self.tracks


def apply_homography(H, px, py):
    p = np.array([px, py, 1.0])
    g = H.dot(p)
    g = g / g[2]
    return float(g[0]), float(g[1])  

def estimate_speed(track, pixels_per_meter=None, H=None):
    
    if len(track) < 2:

        return None
   
    t0, x0, y0 = track[0]
    tn, xn, yn = track[-1]
    dt = tn - t0
    if dt <= 0.001:
        return None
    if H is not None:
        gx0, gy0 = apply_homography(H, x0, y0)
        gxn, gyn = apply_homography(H, xn, yn)
        dist_m = hypot(gx0 - gxn, gy0 - gyn)
    elif pixels_per_meter and pixels_per_meter > 0:
        dist_px = hypot(xn - x0, yn - y0)
        dist_m = dist_px / pixels_per_meter
    else:
        return None
    speed_m_s = dist_m / dt
    speed_kmh = speed_m_s * 3.6
    return speed_m_s, speed_kmh

def pairwise_ground_distances(objects, H=None, pixels_per_meter=None):
    
    ids = list(objects.keys())
    coords = []
    for oid in ids:
        cx, cy = objects[oid]
        if H is not None:
            gx, gy = apply_homography(H, cx, cy)
        elif pixels_per_meter:
            gx = cx / pixels_per_meter
            gy = cy / pixels_per_meter
        else:
            gx, gy = None, None
        coords.append((gx, gy))
    dists = {}  
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            id1, id2 = ids[i], ids[j]
            g1 = coords[i]; g2 = coords[j]
            if g1[0] is None or g2[0] is None:
                d = None
            else:
                d = hypot(g1[0]-g2[0], g1[1]-g2[1])
            dists[(id1, id2)] = d
    return dists


def main(args):
    from ultralytics import YOLO
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source if args.source else 0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open source: " + str(args.source))

    fps = args.fps if args.fps else (cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_dt = 1.0 / fps

    H = None
    if args.homography:
        H = np.load(args.homography)

    ppm = args.pixels_per_meter

    tracker = CentroidTracker(max_lost=30, max_distance=args.max_assign_dist)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    csvf = open(args.csv or "yolo_speeds.csv", "w", newline="")
    cw = csv.writer(csvf)
    cw.writerow(["frame", "object_id", "speed_m_s", "speed_kmh"])

    frame_idx = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_ts = frame_idx / float(fps)

        
        results = model(frame, imgsz=args.imgsz, conf=args.conf, device=args.device)[0]

        
        boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()  # N x 4
            confs = results.boxes.conf.cpu().numpy()
            cls = results.boxes.cls.cpu().numpy().astype(int)
            for (b, c, cl) in zip(xyxy, confs, cls):
                
                if args.vehicle_only:
                    
                    if cl not in (2, 3, 5, 7):
                        continue
                x1, y1, x2, y2 = map(int, b)
                boxes.append((x1, y1, x2, y2))

        objects, tracks = tracker.update(boxes, frame_ts)

        
        speeds = {}
        for oid, track in tracks.items():
            sp = estimate_speed(track, pixels_per_meter=ppm, H=H)
            if sp is not None:
                speeds[oid] = sp 

        dists = pairwise_ground_distances(objects, H=H, pixels_per_meter=ppm)

        for oid, centroid in objects.items():
            cx, cy = centroid
            label = f"ID {oid}"
            if oid in speeds:
                _, kmh = speeds[oid]
                label += f" {kmh:.1f} km/h"
                cw.writerow([frame_idx, oid, f"{speeds[oid][0]:.3f}", f"{speeds[oid][1]:.2f}"])
            cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)
            cv2.putText(frame, label, (cx+8, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 2)

        
        for (id1, id2), d in dists.items():
            if d is not None and d < args.show_dist_threshold:
                
                c1 = objects.get(id1); c2 = objects.get(id2)
                if c1 and c2:
                    cv2.line(frame, (c1[0], c1[1]), (c2[0], c2[1]), (0,0,255), 1)
                    midx = int((c1[0]+c2[0])/2); midy = int((c1[1]+c2[1])/2)
                    cv2.putText(frame, f"{d:.1f}m", (midx+5, midy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)


        for (x1,y1,x2,y2) in boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,128,255), 2)

        cv2.imshow("YOLO speed", frame)
        if out:
            out.write(frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

        frame_idx += 1

    csvf.close()
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="YOLOv8 .pt or model string (e.g. yolov8n.pt)")
    parser.add_argument("--source", default=None, help="Video file or camera index (omit for webcam)")
    parser.add_argument("--homography", default=None, help="Path to homography .npy mapping image->ground in meters")
    parser.add_argument("--pixels-per-meter", type=float, default=None, help="Fallback scale if no homography")
    parser.add_argument("--fps", type=float, default=None, help="Video FPS override")
    parser.add_argument("--output", default=None, help="Annotated video output path")
    parser.add_argument("--csv", default="yolo_speeds.csv", help="CSV of speeds")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--vehicle-only", action="store_true", help="Only keep vehicle classes (car, motorcycle, bus, truck)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--device", default="cpu", help="ultralytics device, e.g. cpu or 0 for GPU")
    parser.add_argument("--max_assign_dist", type=int, default=120, help="Centroid assignment max distance (pixels)")
    parser.add_argument("--show_dist_threshold", type=float, default=10.0, help="Show pairwise distances < threshold (meters)")
    args = parser.parse_args()
    main(args)