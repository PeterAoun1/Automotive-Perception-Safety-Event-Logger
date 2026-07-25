"""
Interactive homography calibrator.
- Click 4 or more points on the image that correspond to known ground points.
- After selecting points, input their real-world coordinates in meters (X,Y).
- Saves homography to homography.npy (image -> ground meters).

Usage:
 python homography_calibrator.py --image frame.jpg --output homography.npy
"""
import cv2
import numpy as np
import argparse

pts = []
clone = None

def click(event, x, y, flags, param):
    global pts, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        cv2.circle(clone, (x, y), 4, (0,255,0), -1)
        cv2.imshow("calib", clone)

def main(args):
    global pts, clone
    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError("Cannot open image: " + args.image)
    clone = img.copy()
    cv2.imshow("calib", clone)
    cv2.setMouseCallback("calib", click)

    print("Click 4 or more ground points in the image (clockwise or arbitrary). Press 'q' when done.")
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    if len(pts) < 4:
        raise RuntimeError("Need at least 4 points, got {}".format(len(pts)))

    print("You clicked these image points (in pixels):")
    for i, p in enumerate(pts):
        print(i, p)

    world_pts = []
    print("Now enter the corresponding real-world coordinates for each point in meters as 'X Y' (e.g. 0.0 0.0).")
    for i, p in enumerate(pts):
        inp = input(f"Point {i} image {p} -> world X Y: ")
        xs = inp.strip().split()
        if len(xs) != 2:
            raise RuntimeError("Enter two numbers")
        world_pts.append((float(xs[0]), float(xs[1])))

    img_pts_np = np.array(pts, dtype=np.float32)
    world_pts_np = np.array(world_pts, dtype=np.float32)

    H, mask = cv2.findHomography(img_pts_np, world_pts_np, method=0)
    print("Homography matrix (image -> ground meters):\n", H)
    np.save(args.output, H)
    print("Saved homography to", args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Image to click (one frame from your video)")
    parser.add_argument("--output", default="homography.npy", help="Output .npy file")
    args = parser.parse_args()
    main(args)