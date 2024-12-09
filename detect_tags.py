import sys
sys.path.append("/usr/local/lib/python3.11/site-packages")

from apriltag import apriltag

def detect_tags(images, family="tagStandard41h12"):

    detector = apriltag(family)
    for image in images:
        detections = detector.detect(image)
        print(detections)

    return detections

def main():
    import cv2
    import numpy as np


if __name__ == "__main__":
    main()


