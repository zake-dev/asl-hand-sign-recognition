import cv2
import numpy as np

def apply_canny_edge_detection(image):
  # Gray scale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Reduce noise
  blurred = cv2.GaussianBlur(image, (5, 5), 0)
  # Erosion (emphasize edge lines)
  eroded = cv2.erode(blurred, np.ones((3, 3), np.uint8))
  # Canny (with sensitive thresholds)
  edge_detected = cv2.Canny(eroded, 30, 60)
  return edge_detected