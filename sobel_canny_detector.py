import cv2
import numpy as np

def apply_sobel_canny_detection(image):
  # Grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Reduce noise
  blurred = cv2.GaussianBlur(gray, (3, 3), 0)
  # Sobel detection
  sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
  sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
  magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
  sobel_result = (magnitude * 255 / magnitude.max()).astype(np.uint8)

  # Canny detection
  blurred = cv2.GaussianBlur(sobel_result, (3, 3), 0)
  eroded = cv2.erode(blurred, np.ones((3, 3), np.uint8))
  combined_result = cv2.Canny(eroded, 50, 160)
  return sobel_result, combined_result