import cv2
import numpy as np

def remove_white_area(image):
  lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

  lower_threshold, upper_threshold = np.array([215]), np.array([255])
  white_mask = cv2.inRange(lab_image[:, :, 0], lower_threshold, upper_threshold)

  kernel = np.ones((5, 5), np.uint8)
  white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

  inpainted_image = cv2.inpaint(image.astype(np.uint8), white_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
  return inpainted_image

def apply_sobel_canny_detection(image):
  image = remove_white_area(image)
  # Grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Reduce noise
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  # Apply historgram equalization
  equalized = cv2.equalizeHist(blurred)
  # Sobel detection
  sobel_x = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
  sobel_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
  magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
  sobel_result = (magnitude * 255 / magnitude.max()).astype(np.uint8)

  # Canny detection
  blurred = cv2.GaussianBlur(sobel_result, (3, 3), 0)
  combined_result = cv2.Canny(blurred, 100, 200)
  return sobel_result, combined_result