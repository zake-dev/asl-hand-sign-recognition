import cv2
import numpy as np

def apply_skin_color_detection(image):
  # HSV color
  image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  HSV = cv2.inRange(image_HSV, (0, 15, 0), (17,170,255)) 
  HSV = cv2.morphologyEx(HSV, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

  # YCrCb color
  image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
  YCrCb = cv2.inRange(image_YCrCb, (0, 135, 85), (255,180,135)) 
  YCrCb = cv2.morphologyEx(YCrCb, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

  # Merge skin detection (YCbCr and hsv)
  merged = cv2.bitwise_and(YCrCb,HSV)
  merged = cv2.medianBlur(merged,3)
  merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

  HSV_converted = cv2.bitwise_not(HSV)
  YCrCb_converted = cv2.bitwise_not(YCrCb)
  merged_converted = cv2.bitwise_not(merged)

  return HSV_converted, YCrCb_converted, merged_converted