import cv2
from canny_detector import *
from sobel_canny_detector import *
from skin_detector import *
import random

# Crop outline of image by padding size
def crop_image(image, padding=8):
  height, width, _ = image.shape
  crop_size = min(height - padding, width - padding)
  x = (width - crop_size) // 2
  y = (height - crop_size) // 2
  cropped = image[y:y + crop_size, x:x + crop_size]
  return cropped

# List up image set to be loaded
sign_types = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] 
random_numbers = [random.randint(1, 3000) for _ in range(10)]

# Show preprocessed images
sign = random.choice(sign_types)
for num in random_numbers:
  image = cv2.imread(f'./images/asl_alphabet_train/{sign}/{sign}{num}.jpg')
  cropped = crop_image(image)

  # Apply specific technique
  canny = apply_canny_edge_detection(cropped)
  sobel, sobel_canny = apply_sobel_canny_detection(cropped)
  HSV_converted, YCrCb_converted, merged_converted = apply_skin_color_detection(cropped)

  # Display results
  cv2.imshow('Original', cropped)
  cv2.imshow('HSV', HSV_converted)
  cv2.imshow('YCrCb', YCrCb_converted)
  cv2.imshow('HSV + YCrCb', merged_converted)
  cv2.imshow('Canny only', canny)
  cv2.imshow('Sobel only', sobel)
  cv2.imshow('Sobel + Canny', sobel_canny)
  cv2.waitKey(0)

cv2.destroyAllWindows()