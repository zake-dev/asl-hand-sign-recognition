import cv2
from sobel_canny_detector import *
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

# Preprocess all loaded images in list
print("Start Preprocessing...")
for sign_type in sign_types:
  for i in range(1, 3001):
    image = cv2.imread(f'./images/asl_alphabet_train/{sign_type}/{sign_type}{i}.jpg')
    cropped = crop_image(image)

    # Apply specific technique
    sobel, sobel_canny = apply_sobel_canny_detection(cropped)

    # Results
    filepath = './preprocessed/sobel-canny-combination/'
    cv2.imwrite(f'{filepath}{sign_type}/{sign_type}{i}.jpg', sobel_canny)
    print(f'{sign_type} {i}/3000')
