import cv2
from sobel_canny_detector import *

# Crop outline of image by padding size
def crop_image(image, padding=8):
  height, width, _ = image.shape
  crop_size = min(height - padding, width - padding)
  x = (width - crop_size) // 2
  y = (height - crop_size) // 2
  cropped = image[y:y + crop_size, x:x + crop_size]
  return cropped

# List up image set to be loaded
sign_type = 'A/A'
filenames = ['200', '509', '747', '1000', '1797', '2249', '2995']

# Preprocess all loaded images in list
for filename in filenames:
  image = cv2.imread("./images/asl_alphabet_train/" + sign_type + filename + ".jpg")
  original = image.copy()
  cropped = crop_image(image)

  # Apply specific technique
  sobel, sobel_canny = apply_sobel_canny_detection(cropped)

  # Results
  filepath = './preprocessed/sobel-canny-combination/'
  cv2.imwrite(filepath + filename + "_Sobel-Only.jpg", sobel)
  cv2.imwrite(filepath + filename + "_Sobel-Canny_Combination.jpg", sobel_canny)
  cv2.imwrite(filepath + filename + "_Original.jpg", image)
  cv2.imshow('Sobel Only', sobel)
  cv2.imshow('Sobel+Canny', sobel_canny)
  cv2.imshow('Original', original)
  cv2.waitKey(0)

cv2.destroyAllWindows()  