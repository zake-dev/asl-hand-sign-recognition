import cv2
import numpy as np

# Open image
filenames = ['A/A200', 'A/A509', 'A/A747', 'A/A1000', 'A/A1797', 'A/A2249', 'A/A2995']
filenames = ['B/B200', 'B/B509', 'B/B747', 'B/B1000', 'B/B1797', 'B/B2249', 'B/B2995']
filenames = ['C/C200', 'C/C509', 'C/C747', 'C/C1000', 'C/C1797', 'C/C2249', 'C/C2995']

for filename in filenames:
  image = cv2.imread("./images/asl_alphabet_train/" + filename + ".jpg", cv2.IMREAD_GRAYSCALE)

  sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
  sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

  magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
  result = (magnitude * 255 / magnitude.max()).astype(np.uint8)

  # Results
  filepath = './preprocessed/edge-detection/'
  # cv2.imwrite(filepath + filename + "_Canny.jpg", edges)
  # cv2.imwrite(filepath + filename + "_Original.jpg", image)
  cv2.imshow('Otsu', result)
  cv2.imshow('Original', image)
  cv2.waitKey(0)

cv2.destroyAllWindows()  