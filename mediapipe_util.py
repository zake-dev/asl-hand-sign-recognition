import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

hands = mp.solutions.hands.Hands()

def find_hand_area(image, padding):
  results = hands.process(image)
  print(results.multi_hand_landmarks)

  if results.multi_hand_landmarks:
    for landmarks in results.multi_hand_landmarks:
      min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0

    for landmark in landmarks.landmark:
      x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])

      min_x = min(min_x, x)
      min_y = min(min_y, y)
      max_x = max(max_x, x)
      max_y = max(max_y, y)
    
    return min_x - padding, min_y - padding, max_x + padding, max_y + padding
  

BG_COLOR = (192, 192, 192)
MASK_COLOR = (255, 255, 255)

import cv2

# Create the options that will be used for ImageSegmenter
model_file = open('./deeplab_v3.tflite', 'rb')
model_asset = model_file.read()
model_file.close()

base_options = python.BaseOptions(model_asset_buffer=model_asset)
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)

# Create the image segmenter
with vision.ImageSegmenter.create_from_options(options) as segmenter:

  # Create the MediaPipe image file that will be segmented
  image = mp.Image.create_from_file("./images/asl_alphabet_train/O/O278.jpg")

  # Retrieve the masks for the segmented image
  segmentation_result = segmenter.segment(image)
  category_mask = segmentation_result.category_mask

  # Generate solid color images for showing the output segmentation mask.
  image_data = image.numpy_view()
  fg_image = np.zeros(image_data.shape, dtype=np.uint8)
  fg_image[:] = MASK_COLOR
  bg_image = np.zeros(image_data.shape, dtype=np.uint8)
  bg_image[:] = BG_COLOR

  condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
  output_image = np.where(condition, fg_image, bg_image)

  cv2.imshow("segmented", output_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()