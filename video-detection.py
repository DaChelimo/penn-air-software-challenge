import cv2

# from google.colab.patches import cv2_imshow


# TODO:  Office Hours Feedback
# Print image after every operation
# Change green to black (for background)
# Apply blur to green
# Do the Canny thing at the very end


# videoStream = cv2.VideoCapture('PennAir 2024 App Dynamic.mp4')
videoStream = cv2.VideoCapture('PennAir 2024 App Dynamic Hard.mp4')


if not videoStream.isOpened():
    print("Error: Could not open the video stream.")
    exit()

# Initialize the background subtractor
mog = cv2.createBackgroundSubtractorMOG2()

# Since the algorithm needs some time to learn what exactly a 'stable background' is,
# we feed it a couple of frames before displaying the render
for i in range(200):
    isFrameCorrect, frame = videoStream.read()
    if not isFrameCorrect:
        break
    fgMask = mog.apply(frame)


# Start video stream processing
while True:
  isFrameCorrect, frame = videoStream.read()

  if not isFrameCorrect:
    print("Error: Could not read a frame from the video stream.")
    break

  # Duplicate the image so that we can edit the copy instead
  outputFrame = frame.copy()

  # Apply the background subtractor to the current frame to get a foreground mask
  fgMask = mog.apply(outputFrame)

  # Find contours in the foreground mask
  contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # TODO: THIS WAS MY CODE BEFORE INCLUDING BACKGROUND AGONISTICS
  # # Blur image to make each pixel 'pick up its surrounding color'
  # blurredImage = cv2.GaussianBlur(outputFrame, (21, 21), 0)

  # # Converting to hsv so that I can apply a mask that converts every green pixel to black
  # # (Why hsv and not your regular BGR... this solves the issue of bright and dull lighting on a color: in BGR,
  # #  the color has different values while in HSV, the color has one value only)
  # hsv_image = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2HSV)
  # lower_green = np.array([40, 50, 50])
  # upper_green = np.array([80, 253, 200])
  #
  # mask = cv2.inRange(hsv_image, lower_green, upper_green)
  # blurredImage[mask > 0] = [0, 0, 0]

  # edges = cv2.Canny(blurredImage, 10, 300)
  # contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


  for i, contour in enumerate(contours):

    # Discard any small contours since they are most likely to be noise
    if cv2.contourArea(contour) > 400:
      cv2.drawContours(outputFrame, [contour], -1, (0, 255, 0), 2)

      # Use moments to determine the center of the circle
      shapeMoment = cv2.moments(contour)
      if shapeMoment["m00"] != 0:
        cX = int(shapeMoment["m10"] / shapeMoment["m00"])
        cY = int(shapeMoment["m01"] / shapeMoment["m00"])

        cv2.circle(outputFrame, (cX, cY), 5, (0, 0, 255), -1)

        coords_text = f"coords: [{cX}, {cY}]"
        cv2.putText(outputFrame, coords_text, (cX, cY + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

  cv2.imshow("Shape detections", outputFrame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

videoStream.release()
cv2.destroyAllWindows()

