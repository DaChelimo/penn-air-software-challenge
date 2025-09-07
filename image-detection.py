import cv2
import numpy as np


# Load the image
image = cv2.imread('PennAir 2024 App Static.png')
if image is None:
    print("Error: Could not load the image. Please check the file path.")
    exit()

# Duplicate the image so that we can edit the copy instead
outputImage = image.copy()


blurredImage = cv2.GaussianBlur(outputImage, (3, 3), 0)

# Convert to hsv so that I can apply a mask that converts every green pixel to black
hsv_image = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2HSV)
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 253, 200])

# Get all pixels that fall in this color range and set them to black
mask = cv2.inRange(hsv_image, lower_green, upper_green)
blurredImage[mask > 0] = [0, 0, 0]

# Determine your edges and contours from the image
edges = cv2.Canny(blurredImage, 50, 300)
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for i, contour in enumerate(contours):

    #  Remove noise by filtering out small contours
    if cv2.contourArea(contour) > 500:

        # Draw on outline, identify the shape's centre and a draw a red circle identifier
        cv2.drawContours(outputImage, [contour], -1, (0, 255, 0), 2)

        shapeMoment = cv2.moments(contour)
        if shapeMoment["m00"] != 0:
            cX = int(shapeMoment["m10"] / shapeMoment["m00"])
            cY = int(shapeMoment["m01"] / shapeMoment["m00"])

            cv2.circle(outputImage, (cX, cY), 5, (0, 0, 255), -1) # Dot at centre
            cv2.putText(outputImage, f"Shape {i+1}", (cX + 50, cY + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2) # Name of circle

cv2.imshow("Shape Detection from Image", outputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()