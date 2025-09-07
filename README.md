# Penn Aerial Robotics - Software Challenge 2025
This project detects shapes in videos, outlines them, and marks their centers. The algorithm uses background subtraction for shape detection in different scenes (achieving background agnosticism).

## Approach

1. Load video with OpenCV `VideoCapture`.
2. Initialize a background subtractor (`cv2.createBackgroundSubtractorMOG2`) and feed a few frames (exactly 100 frames) to learn the stable background.
3. For each frame:
   - Apply the background subtractor to get a foreground mask.
   - Find contours in the foreground mask.
   - Filter out small contours (`area < 400`) to reduce noise.
   - Draw contours and mark centers with coordinates.
4. Display processed frames


#### Challenges

Slight delay before program starts. This is because right at the start, the algorithms lacks adequate training so the outline predictions are beyond terrible (would lowkey love to discuss how to resolve this issue: I'm currently at the optimally lowest number of frames)


#### Setup
```bash
## Requirements

- Python 3.x  
- OpenCV (`opencv-python`)  
- NumPy (`numpy`)


## Install  OpenCV using:
pip install opencv-python
```

<br>

## Image Results 
### For the image processing part (Task 1)
<img width="962" height="605" alt="Part 1 - Image Processing" src="https://github.com/user-attachments/assets/2808cec9-3675-42a5-a50b-89297a144393" /><br><br><br>

## Video Results 
### For the first video processing (Task 2)
[Link to the video on Github] (https://github.com/user-attachments/assets/e121c74d-7c77-4269-90d9-b2e3347d932c)
<img width="1512" height="982" alt="Part 2 - Video Processing " src="https://github.com/user-attachments/assets/8d31da98-0432-41f4-8698-7fe22bc66abf" /><br><br>


### For the second video processing (Task 3)
[Link to the video on Github] (https://github.com/user-attachments/assets/965bdd93-58f7-428e-9010-2f012631a661)
<img width="1512" height="982" alt="Part 3 - Different Background Video Processing " src="https://github.com/user-attachments/assets/616958aa-3224-49f2-829d-049086705336" />



