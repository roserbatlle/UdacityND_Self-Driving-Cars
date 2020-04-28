## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./results/calibration3.png "Cal3"
[image2]: ./results/calibration3_app.png "Cal3 Appereance"
[image3]: ./results/test1_app.png "Test1"
[image4]: ./results/test1_pipeline.png "Test1 Pipeline Result"
[image5]: ./results/test1_prespective_trans.png "Test1 Prespective Transform"
[image6]: ./results/test1_lines.png "Test1 Identified Lines"
[image7]: ./results/test1_result.png "Test1 Result"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

This project starts with the camera calibration, which is implemented in the third cell of the notebook. This procedure is done by using several images of a 9x6 chessboard as reference.

To begin, the object and image points are extracted and stored in two arrays `objpoints` and `imgpoints` respectively. For each calibration image, the corners of the board are found through`cv2.findChessboardCorners()` function.
With the help of  `cv2.drawChessboardCorners()`, the corners are drawn and displayed on top of the imported image. 
For example, for image calibration3.jpg: 

![alt text][image1]

It is clear that this image suffers from distortion, as it is expected. Therefore, the next step is to undistort the image. For this, `cal_undistort()` function is used, which can be found in cell 2, as one of the useful functions for the project. 
`cal_undistort()` first calibrates the camera with `cv2.calibrateCamera()` and then undistort the image with `cv2.undistort()`. The undistorted image is returned.
Now, following the previous example, image calibration3.jpg has this appearance: 
![alt text][image2]



### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The pipeline implementation starts in the fifth cell. To undistort the input image, in this case test1.jpg, the same procedure as explained in the previous section is followed. 
Consequently, the obtained result is: 

![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Considering the explanations of previous lessons, the threshold applied in the image (sixth code cell) adds together Sobel and S thresholds, with the functions `abs_sobel_thresh()` and `hls_select()` that are described in the second cell. 

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In order to create a perspective transform, these steps were followed: 
* Identifying source points: 
```python 
    src = np.float32([[585,470], [245,720], [1145, 720],[735,470]])
```
* Defining destination points: 
```python 
    dst = np.float32([[320,0], [320,720], [960, 720],[960,0]])
```
* Obtaining the transform matrix with `cv2.getPrespectiveTransform()`:  
```python 
    M = cv2.getPerspectiveTransform(src, dst)
```
In this step, the inverse of matrix M is also computed, for later on proposes. 
* Gathering the top-view of the image with `cv2.warpPerspective():` 
```python 
    wraped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
```

The achieved perspective transform is in the seventh code cell.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify the lane-lines in the previous warped image, the function `fit_polynomial()` is called. This function is described in the second cell, as one of the useful functions for this project, and its goal is to find the lane line pixels and then fit them employing `np.polyfit()`. 
To find these lane line pixels, `fit_polynomial()`runs the function `find_lane_pixels()`, which computes the histogram of the image, identifies the pixels positions and extracts their positions in the image. 
Nonetheless, for these functions some hyperparameters must be defined: number of sliding windows, width of the windows and the minimum number of pixels found to recenter the window. 

<img src=./results/test1_lines.png width="256" height="455">

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is obtained after running `calculate_curv()`, which goal is to obtain the curvature in left and right directions and the offset of the car, assuming the camera is mounted in the centre of the vehicle. This function reuses part of the code designed in `find_lane_pixels()` as it requires detecting the lane lines as well. 
Then, with left and right curvature, the radius is obtained. From the offset value, the direction of the car is acquired. For the tested image, the result is: 
```
Curvature radius: 2959.5670736
Car offset: -0.0185 >> The vehicle is moving towards left.
```
This calculus is implemented in the 9th code cell. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally, to reference the identified area to the road, a new function `lane_lines_road()` has been arrange. This function can also be found in the second cell as part of the basic functions for this project. 
The recognized area is back down onto the road thanks to `cv2.fillPoly()`, `cv2.warpPerspe-ctive()` and the inverse of the transformation matrix (previously computed in step 3).  Further, the results are combined into the original image with `cv2.addWeighted()`. 


<img src=./results/test1_result.png width="400">

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

In the 12th code cell,  the process_image() function is implemented. It gathers together all the previous steps and it is used to apply the identified lane lines down to the road in a 50 sec video. The output can be seen [here](https://youtu.be/GG94IfMVJvs). 


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I believe that the most challenging step in this project has been finding the correct perspective transform, as it is key to find the lane lines, their curvature and downing the identified area back to the road. 
Further, considering highly accentuated curves might be a bit difficult to process by the pipeline. While it performs correctly with straight lines, when facing a curvature there is a bit mismatch to identify the area correctly. Therefore, poorly performance would be expected in this situations. 
