# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images/solidWhiteRight.jpg "solidWhiteRight"
[image2]: ./test_images_output/solidWhiteRight-out.jpg "solidWhiteRight Output"
[image3]: ./test_images_output/results.jpg "Results"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

In order to design the proposed pipeline, seven steps were followed: 

* Import and copy a figure from the desired set, e.g. solidWhiteRight.jpg. 

![alt text][image1]

* Define color thresholds. 

* Defined kernel size and perfomed Gaussian smoothing. 

* Applied Canny’s transform, with thresold levels [50, 150]. 

* Masked the edges, created vertices and used cv2.fillpoly() to fill up the bounded area. 

* Defined the Hough parameters. 

* Created the lines that persue the lane line on the road. These lines rely on the Hough parameters, meaning, they are key for the development of this project. As well as, the vertices applied in the image. 

* Draw the lines on the image’s edges. 

* Save the new image, with the detected lane lines, into the folder test_images_output. In the case of solideWhiteRight.jpg, the obtained result is: 

![alt text][image2]

The results of the other test images can be observed below: 

![alt-text-1][image3] 

### 2. Identify potential shortcomings with your current pipeline


One potential shoritcoming of this pipeline would be when encountering highly curved situations. To my understanding, the designed system would not be capable to indentify highly close curves. 

Another shortcoming could appear when using this pipeline with images that have different light and color conditions than for the ones it was designed for. 



### 3. Suggest possible improvements to your pipeline

For improving the designed pipeline in this project, adaptability mechanisms could be implemented or using techniques to obtain information from the pictures that would not be subject to either brightness or color conditions. Therefore, the system would be more likely to draw the lines correctly for any time of situation. 
