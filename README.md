# Jetson-Nano
The goal of this project is to use the jetson nano for image and depth analysis. It follows a similar method to the drone project and incorporates new features such as the Nividia GPU and the depth analysis.   

This project was setup using the files and method from yahboom. The link can be found here: http://www.yahboom.net/study/jetson-nano
There are currently two files for the project, Frame_Analysis_Windows_With_Nividia.py uses the yolov11 docker to run ultralytics on the jetson system. It loops through a set of images and does object detection on each one. The memory usage and detection information would be stored on a excel file for further analysis. 

Depth_Test_2.py uses the depthnet to estimate the depth field of the image. It would first desaturate it then analyze it to produce the depth map. The goal is to increase the accuracy of the analysis and use the depth estimation for further calculations such as weight.
