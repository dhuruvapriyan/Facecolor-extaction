Methods Followed:
1. Detecting the face using Dlib Frontal face Predictor.
2. using shape predictor plotting facial landmarks on the picture using Pre Trained model.

3.On applying convex hell algorithm on the facial landmarked point ,By this we can remove all noises.

4.k means clustering algorithm with 3 clusters gives us the different colour pixels in the image.

5. plotting RGB values of skin using histogram of clustering algorithm.

6.Classifying color of a person based on Von Luschan classification.

Remarks:
1.This model takes some time to train datasets.but it accuracy is good.

A New User can use the RGB color values of the skin to make their own skin colour classification.
The above code displays [1] [2] [5] [6] steps of the Method  

Dependencies needed :
Python 3.x
Numpy
Scikit Learn
Dlib
Matploitlab
Opencv
imutils

Running this script:
This script can run with Command Prompt or with Anaconda Prompt.
Command to run this : python ./FaceColorExtraction.py shape_predictor_68_face_landmarks.dat ./examples/faces

format : python  [space] script_path [space] Trained_dat_path [space] Image_folder_path

To Download the Pretrained Dataset : http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 [63 Mb]

I have attached the sample face dataset folder.

Further Work:
To Train and Test skin color values of images on Deep Neural Nets

