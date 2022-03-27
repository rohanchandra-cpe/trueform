# trueform
hoohacks22 submission from Neil Phan (nnp3axx), Alip Arslan (aa8pss), Rohan Chandra (rc8yw) 

TrueForm is a body-tracking AI that determines whether a user is performing an exercise using the proper form.
By leveraging Tensorflow's MoveNet model and OpenCV, we are able to draw a skeleton around the user and determine 
the angles between different edges in the skeleton. We can then measure these angles and check whether they are in a pre-determined range or not. 

For example, in order to do a proper push-up, the user's back needs to be straight. 
