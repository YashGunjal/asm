# Active Shape Model
Active Shape Model for Facial Keypoint Detection

The shape of an object is represented by a set of points (controlled by the shape model). The ASM algorithm aims to match the model to a new image.
It use Principal Component Analysis to reduce the number of points to be examine or we can say that to define a relation between the the points in the shape.
Here we just consider objects made up from a finite number k of points in n dimensions. Often, these points are selected on the continuous surface of complex objects, such as a human bone, and in this case they are called landmark points.

To Run:
Run file name Shape_to_image.py
which take two argument it goes like this

python Shape_to_image.py [Shape to fit] [Image to find shape]

