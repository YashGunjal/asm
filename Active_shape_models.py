#!/usr/bin/env python

import sys
import os
import cv
import glob
import math
import numpy as np
from random import randint

import PCA_analysis as PCA

def drange(start, stop, step):
  r = start
  while r < stop:
    yield r
    r += step

class Point ( object ):
  """ Class to represent a point in 2d cartesian space """
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __add__(self, p):
    """ Return a new point which is equal to this point added to p
    :param p: The other point
    """
    return Point(self.x + p.x, self.y + p.y)

  def __div__(self, i):
    return Point(self.x/i, self.y/i)

  def __eq__(self, other):
    return self.x == other.x and self.y == other.y

  def __ne__(self, other):
    return not self.__eq__(other)

  def __repr__(self):
    """return a string representation of this point. """
    return '(%f, %f)' % (self.x, self.y)

  def dist(self, p):
    """ Return the distance of this point to another point

    :param p: The other point
    """
    return math.sqrt((p.x - self.x)**2 + (p.y - self.y)**2)

class Shape ( object ):
  """ Class to represent a shape.  This is essentially a list of Point
  objects
  """
  def __init__(self, pts = []):
    self.pts = pts
    self.num_pts = len(pts)

  def __add__(self, other):
    """ Operator overloading so that we can add one shape to another
    """
    s = Shape([])
    for i,p in enumerate(self.pts):
      s.add_point(p + other.pts[i])
    return s

  def __div__(self, i):
    """ Division by a constant.
    Each point gets divided by i
    """
    s = Shape([])
    for p in self.pts:
      s.add_point(p/i)
    return s

  def __eq__(self, other):
    for i in range(len(self.pts)):
      if self.pts[i] != other.pts[i]:
        return False
    return True

  def __ne__(self, other):
    return not self.__eq__(other)

  def add_point(self, p):
    self.pts.append(p)
    self.num_pts += 1

  def transform(self, t):
    s = Shape([])
    for p in self.pts:
      s.add_point(p + t)
    return s

  """ Helper methods for shape alignment """
  def __get_X(self, w):
    return sum([w[i]*self.pts[i].x for i in range(len(self.pts))])
  def __get_Y(self, w):
    return sum([w[i]*self.pts[i].y for i in range(len(self.pts))])
  def __get_Z(self, w):
    return sum([w[i]*(self.pts[i].x**2+self.pts[i].y**2) for i in range(len(self.pts))])
  def __get_C1(self, w, s):
    return sum([w[i]*(s.pts[i].x*self.pts[i].x + s.pts[i].y*self.pts[i].y) \
        for i in range(len(self.pts))])
  def __get_C2(self, w, s):
    return sum([w[i]*(s.pts[i].y*self.pts[i].x - s.pts[i].x*self.pts[i].y) \
        for i in range(len(self.pts))])

  def get_alignment_params(self, s, w):
    """ Gets the parameters required to align the shape to the given shape
    using the weight matrix w.  This applies a scaling, transformation and
    rotation to each point in the shape to align it as closely as possible
    to the shape.

    This relies on some linear algebra which we use numpy to solve.

    [ X2 -Y2   W   0][ax]   [X1]
    [ Y2  X2   0   W][ay] = [Y1]
    [ Z    0  X2  Y2][tx]   [C1]
    [ 0    Z -Y2  X2][ty]   [C2]

    We want to solve this to find ax, ay, tx, and ty

    :param shape: The shape to align to
    :param w: The weight matrix
    :return x: [ax, ay, tx, ty]
    """

    X1 = s.__get_X(w)
    X2 = self.__get_X(w)
    Y1 = s.__get_Y(w)
    Y2 = self.__get_Y(w)
    Z = self.__get_Z(w)
    W = sum(w)
    C1 = self.__get_C1(w, s)
    C2 = self.__get_C2(w, s)

    a = np.array([[ X2, -Y2,   W,  0],
                  [ Y2,  X2,   0,  W],
                  [  Z,   0,  X2, Y2],
                  [  0,   Z, -Y2, X2]])

    b = np.array([X1, Y1, C1, C2])
    # Solve equations
    # result is [ax, ay, tx, ty]
    return np.linalg.solve(a, b)

  def apply_params_to_shape(self, p):
    new = Shape([])
    # For each point in current shape
    for pt in self.pts:
      new_x = (p[0]*pt.x - p[1]*pt.y) + p[2]
      new_y = (p[1]*pt.x + p[0]*pt.y) + p[3]
      new.add_point(Point(new_x, new_y))
    return new

  def align_to_shape(self, s, w):
    p = self.get_alignment_params(s, w)
    return self.apply_params_to_shape(p)

  def get_vector(self):
    vec = np.zeros((self.num_pts, 2))
    for i in range(len(self.pts)):
      vec[i,:] = [self.pts[i].x, self.pts[i].y]
    return vec.flatten()

  def get_normal_to_point(self, p_num):
    # Normal to first point
    x = 0; y = 0; mag = 0
    if p_num == 0:
      x = self.pts[1].x - self.pts[0].x
      y = self.pts[1].y - self.pts[0].y
    # Normal to last point
    elif p_num == len(self.pts)-1:
      x = self.pts[-1].x - self.pts[-2].x
      y = self.pts[-1].y - self.pts[-2].y
    # Must have two adjacent points, so...
    else:
      x = self.pts[p_num+1].x - self.pts[p_num-1].x
      y = self.pts[p_num+1].y - self.pts[p_num-1].y
    mag = math.sqrt(x**2 + y**2)
    return (-y/mag, x/mag)

  @staticmethod
  def from_vector(vec):
    s = Shape([])
    for i,j in np.reshape(vec, (-1,2)):
      s.add_point(Point(i, j))
    return s

class ShapeViewer ( object ):
  """ Provides functionality to display a shape in a window
  """
  @staticmethod
  def show_shapes(shapes):
    """ Function to show all of the shapes which are passed to it
    """
    cv.NamedWindow("Shape Model", cv.CV_WINDOW_AUTOSIZE)
    # Get size for the window
    max_x = int(max([pt.x for shape in shapes for pt in shape.pts]))
    max_y = int(max([pt.y for shape in shapes for pt in shape.pts]))
    min_x = int(min([pt.x for shape in shapes for pt in shape.pts]))
    min_y = int(min([pt.y for shape in shapes for pt in shape.pts]))

    i = cv.CreateImage((max_x-min_x+20, max_y-min_y+20), cv.IPL_DEPTH_8U, 3)
    cv.Set(i, (0, 0, 0))
    for shape in shapes:
      r = randint(0, 255)
      g = randint(0, 255)
      b = randint(0, 255)
      #r = 0
      #g = 0
      #b = 0
      for pt_num, pt in enumerate(shape.pts):
        # Draw normals
        #norm = shape.get_normal_to_point(pt_num)
        #cv.Line(i,(pt.x-min_x,pt.y-min_y), \
        #    (norm[0]*10 + pt.x-min_x, norm[1]*10 + pt.y-min_y), (r, g, b))
        cv.Circle(i, (int(pt.x-min_x), int(pt.y-min_y)), 2, (r, g, b), -1)
    cv.ShowImage("Shape Model",i)

  @staticmethod
  def show_modes_of_variation(model, mode):
    # Get the limits of the animation
    start = -2*math.sqrt(model.evals[mode])
    stop = -start
    step = (stop - start) / 100

    b_all = np.zeros(model.modes)
    b = start
    while True:
      b_all[mode] = b
      s = model.generate_example(b_all)
      ShapeViewer.show_shapes([s])
      # Reverse direction when we get to the end to keep it running
      if (b < start and step < 0) or (b > stop and step > 0):
        step = -step
      b += step
      c = cv.WaitKey(10)
      if chr(255&c) == 'q': break

  @staticmethod
  def draw_model_fitter(f):
    cv.NamedWindow("Model Fitter", cv.CV_WINDOW_AUTOSIZE)
    # Copy image
    i = cv.CreateImage(cv.GetSize(f.image), f.image.depth, 3)
    cv.Copy(f.image, i)
    for pt_num, pt in enumerate(f.shape.pts):
      # Draw normals
      cv.Circle(i, (int(pt.x), int(pt.y)), 2, (0,0,0), -1)
    cv.ShowImage("Shape Model",i)
    cv.WaitKey()

class PointsReader ( object ):
  """ Class to read from files provided on Tim Cootes's website."""
  @staticmethod
  def read_points_file(filename):
    """ Read a .pts file, and returns a Shape object """
    s = Shape([])
    num_pts = 0
    with open(filename) as fh:
      # Get expected number of points from file
      first_line = fh.readline()
      if first_line.startswith("version"):
        # Then it is a newer type of file...
        num_pts = int(fh.readline().split()[1])
        # Drop the {
        fh.readline()
      else:
        # It is an older file...
        num_pts = int(first_line)
      for line in fh:
        if not line.startswith("}"):
          pt = line.strip().split()
          s.add_point(Point(float(pt[0]), float(pt[1])))
    if s.num_pts != num_pts:
      print "Unexpected number of points in file.  "\
      "Expecting %d, got %d" % (num_pts, s.num_pts)
    return s

  @staticmethod
  def read_directory(dirname):
    """ Reads an entire directory of .pts files and returns
    them as a list of shapes
    """
    pts = []
    for file in glob.glob(os.path.join(dirname, "*.pts")):
      pts.append(PointsReader.read_points_file(file))
    return pts

class ModelFitter:
  """
  Class to fit a model to an image

  :param asm: A trained active shape model
  :param image: An OpenCV image
  :param t: A transformation to move the shape to a new origin
  """
  def __init__(self, asm, image, t=Point(0.0,0.0)):
    self.image = image
    self.g_image = []
    for i in range(0,4):
      self.g_image.append(self.__produce_gradient_image(image, 2**i))
    self.asm = asm
    # Copy mean shape as starting shape and transform it to origin
    self.shape = Shape.from_vector(asm.mean).transform(t)
    # And resize shape to fit image if required
    if self.__shape_outside_image(self.shape, self.image):
      self.shape = self.__resize_shape_to_fit_image(self.shape, self.image)

  def __shape_outside_image(self, s, i):
    for p in s.pts:
      if p.x > i.width or p.x < 0 or p.y > i.height or p.y < 0:
        return True
    return False

  def __resize_shape_to_fit_image(self, s, i):
    # Get rectagonal boundary orf shape
    min_x = min([pt.x for pt in s.pts])
    min_y = min([pt.y for pt in s.pts])
    max_x = max([pt.x for pt in s.pts])
    max_y = max([pt.y for pt in s.pts])

    # If it is outside the image then we'll translate it back again
    if min_x > i.width: min_x = 0
    if min_y > i.height: min_y = 0
    ratio_x = (i.width-min_x) / (max_x - min_x)
    ratio_y = (i.height-min_y) / (max_y - min_y)
    new = Shape([])
    for pt in s.pts:
      new.add_point(Point(pt.x*ratio_x if ratio_x < 1 else pt.x, \
                          pt.y*ratio_y if ratio_y < 1 else pt.y))
    return new

  def __produce_gradient_image(self, i, scale):
    size = cv.GetSize(i)
    grey_image = cv.CreateImage(size, 8, 1)

    size = [s/scale for s in size]
    grey_image_small = cv.CreateImage(size, 8, 1)

    cv.CvtColor(i, grey_image, cv.CV_RGB2GRAY)

    df_dx = cv.CreateImage(cv.GetSize(i), cv.IPL_DEPTH_16S, 1)
    cv.Sobel( grey_image, df_dx, 1, 1)
    cv.Convert(df_dx, grey_image)
    cv.Resize(grey_image, grey_image_small)#, interpolation=cv.CV_INTER_NN)
    cv.Resize(grey_image_small, grey_image)#, interpolation=cv.CV_INTER_NN)
    return grey_image

  def do_iteration(self, scale):
    """ Does a single iteration of the shape fitting algorithm.
    This is useful when we want to show the algorithm converging on
    an image

    :return shape: The shape in its current orientation
    """

    # Build new shape from max points along normal to current
    # shape
    s = Shape([])
    for i, pt in enumerate(self.shape.pts):
      s.add_point(self.__get_max_along_normal(i, scale))

    new_s = s.align_to_shape(Shape.from_vector(self.asm.mean), self.asm.w)

    var = new_s.get_vector() - self.asm.mean
    new = self.asm.mean
    for i in range(len(self.asm.evecs.T)):
      b = np.dot(self.asm.evecs[:,i],var)
      max_b = 2*math.sqrt(self.asm.evals[i])
      b = max(min(b, max_b), -max_b)
      new = new + self.asm.evecs[:,i]*b

    self.shape = Shape.from_vector(new).align_to_shape(s, self.asm.w)

  def __get_max_along_normal(self, p_num, scale):
    """ Gets the max edge response along the normal to a point

    :param p_num: Is the number of the point in the shape
    """

    norm = self.shape.get_normal_to_point(p_num)
    p = self.shape.pts[p_num]

    # Find extremes of normal within the image
    # Test x first
    min_t = -p.x / norm[0]
    if p.y + min_t*norm[1] < 0:
      min_t = -p.y / norm[1]
    elif p.y + min_t*norm[1] > self.image.height:
      min_t = (self.image.height - p.y) / norm[1]

    # X first again
    max_t = (self.image.width - p.x) / norm[0]
    if p.y + max_t*norm[1] < 0:
      max_t = -p.y / norm[1]
    elif p.y + max_t*norm[1] > self.image.height:
      max_t = (self.image.height - p.y) / norm[1]

    # Swap round if max is actually larger...
    tmp = max_t
    max_t = max(min_t, max_t)
    min_t = min(min_t, tmp)

    # Get length of the normal within the image
    x1 = min(p.x+max_t*norm[0], p.x+min_t*norm[0])
    x2 = max(p.x+max_t*norm[0], p.x+min_t*norm[0])
    y1 = min(p.y+max_t*norm[1], p.y+min_t*norm[1])
    y2 = max(p.y+max_t*norm[1], p.y+min_t*norm[1])
    l = math.sqrt((x2-x1)**2 + (y2-y1)**2)

    img = cv.CreateImage(cv.GetSize(self.image), self.g_image[scale].depth, 1)
    cv.Copy(self.g_image[scale], img)
    #cv.Circle(img, \
    #    (int(norm[0]*min_t + p.x), int(norm[1]*min_t + p.y)), \
    #    5, (0, 0, 0))
    #cv.Circle(img, \
    #    (int(norm[0]*max_t + p.x), int(norm[1]*max_t + p.y)), \
    #    5, (0, 0, 0))

    # Scan over the whole line
    max_pt = p
    max_edge = 0

    # Now check over the vector
    #v = min(max_t, -min_t)
    #for t in drange(min_t, max_t, (max_t-min_t)/l):
    search = 20+scale*10
    # Look 6 pixels to each side too
    for side in range(-6, 6):
      # Normal to normal...
      new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])
      for t in drange(-search if -search > min_t else min_t, \
                       search if search < max_t else max_t , 1):

        x = int(norm[0]*t + new_p.x)
        y = int(norm[1]*t + new_p.y)
        if x < 0 or x > self.image.width or y < 0 or y > self.image.height:
          continue
#        cv.Circle(img, (x, y), 3, (100,100,100))
        #print x, y, self.g_image.width, self.g_image.height
        if self.g_image[scale][y-1, x-1] > max_edge:
          max_edge = self.g_image[scale][y-1, x-1]
          max_pt = Point(new_p.x + t*norm[0], new_p.y + t*norm[1])

#    for point in self.shape.pts:
#      cv.Circle(img, (int(point.x), int(point.y)), 3, (255,255,255))
##
#    cv.Circle(img, (int(max_pt.x), int(max_pt.y)), 3, (255,255,255))
##
#    cv.NamedWindow("Scale", cv.CV_WINDOW_AUTOSIZE)
#    cv.ShowImage("Scale",img)
#    cv.WaitKey()
#
    return max_pt

class ActiveShapeModel:
  """
  """
  def __init__(self, shapes = []):
    self.shapes = shapes
    # Make sure the shape list is valid
    self.__check_shapes(shapes)
    # Create weight matrix for points
    print "Calculating weight matrix..."
    self.w = self.__create_weight_matrix(shapes)
    # Align all shapes
    print "Aligning shapes with Procrustes analysis..."
    self.shapes = self.__procrustes(shapes)
    print "Constructing model..."
    # Initialise this in constructor
    (self.evals, self.evecs, self.mean, self.modes) = \
        self.__construct_model(self.shapes)

  def __check_shapes(self, shapes):
    """ Method to check that all shapes have the correct number of
    points """
    if shapes:
      num_pts = shapes[0].num_pts
      for shape in shapes:
        if shape.num_pts != num_pts:
          raise Exception("Shape has incorrect number of points")

  def __get_mean_shape(self, shapes):
    s = shapes[0]
    for shape in shapes[1:]:
      s = s + shape
    return s / len(shapes)

  def __construct_model(self, shapes):
    """ Constructs the shape model
    """
    shape_vectors = np.array([s.get_vector() for s in self.shapes])
    mean = np.mean(shape_vectors, axis=0)

    # Move mean to the origin
    # FIXME Clean this up...
    mean = np.reshape(mean, (-1,2))
    min_x = min(mean[:,0])
    min_y = min(mean[:,1])

    #mean = np.array([pt - min(mean[:,i]) for i in [0,1] for pt in mean[:,i]])
    #mean = np.array([pt - min(mean[:,i]) for pt in mean for i in [0,1]])
    mean[:,0] = [x - min_x for x in mean[:,0]]
    mean[:,1] = [y - min_y for y in mean[:,1]]
    #max_x = max(mean[:,0])
    #max_y = max(mean[:,1])
    #mean[:,0] = [x/(2) for x in mean[:,0]]
    #mean[:,1] = [y/(3) for y in mean[:,1]]
    mean = mean.flatten()
    #print mean

    # Produce covariance matrix
    cov = np.cov(shape_vectors, rowvar=0)
    # Find eigenvalues/vectors of the covariance matrix
    evals, evecs = np.linalg.eig(cov)

    # Find number of modes required to describe the shape accurately
    t = 0
    for i in range(len(evals)):
      if sum(evals[:i]) / sum(evals) < 0.99:
        t = t + 1
      else: break
    print "Constructed model with %d modes of variation" % t
    return (evals[:t], evecs[:,:t], mean, t)

  def generate_example(self, b):
    """ b is a vector of floats to apply to each mode of variation
    """
    # Need to make an array same length as mean to apply to eigen
    # vectors
    full_b = np.zeros(len(self.mean))
    for i in range(self.modes): full_b[i] = b[i]

    p = self.mean
    for i in range(self.modes): p = p + full_b[i]*self.evecs[:,i]

    # Construct a shape object
    return Shape.from_vector(p)

  def __procrustes(self, shapes):
    """ This function aligns all shapes passed as a parameter by using
    Procrustes analysis

    :param shapes: A list of Shape objects
    """
    # First rotate/scale/translate each shape to match first in set
    shapes[1:] = [s.align_to_shape(shapes[0], self.w) for s in shapes[1:]]

    # Keep hold of a shape to align to each iteration to allow convergence
    a = shapes[0]
    trans = np.zeros((4, len(shapes)))
    converged = False
    current_accuracy = sys.maxint
    while not converged:
      # Now get mean shape
      mean = self.__get_mean_shape(shapes)
      # Align to shape to stop it diverging
      mean = mean.align_to_shape(a, self.w)
      # Now align all shapes to the mean
      for i in range(len(shapes)):
        # Get transformation required for each shape
        trans[:, i] = shapes[i].get_alignment_params(mean, self.w)
        # Apply the transformation
        shapes[i] = shapes[i].apply_params_to_shape(trans[:,i])

      # Test if the average transformation required is very close to the
      # identity transformation and stop iteration if it is
      accuracy = np.mean(np.array([1, 0, 0, 0]) - np.mean(trans, axis=1))**2
      # If the accuracy starts to decrease then we have reached limit of precision
      # possible
      if accuracy > current_accuracy: converged = True
      else: current_accuracy = accuracy
    return shapes

  def __create_weight_matrix(self, shapes):
    """ Private method to produce the weight matrix which corresponds
    to the training shapes

    :param shapes: A list of Shape objects
    :return w: The matrix of weights produced from the shapes
    """
    # Return empty matrix if no shapes
    if not shapes:
      return np.array()
    # First get number of points of each shape
    num_pts = shapes[0].num_pts

    # We need to find the distance of each point to each
    # other point in each shape.
    distances = np.zeros((len(shapes), num_pts, num_pts))
    for s, shape in enumerate(shapes):
      for k in range(num_pts):
        for l in range(num_pts):
          distances[s, k, l] = shape.pts[k].dist(shape.pts[l])

    # Create empty weight matrix
    w = np.zeros(num_pts)
    # calculate range for each point
    for k in range(num_pts):
      for l in range(num_pts):
        # Get the variance in distance of that point to other points
        # for all shapes
        w[k] += np.var(distances[:, k, l])
    # Invert weights
    return 1/w
