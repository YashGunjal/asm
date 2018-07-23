#!/usr/bin/env python

from Active_shape_models import *

def main():
  shapes = PointsReader.read_directory(sys.argv[1])
  a = ActiveShapeModel(shapes)
  # load the image
  i = cv.LoadImage(sys.argv[2])
  m = ModelFitter(a, i)
  ShapeViewer.draw_model_fitter(m)

  for i in range(1):
    m.do_iteration(3)
    ShapeViewer.draw_model_fitter(m)
  for i in range(1):
    m.do_iteration(2)
    ShapeViewer.draw_model_fitter(m)
  for i in range(10):
    m.do_iteration(3)
    ShapeViewer.draw_model_fitter(m)
  for j in range(100):
    m.do_iteration(0)
    ShapeViewer.draw_model_fitter(m)


if __name__ == "__main__":
  main()

