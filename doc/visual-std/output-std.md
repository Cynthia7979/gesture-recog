# Output Standard 

This file clarifies the standard of the model's runtime output.

  For making a visual report, we need to plot a diagram about the change of loss
per epoch. There are many ways drawing the diagram, but a standard could let us
to have the ability to use various ways to implement the plotting, which could
give a good interface.  



---

## The Standard

Try to follow this standard when training a model:

- Every data should be in a particular file of a particular directory.

- The format must be simple enough for a program (which uses simple IO built-ins,
  some special ways does not count. e.g. json is not included) to read

