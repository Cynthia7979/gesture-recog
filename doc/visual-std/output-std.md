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

Specify:

1. Under the training directory, make a directory named *visual-loss*
2. Create a file named like this: *number of epoch* + "_all"
   - **Ex:** 0_all (the first epoch)
3. All **loss** values of this epoch need to be written into the file,
   each **loss** values are separated by a **line break**
4. Do this job for every epoch

NOTE: If you could, please create a file called *"loss_all"*. Calculate the
      the average loss for every epoch and write them into the file, separate
      the average of each epochs by a **line break**
NOTE: All data must be stored in ***text***

