This archive contains a nested set of training data for a web page
categorization benchmark. Each file is in .dst format, which will be
described below. The benchmark task is to predict whether web page belongs
to a category based on the presence of 300 selected keywords on the page.

There are 8 training files, each with a different number of training
examples:

   Filename          Number of Examples
   ========          ==================
   web-1a.dst             2477
   web-2a.dst             3470
   web-3a.dst             4912
   web-4a.dst             7366
   web-5a.dst             9888
   web-6a.dst            17188
   web-7a.dst            24692
   web-a.dst             49749

There are two benchmarks in the technical report. The first is to train a
linear SVM with C = 1. The second is to train a Gaussian SVM with C = 5,
with Gaussian variance of 10. These parameters were chosen to maximize
performance on a validation set. The timing results are given in the technical
report.

To make sure that the benchmark is run properly, check the number of bound
and number of non-bound Lagrange multipliers on the tables listed in the
technical report. Another check is to print out the threshold (b) for the
trained SVM. The thresholds should be very close to the following numbers:

Linear SVM:

   Filename          Threshold
   ========          =========
   web-1a.dst        1.08553
   web-2a.dst        1.10861
   web-3a.dst        1.06354
   web-4a.dst        1.07142
   web-5a.dst        1.08431
   web-6a.dst        1.02703
   web-7a.dst        1.02946
   web-a.dst         1.03446

Gaussian SVM:

   Filename          Threshold
   ========          =========
   web-1a.dst         0.177057
   web-2a.dst         0.0116676
   web-3a.dst        -0.0161608
   web-4a.dst        -0.0329898
   web-5a.dst        -0.0722572
   web-6a.dst        -0.19304
   web-7a.dst        -0.242451
   web-a.dst         -0.3587

===========
.DST FORMAT
===========

The .dst file format is very simple: comma delimited lines of text.

Lines that start with "A" describe a variable, whose id number is the
second field in the line. The third field in the line is either 101 (if the
variable is a class label) or 1 (if the variable is not). For this
benchmark, all you need to use is that variable id 1000 is the class
label, while variable ids from 1001 to 1300 are input attributes.

Lines that start with "C" indicate the start of a new training example. The
rest of the line identifies the training example.

Lines that start with "V" are attributes of the current example. The second
field is the variable id (see "A" lines above), and the third field is the
value of the variable. If a variable is not specified by a "V" line, its
value defaults to 0. 

For example, in this benchmark, if an example has a line that says 

    V,1000,0

it means that the example has a negative label. If the following line
appears:

    V,1015,1

it means that input attribute #15 is true (1). If no such line appears
before the next "C" line, it means that input attribute #15 is false (0).


