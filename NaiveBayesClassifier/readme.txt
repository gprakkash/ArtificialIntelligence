Name: Gaurav Prakash

Programming language used: Java

compilation instuctions:
run javac NaiveBayesClassifier.java

to run the code:
copy the <training_file> and <test_file> in the same directory

then, run:
java NaiveBayesClassifier <training_file> <test_file> histograms <number>
java NaiveBayesClassifier <training_file> <test_file> gaussians
java NaiveBayesClassifier <training_file> <test_file> mixtures <number>

Note: For mixtures, it takes 2-3 minutes to output the result, if number argument is passed as 5. The time varies as per the value of number. Meanwhile, you will notice "Processing..." message on the console.

File List:
NaiveBayesClassifier.java
Histogram.java
Bin.java
Gaussian.java
MixtureOfGaussians.java

