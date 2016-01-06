import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NaiveBayesClassifier {
	public static final String HISTOGRAMS = "histograms";
	public static final String GAUSSIANS = "gaussians";
	public static final String MIXTURES = "mixtures";
	private static String trainingFile;
	private static String testFile;
	private static String approximateMethod; // for estimating probability
												// distribution
	private static int numOfAttributes = -1;
	private static int numOfClasses = -1;
	private static int numOfBins = -1;
	private static int numOfGaussians = -1;
	private static List<Double> minList = null;
	private static List<Double> maxList = null;
	private static List<Double> probDistributionOfClass = null;
	private static List<Histogram> histograms = null;
	private static List<List<Gaussian>> gaussians = null;
	private static List<List<MixtureOfGaussians>> mixtures = null;

	public static void main(String[] args) {
		readArguments(args);
		List<List<Double>> trainingExamples = readFile(trainingFile);
		List<List<Double>> testExamples = readFile(testFile);
		numOfClasses = getNumberOfClasses(trainingExamples);

		if (approximateMethod.equals(HISTOGRAMS)) {

			histograms = trainingPhaseForHistograms(trainingExamples);
			classificationPhase(testExamples);

		} else if (approximateMethod.equals(GAUSSIANS)) {
			gaussians = trainingPhaseForGaussians(trainingExamples);
			classificationPhase(testExamples);
		} else if (approximateMethod.equals(MIXTURES)) {

			mixtures = trainingPhaseForMixtures(trainingExamples);
			classificationPhase(testExamples);
		} else {
			System.out.println("Invalid approximate method");
			System.exit(1);
		}
	}

	private static List<List<MixtureOfGaussians>> trainingPhaseForMixtures(List<List<Double>> examples) {
		System.out.print("Processing...\n");
		// generating Probability distribution of Classes
		List<Integer> classDistribution = generateClassDistribution(examples);
		generateProbDistributionOfClass(classDistribution, examples.size());

		List<List<MixtureOfGaussians>> mixtures = new ArrayList<List<MixtureOfGaussians>>();
		// storing examples in the form of
		// listOfAttrbutes<listOfClasses<listOfVals>>>
		List<List<List<Double>>> data = new ArrayList<List<List<Double>>>();

		minList = new ArrayList<Double>();
		maxList = new ArrayList<Double>();

		// initializing training data
		for (int i = 0; i < numOfAttributes; i++) {
			List<List<Double>> listOfClasses = new ArrayList<List<Double>>();
			for (int j = 0; j < numOfClasses; j++) {
				List<Double> listOfVals = new ArrayList<Double>();
				listOfClasses.add(listOfVals);
			}
			data.add(listOfClasses);
		}

		// read each training example and form a List<List<Double>> to represent
		// a List<Double> for each class and each attribute
		for (int i = 0; i < examples.size(); i++) {
			// changes made to pattern will reflect in the examples list,
			// although we don't make any changes
			List<Double> pattern = getPattern(examples.get(i));
			int myClass = getClass(examples.get(i));

			// copy the value of each attribute to corresponding
			// List<List<Double>>
			for (int attribute = 0; attribute < numOfAttributes; attribute++) {
				double value = pattern.get(attribute);
				data.get(attribute).get(myClass).add(value);
			}
		}

		for (int attribute = 0; attribute < numOfAttributes; attribute++) {
			List<MixtureOfGaussians> listOfMixtureOfGaussians = new ArrayList<MixtureOfGaussians>();
			for (int myClass = 0; myClass < numOfClasses; myClass++) {
				// Initialize k Gaussians
				MixtureOfGaussians mixtureOfGaussians = initializeMixtureOfGaussians(data.get(attribute).get(myClass));
				expectationMaximization(mixtureOfGaussians, data.get(attribute).get(myClass));
				listOfMixtureOfGaussians.add(mixtureOfGaussians);
			}
			mixtures.add(listOfMixtureOfGaussians);
		}
		printMixtureOfGaussians(mixtures);
		return mixtures;
	}

	private static void printMixtureOfGaussians(List<List<MixtureOfGaussians>> mixtures) {
		MixtureOfGaussians mixtureOfGaussians;
		for (int myClass = 0; myClass < numOfClasses; myClass++) {
			for (int attribute = 0; attribute < numOfAttributes; attribute++) {
				mixtureOfGaussians = mixtures.get(attribute).get(myClass);
				for (int gaussianNum = 0; gaussianNum < mixtureOfGaussians.getMixture().size(); gaussianNum++) {
					Gaussian gaussian = mixtureOfGaussians.getMixture().get(gaussianNum);
					System.out.printf("Class %d, attribute %d, mean = %.2f, std = %.2f\n", myClass, attribute,
							gaussian.getMean(), gaussian.getStd());
				}
			}
		}
	}

	private static void expectationMaximization(MixtureOfGaussians mixtureOfGaussians, List<Double> listOfValues) {
		if (listOfValues.size() == 0)
			return;
		for (int i = 0; i < 50; i++) {
			mixtureOfGaussians.setProbij(eStep(mixtureOfGaussians, listOfValues));
			mStep(mixtureOfGaussians, listOfValues);
		}
	}

	private static void mStep(MixtureOfGaussians mixtureOfGaussians, List<Double> listOfValues) {
		for (int i = 0; i < numOfGaussians; i++) {
			Gaussian gaussian = mixtureOfGaussians.getMixture().get(i);
			List<Double> probDistribution = mixtureOfGaussians.getProbij().get(i);
			double mean, std, weight;
			mean = getMean(listOfValues, probDistribution);
			std = getStd(listOfValues, mean, probDistribution);
			// to prevent the std to get very close to zero
			if (std < 0.1)
				std = 0.1;
			weight = getWeight(probDistribution, mixtureOfGaussians.getProbij());
			// updating the mixtureOfGaussians
			gaussian.setMean(mean);
			gaussian.setStd(std);
			mixtureOfGaussians.getWeights().set(i, weight);
		}
	}

	private static double getWeight(List<Double> probDistribution, List<List<Double>> probij) {
		double weight;
		double numerator = 0, denominator = 0;
		for (int j = 0; j < probDistribution.size(); j++) {
			numerator = numerator + probDistribution.get(j);

			for (int k = 0; k < probij.size(); k++) {
				for (int l = 0; l < probij.get(k).size(); l++) {
					denominator = denominator + probij.get(k).get(l);
				}
			}
		}
		weight = numerator / denominator;
		return weight;
	}

	private static double getMean(List<Double> listOfValues, List<Double> probDistribution) {
		double mean;
		double numerator = 0, denominator = 0;
		for (int j = 0; j < listOfValues.size(); j++) {
			numerator = numerator + probDistribution.get(j) * listOfValues.get(j);
			denominator = denominator + probDistribution.get(j);
		}
		mean = numerator / denominator;
		return mean;
	}

	private static double getStd(List<Double> listOfValues, double mean, List<Double> probDistribution) {
		double std;
		double numerator = 0, denominator = 0;
		for (int j = 0; j < listOfValues.size(); j++) {
			numerator = numerator + probDistribution.get(j) * Math.pow((listOfValues.get(j) - mean), 2);
			denominator = denominator + probDistribution.get(j);
		}
		std = Math.sqrt(numerator / denominator);
		return std;
	}

	private static List<List<Double>> eStep(MixtureOfGaussians mixtureOfGaussians, List<Double> listOfValues) {
		// indicates the probability that the ith Gaussian generated the jth
		// value
		List<List<Double>> probij = new ArrayList<List<Double>>();

		/*
		 * if (numOfGaussians != mixtureOfGaussians.getMixture().size()) {
		 * System.out.println(
		 * "numOfGaussians not same as mixtureOfGaussians size.");
		 * System.exit(1); }
		 */

		for (int i = 0; i < numOfGaussians; i++) {
			List<Double> probDistribution = new ArrayList<Double>();
			for (int j = 0; j < listOfValues.size(); j++) {
				Gaussian gaussian = mixtureOfGaussians.getMixture().get(i);
				double value = listOfValues.get(j);
				double weight = mixtureOfGaussians.getWeights().get(i);
				double prob = getProbOfValueGivenClassFrmGaussian(value, gaussian) * weight
						/ getProbOfValueGivenClassFrmMixture(value, mixtureOfGaussians);
				probDistribution.add(prob);
			}
			probij.add(probDistribution);
		}
		return probij;
	}

	private static double getProbOfValueGivenClassFrmMixture(double value, MixtureOfGaussians mixtureOfGaussians) {
		double probOfValueGivenClass = 0;

		for (int i = 0; i < numOfGaussians; i++) {
			Gaussian gaussian = mixtureOfGaussians.getMixture().get(i);
			double weight = mixtureOfGaussians.getWeights().get(i);
			probOfValueGivenClass = probOfValueGivenClass
					+ weight * getProbOfValueGivenClassFrmGaussian(value, gaussian);
		}
		return probOfValueGivenClass;
	}

	private static double getProbOfValueGivenClassFrmGaussian(double value, Gaussian gaussian) {

		double mean = gaussian.getMean();
		double std = gaussian.getStd();
		// N(x)=( (1/(std*sqrt(2*pi))) * pow(e, -pow((x-mean), 2)/(2*pow(std,
		// 2))) )
		double probOfValueGivenClass;
		if (std > 0)
			probOfValueGivenClass = (1 / (std * Math.sqrt(2 * Math.PI)))
					* (Math.pow(Math.E, -(Math.pow(value - mean, 2) / (2 * std * std))));
		else
			probOfValueGivenClass = 0.0;
		return probOfValueGivenClass;
	}

	private static MixtureOfGaussians initializeMixtureOfGaussians(List<Double> listOfValues) {
		double min, max;
		if (listOfValues.size() == 0) {
			min = 0;
			max = 0;
		} else {
			min = getMinOfList(listOfValues);
			max = getMaxOfList(listOfValues);
		}
		double g = (max - min) / numOfGaussians;
		MixtureOfGaussians mixtureOfGaussians = new MixtureOfGaussians();
		List<Double> weights = new ArrayList<Double>();
		List<Gaussian> mixture = new ArrayList<Gaussian>();

		for (int k = 0; k < numOfGaussians; k++) {
			Gaussian gaussian = new Gaussian();
			gaussian.setMean(min + (k * g) + (g / 2));
			gaussian.setStd(1);

			mixture.add(gaussian);

			weights.add((double) 1 / numOfGaussians);
		}
		mixtureOfGaussians.setMixture(mixture);
		mixtureOfGaussians.setWeights(weights);

		return mixtureOfGaussians;
	}

	private static double getMinOfList(List<Double> listOfValues) {
		double min = listOfValues.get(0);
		for (int i = 0; i < listOfValues.size(); i++) {
			if (listOfValues.get(i) < min)
				min = listOfValues.get(i);
		}
		return min;
	}

	private static double getMaxOfList(List<Double> listOfValues) {
		double max = listOfValues.get(0);
		for (int i = 0; i < listOfValues.size(); i++) {
			if (listOfValues.get(i) > max)
				max = listOfValues.get(i);
		}
		return max;
	}

	private static List<List<Gaussian>> trainingPhaseForGaussians(List<List<Double>> examples) {
		// variable declarations
		List<List<Gaussian>> gaussians = new ArrayList<List<Gaussian>>();
		List<List<List<Double>>> data = new ArrayList<List<List<Double>>>();
		List<Integer> classDistribution = generateClassDistribution(examples);
		generateProbDistributionOfClass(classDistribution, examples.size());

		// initializing training data
		for (int i = 0; i < numOfAttributes; i++) {
			List<List<Double>> listOfClasses = new ArrayList<List<Double>>();
			for (int j = 0; j < numOfClasses; j++) {
				List<Double> listOfVals = new ArrayList<Double>();
				listOfClasses.add(listOfVals);
			}
			data.add(listOfClasses);
		}

		// read each training example and form a List<List<Double>> to represent
		// a List<Double> for each class and each attribute
		for (int i = 0; i < examples.size(); i++) {
			// changes made to pattern will reflect in the examples list,
			// although we don't make any changes
			List<Double> pattern = getPattern(examples.get(i));
			int myClass = getClass(examples.get(i));

			// copy the value of each attribute to corresponding
			// List<List<Double>>
			for (int attribute = 0; attribute < numOfAttributes; attribute++) {
				double value = pattern.get(attribute);
				data.get(attribute).get(myClass).add(value);
			}
		}

		for (int attribute = 0; attribute < numOfAttributes; attribute++) {
			List<Gaussian> gaussiansForAttribute = new ArrayList<Gaussian>();
			for (int myClass = 0; myClass < numOfClasses; myClass++) {
				List<Double> values = new ArrayList<Double>();
				values = data.get(attribute).get(myClass);
				double mean = getMean(data.get(attribute).get(myClass));
				double std = getStd(mean, values);
				Gaussian gaussian = new Gaussian();
				gaussian.setMean(mean);
				gaussian.setStd(std);
				gaussiansForAttribute.add(gaussian);
			}
			gaussians.add(gaussiansForAttribute);
		}

		printGaussians(gaussians);
		return gaussians;
	}

	private static void printGaussians(List<List<Gaussian>> gaussians) {
		Gaussian gaussian;
		for (int myClass = 0; myClass < numOfClasses; myClass++) {
			for (int attribute = 0; attribute < numOfAttributes; attribute++) {
				gaussian = gaussians.get(attribute).get(myClass);
				System.out.printf("Class %d, attribute %d, mean = %.2f, std = %.2f\n", myClass, attribute,
						gaussian.getMean(), gaussian.getStd());
			}
		}

	}

	private static double getMean(List<Double> list) {
		double sum = 0;
		double mean;
		for (int i = 0; i < list.size(); i++) {
			sum += list.get(i);
		}
		if (list.size() != 0)
			mean = sum / list.size();
		else
			mean = 0;
		return mean;
	}

	private static double getStd(double mean, List<Double> list) {
		double variance = 0;
		double std;
		for (int i = 0; i < list.size(); i++) {
			variance += Math.pow((list.get(i) - mean), 2);
		}
		if (list.size() >= 2)
			variance = variance / (list.size() - 1);
		else
			variance = 0;
		std = Math.sqrt(variance);
		return std;
	}

	private static void classificationPhase(List<List<Double>> examples) {
		int exampleId = 0;
		double accuracy;
		double accuracySum = 0.0;
		int predictedClass, trueClass;
		double probability;

		for (int i = 0; i < examples.size(); i++) {
			List<Double> pattern = getPattern(examples.get(i));
			trueClass = getClass(examples.get(i));
			List<Double> probDistributionOfClassGivenPattern = null;

			// different functions are called according to the approximate
			// method chosen to find probability estimates
			if (approximateMethod.equals(HISTOGRAMS)) {
				probDistributionOfClassGivenPattern = getProbDistrOfClassGivenPatternFrmH(histograms, examples,
						pattern);
			} else if (approximateMethod.equals(GAUSSIANS)) {
				probDistributionOfClassGivenPattern = getProbDistrOfClassGivenPatternFrmG(gaussians, examples, pattern);
			} else if (approximateMethod.equals(MIXTURES)) {
				probDistributionOfClassGivenPattern = getProbDistrOfClassGivenPatternFrmM(mixtures, examples, pattern);
			}
			predictedClass = predictClass(probDistributionOfClassGivenPattern);
			probability = probDistributionOfClassGivenPattern.get(predictedClass);
			accuracy = getAccuracy(predictedClass, trueClass, probDistributionOfClassGivenPattern);
			accuracySum = accuracySum + accuracy;
			System.out.printf("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n", exampleId,
					predictedClass, probability, trueClass, accuracy);
			exampleId++;
		}
		System.out.printf("classification accuracy=%6.4f\n", accuracySum / examples.size());

	}

	private static List<Double> getProbDistrOfClassGivenPatternFrmM(List<List<MixtureOfGaussians>> mixtures,
			List<List<Double>> examples, List<Double> pattern) {
		List<Double> probDistributionOfClassGivenPattern = new ArrayList<Double>();

		for (int i = 0; i < numOfClasses; i++) {
			probDistributionOfClassGivenPattern.add(getProbOfClassGivenPatternFrmM(i, pattern, mixtures));
		}
		return probDistributionOfClassGivenPattern;
	}

	private static Double getProbOfClassGivenPatternFrmM(int myClass, List<Double> pattern,
			List<List<MixtureOfGaussians>> mixtures) {

		double probOfPatternGivenClass = getProbOfPatternGivenClassFrmM(pattern, myClass, mixtures);
		double probOfClass = probDistributionOfClass.get(myClass);
		double probOfPattern = 0;
		double probOfClassGivenPattern;
		for (int i = 0; i < numOfClasses; i++) {
			probOfPattern = probOfPattern
					+ getProbOfPatternGivenClassFrmM(pattern, i, mixtures) * probDistributionOfClass.get(i);
		}

		if (probOfPattern > 0.0)
			probOfClassGivenPattern = probOfPatternGivenClass * probOfClass / probOfPattern;
		else
			probOfClassGivenPattern = 0.0;
		return probOfClassGivenPattern;
	}

	private static double getProbOfPatternGivenClassFrmM(List<Double> pattern, int myClass,
			List<List<MixtureOfGaussians>> mixtures) {
		double probOfPatternGivenClass = 1;

		for (int attribute = 0; attribute < pattern.size(); attribute++) {
			double value = pattern.get(attribute);
			double probOfValueGivenClass = getProbOfValueGivenClassFrmMixture(value,
					mixtures.get(attribute).get(myClass));
			probOfPatternGivenClass = probOfPatternGivenClass * probOfValueGivenClass;
		}

		return probOfPatternGivenClass;
	}

	private static double getAccuracy(int predictedClass, int trueClass, List<Double> distribution) {
		int numOfTies = 0;
		if ((double) distribution.get(predictedClass) != (double) distribution.get(trueClass))
			return 0.0;
		else {
			numOfTies = getNumOfTies(distribution);
			if (numOfTies > 0)
				return (1.0 / (double) (numOfTies + 1));
			else
				return 1.0;
		}
	}

	private static int getNumOfTies(List<Double> distribution) {
		double probOfPredictedClass = distribution.get(predictClass(distribution));
		int numOfTies = -1;
		for (int i = 0; i < distribution.size(); i++) {
			if (distribution.get(i) == probOfPredictedClass)
				numOfTies++;// if there are no ties this statement will be
							// executed once
		}
		return numOfTies;
	}

	private static int predictClass(List<Double> distribution) {
		double maxProb = -1;
		Random rand = new Random();
		List<Integer> classesWithMaxProb = new ArrayList<Integer>();
		int predictedClass = -1;
		for (int i = 0; i < distribution.size(); i++) {
			if (distribution.get(i) > maxProb) {
				maxProb = distribution.get(i);
			}
		}
		for (int i = 0; i < distribution.size(); i++) {
			if (distribution.get(i) == maxProb) {
				classesWithMaxProb.add(i);
			}
		}
		predictedClass = classesWithMaxProb.get(rand.nextInt(classesWithMaxProb.size()));
		return predictedClass;
	}

	private static List<Double> getProbDistrOfClassGivenPatternFrmG(List<List<Gaussian>> gaussians,
			List<List<Double>> examples, List<Double> pattern) {
		List<Double> probDistributionOfClassGivenPattern = new ArrayList<Double>();

		for (int i = 0; i < numOfClasses; i++) {
			probDistributionOfClassGivenPattern.add(getProbOfClassGivenPatternFrmG(i, pattern, gaussians));
		}
		return probDistributionOfClassGivenPattern;
	}

	private static Double getProbOfClassGivenPatternFrmG(int myClass, List<Double> pattern,
			List<List<Gaussian>> gaussians) {
		double probOfPatternGivenClass = getProbOfPatternGivenClassFrmG(pattern, myClass, gaussians);
		double probOfClass = probDistributionOfClass.get(myClass);
		double probOfPattern = 0;
		double probOfClassGivenPattern;
		for (int i = 0; i < numOfClasses; i++) {
			probOfPattern = probOfPattern
					+ getProbOfPatternGivenClassFrmG(pattern, i, gaussians) * probDistributionOfClass.get(i);
		}

		if (probOfPattern > 0.0)
			probOfClassGivenPattern = probOfPatternGivenClass * probOfClass / probOfPattern;
		else
			probOfClassGivenPattern = 0.0;
		return probOfClassGivenPattern;
	}

	private static double getProbOfPatternGivenClassFrmG(List<Double> pattern, int myClass,
			List<List<Gaussian>> gaussians) {
		double probOfPatternGivenClass = 1;
		for (int attribute = 0; attribute < pattern.size(); attribute++) {
			double value = pattern.get(attribute);
			double probOfValueGivenClass = getProbOfValueGivenClassFrmG(attribute, value, myClass, gaussians);
			probOfPatternGivenClass = probOfPatternGivenClass * probOfValueGivenClass;
		}
		return probOfPatternGivenClass;
	}

	private static double getProbOfValueGivenClassFrmG(int attribute, double value, int myClass,
			List<List<Gaussian>> gaussians) {
		double mean = gaussians.get(attribute).get(myClass).getMean();
		double std = gaussians.get(attribute).get(myClass).getStd();
		// N(x)=( (1/(std*sqrt(2*pi))) * pow(e, -pow((x-mean), 2)/(2*pow(std,
		// 2))) )
		double probOfValueGivenClass;
		if (std > 0)
			probOfValueGivenClass = (1 / (std * Math.sqrt(2 * Math.PI)))
					* (Math.pow(Math.E, -(Math.pow(value - mean, 2) / (2 * std * std))));
		else
			probOfValueGivenClass = 0.0;
		// System.out.printf("P(%f | %d) = %f\n", value, myClass,
		// probOfValueGivenClass);
		return probOfValueGivenClass;
	}

	private static List<Double> getProbDistrOfClassGivenPatternFrmH(List<Histogram> histograms,
			List<List<Double>> examples, List<Double> pattern) {
		List<Double> probDistributionOfClassGivenPattern = new ArrayList<Double>();

		for (int i = 0; i < numOfClasses; i++) {
			probDistributionOfClassGivenPattern.add(getProbOfClassGivenPatternFrmH(i, pattern, histograms));
		}
		return probDistributionOfClassGivenPattern;
	}

	private static Double getProbOfClassGivenPatternFrmH(int myClass, List<Double> pattern,
			List<Histogram> histograms) {
		double probOfPatternGivenClass = getProbOfPatternGivenClassFrmH(pattern, myClass, histograms);
		double probOfClass = probDistributionOfClass.get(myClass);
		double probOfPattern = 0;
		double probOfClassGivenPattern;
		for (int i = 0; i < numOfClasses; i++) {
			probOfPattern = probOfPattern
					+ getProbOfPatternGivenClassFrmH(pattern, i, histograms) * probDistributionOfClass.get(i);
		}

		if (probOfPattern > 0.0)
			probOfClassGivenPattern = probOfPatternGivenClass * probOfClass / probOfPattern;
		else
			probOfClassGivenPattern = 0.0;
		return probOfClassGivenPattern;
	}

	private static double getProbOfPatternGivenClassFrmH(List<Double> pattern, int myClass,
			List<Histogram> histograms) {
		double probOfPatternGivenClass = 1;
		for (int attribute = 0; attribute < pattern.size(); attribute++) {
			double value = pattern.get(attribute);
			int bin = getBinNumber(value, minList.get(attribute), maxList.get(attribute));
			double probOfBinGivenClass = getProbOfBinGivenClassFrmH(attribute, bin, myClass, histograms);
			probOfPatternGivenClass = probOfPatternGivenClass * probOfBinGivenClass;
		}
		return probOfPatternGivenClass;
	}

	private static double getProbOfBinGivenClassFrmH(int attribute, int bin, int myClass, List<Histogram> histograms) {
		return histograms.get(attribute).getBins().get(bin).getDistribution().get(myClass);
	}

	private static List<Histogram> trainingPhaseForHistograms(List<List<Double>> examples) {
		List<Histogram> histograms = generateHistograms(examples);
		printHistogramData(histograms);
		return histograms;
	}

	private static void printHistogramData(List<Histogram> histograms) {
		double probOfBinGivenClass;
		for (int myClass = 0; myClass < numOfClasses; myClass++) {
			for (int attribute = 0; attribute < numOfAttributes; attribute++) {
				for (int bin = 0; bin < numOfBins; bin++) {
					probOfBinGivenClass = getProbOfBinGivenClassFrmH(attribute, bin, myClass, histograms);
					System.out.printf("Class %d, attribute %d, bin %d, P(bin | class) = %.2f\n", myClass, attribute,
							bin, probOfBinGivenClass);
				}
			}
		}
	}

	private static List<Histogram> generateHistograms(List<List<Double>> examples) {
		List<Integer> classDistribution = generateClassDistribution(examples);
		generateProbDistributionOfClass(classDistribution, examples.size());
		List<Histogram> histograms = new ArrayList<Histogram>();
		minList = new ArrayList<Double>();
		maxList = new ArrayList<Double>();
		for (int attribute = 0; attribute < numOfAttributes; attribute++) {
			double min = getMinimum(examples, attribute);
			double max = getMaximum(examples, attribute);
			minList.add(min);
			maxList.add(max);
			Histogram histogram = new Histogram();
			List<Bin> bins = histogram.getBins();

			// initialize each bins with value 0
			for (int i = 0; i < numOfBins; i++) {
				Bin bin = new Bin();
				initialize(bin.getDistribution(), numOfClasses, 0);
				bins.add(bin);
			}

			// generating a histogram where distribution of a bin represents the
			// distribution of classes given that bin
			for (int i = 0; i < examples.size(); i++) {
				List<Double> pattern = getPattern(examples.get(i));
				int myClass = getClass(examples.get(i));
				double value = pattern.get(attribute);
				int bin = getBinNumber(value, min, max);
				// just copying the reference of the distribution
				List<Double> classDistributionForBin = bins.get(bin).getDistribution();
				classDistributionForBin.set(myClass, classDistributionForBin.get(myClass) + 1);
			}

			for (int i = 0; i < numOfBins; i++) {
				List<Double> classDistributionForBin = bins.get(i).getDistribution();
				for (int myClass = 0; myClass < numOfClasses; myClass++) {
					double countOfMyClass = classDistributionForBin.get(myClass);
					int totalCountOfMyClass = classDistribution.get(myClass);
					double probOfBinGivenClass;
					if (totalCountOfMyClass != 0.0)
						probOfBinGivenClass = countOfMyClass / totalCountOfMyClass;
					else
						probOfBinGivenClass = 0.0;
					classDistributionForBin.set(myClass, probOfBinGivenClass);
				}
			}
			histogram.setBins(bins);
			histograms.add(histogram);
		}
		return histograms;
	}

	private static void generateProbDistributionOfClass(List<Integer> classDistribution, int totalNumOfExamples) {
		probDistributionOfClass = new ArrayList<Double>();
		for (int i = 0; i < classDistribution.size(); i++)
			probDistributionOfClass.add((double) classDistribution.get(i) / totalNumOfExamples);
	}

	private static double getMinimum(List<List<Double>> examples, int attribute) {
		double temp;
		double min = examples.get(0).get(attribute);
		;
		for (int i = 0; i < examples.size(); i++) {
			temp = examples.get(i).get(attribute);
			if (temp < min) {
				min = temp;
			}
		}
		return min;
	}

	private static double getMaximum(List<List<Double>> examples, int attribute) {
		double temp;
		// initializing max with the 1st value encountered
		double max = examples.get(0).get(attribute);

		for (int i = 0; i < examples.size(); i++) {
			temp = examples.get(i).get(attribute);
			if (temp > max) {
				max = temp;
			}
		}
		return max;
	}

	private static int getBinNumber(double value, double min, double max) {
		int bin;
		double binSize = (max - min) / numOfBins;
		if (value < min)
			bin = 0;
		else if (value >= max) {
			bin = numOfBins - 1;
		} else {
			bin = (int) Math.floor((value - min) / binSize);
		}
		return bin;
	}

	private static void initialize(List<Integer> arrayList, int size, int initializationValue) {
		for (int i = 0; i < size; i++)
			arrayList.add(i, initializationValue);
	}

	private static void initialize(List<Double> arrayList, int size, double initializationValue) {
		for (int i = 0; i < size; i++)
			arrayList.add(i, initializationValue);
	}

	private static List<Integer> generateClassDistribution(List<List<Double>> examples) {
		List<Integer> classDistribution = new ArrayList<Integer>();

		initialize(classDistribution, numOfClasses, 0);

		for (int i = 0; i < examples.size(); i++) {
			int myClass = getClass(examples.get(i));
			classDistribution.set(myClass, classDistribution.get(myClass) + 1);
		}
		return classDistribution;
	}

	private static int getNumberOfClasses(List<List<Double>> examples) {
		int maxClass = -1;
		for (int i = 0; i < examples.size(); i++) {
			if (getClass(examples.get(i)) > maxClass)
				maxClass = getClass(examples.get(i));
		}
		return maxClass + 1;
	}

	private static int getClass(List<Double> example) {
		return example.get(example.size() - 1).intValue();
	}

	private static List<Double> getPattern(List<Double> example) {
		return example.subList(0, example.size() - 1);
	}

	private static List<List<Double>> readFile(String testFile) {
		List<List<Double>> examples = new ArrayList<List<Double>>();
		String fileName = System.getProperty("user.dir") + "/" + testFile;// check
		String line = null;
		String[] values = null;
		List<Double> example = null;

		try {
			FileReader fileReader = new FileReader(fileName);

			BufferedReader bufferedReader = new BufferedReader(fileReader);

			while ((line = bufferedReader.readLine()) != null) {
				example = new ArrayList<Double>();
				line = line.trim();
				values = line.split("\\s+");
				for (int i = 0; i < values.length; i++) {
					example.add(Double.parseDouble(values[i]));
				}
				examples.add(example);
			}
			numOfAttributes = example.size() - 1;
			bufferedReader.close();
		} catch (FileNotFoundException ex) {
			System.out.println("Unable to open file '" + fileName + "'");
			System.exit(1);
		} catch (IOException ex) {
			System.out.println("Error reading file '" + fileName + "'");
			System.exit(1);
		}
		return examples;
	}

	private static void readArguments(String[] args) {
		if (args.length != 3 && args.length != 4) {
			System.out.println("Invalid number of arguments");
			System.exit(1);
		}
		trainingFile = args[0];
		testFile = args[1];
		approximateMethod = args[2];
		if (args.length == 4) {
			if (approximateMethod.equals(HISTOGRAMS))
				numOfBins = Integer.parseInt(args[3]);
			else if (approximateMethod.equals(MIXTURES))
				numOfGaussians = Integer.parseInt(args[3]);
		}
	}
}

