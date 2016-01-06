import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class KNearestNeighborClassifier {
	private static String trainingFile;
	private static String testFile;
	// should be the same for trainingFile and testFile
	private static int numOfAttributes = -1;
	private static int numOfClasses = -1;
	private static int argK = -1;

	public static void main(String[] args) {
		readArguments(args);
		List<List<Double>> trainingExamples = readFile(trainingFile);
		List<List<Double>> testExamples = readFile(testFile);
		numOfClasses = getNumberOfClasses(trainingExamples);
		// testing();
		List<List<Double>> normalizedTrainingExamples = trainingPhase(trainingExamples);

		classificationPhase(normalizedTrainingExamples, testExamples);

		System.out.println("Exiting Main");
	}

	private static void classificationPhase(List<List<Double>> normalizedTrainingExamples,
			List<List<Double>> testExamples) {
		List<List<Double>> normalizedTestExamples = normaliztion(testExamples);
		double accuracy;
		double accuracySum = 0.0;
		for (int exampleId = 0; exampleId < normalizedTestExamples.size(); exampleId++) {
			List<Double> example = normalizedTestExamples.get(exampleId);
			List<Double> pattern = getPattern(example);
			int trueClass = getClass(example);
			// List<Integer> indexOfKNearestNeighbors = new
			// ArrayList<Integer>();
			List<Double> nnDetails = new ArrayList<Double>();
			List<List<Double>> kNearestNeighbors = getKNearestNeighbors(pattern, normalizedTrainingExamples, nnDetails);
			List<Double> probDistributionOfClassGivenPattern = getProbDistributionOfClassesGivenPattern(
					kNearestNeighbors);
			int predictedClass = predictClass(probDistributionOfClassGivenPattern);
			accuracy = getAccuracy(predictedClass, trueClass, probDistributionOfClassGivenPattern);
			accuracySum = accuracySum + accuracy;
			System.out.printf("ID=%5d, predicted=%3d, true=%3d, nn=%5d, distance=%7.2f, accuracy=%4.2f\n", exampleId,
					predictedClass, trueClass, nnDetails.get(0).intValue(), nnDetails.get(1), accuracy);
		}
		System.out.printf("classification accuracy=%6.4f\n", accuracySum / normalizedTestExamples.size());

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

	private static List<Double> getProbDistributionOfClassesGivenPattern(List<List<Double>> kNearestNeighbors) {
		List<Double> probDistributionOfClassesGivenPattern = new ArrayList<Double>();
		for (int myClass = 0; myClass < numOfClasses; myClass++) {
			probDistributionOfClassesGivenPattern
					.add((double) getCountOfNeighborsOfClass(myClass, kNearestNeighbors) / argK);
		}
		return probDistributionOfClassesGivenPattern;
	}

	private static int getCountOfNeighborsOfClass(int myClass, List<List<Double>> kNearestNeighbors) {
		int count = 0;
		for (int i = 0; i < kNearestNeighbors.size(); i++) {
			if (myClass == getClass(kNearestNeighbors.get(i)))
				count++;
		}
		return count;
	}

	private static List<List<Double>> getKNearestNeighbors(List<Double> pattern, List<List<Double>> examples,
			List<Double> nnDetails) {
		List<List<Double>> kNearestNeighbors = new ArrayList<List<Double>>();
		List<Double> distances = new ArrayList<Double>();
		double distance;
		for (int i = 0; i < examples.size(); i++) {
			List<Double> neighbor = examples.get(i);
			distance = getDistance(pattern, getPattern(neighbor));
			if (checkIfAmongKNearestNeighbor(distance, distances)) {
				int pos = getPostitionAmongKNearest(distance, distances);
				addToKNearestNeighbor(neighbor, pos, kNearestNeighbors);
				if (pos == 0) {
					nnDetails.add(0, (double) i);
					nnDetails.add(1, distance);
				}
			}
		}
		return kNearestNeighbors;
	}

	private static void addToKNearestNeighbor(List<Double> neighbor, int pos, List<List<Double>> kNearestNeighbors) {
		kNearestNeighbors.add(pos, neighbor);
		if (kNearestNeighbors.size() == argK + 1) {
			kNearestNeighbors.remove(argK);
		}
	}

	private static int getPostitionAmongKNearest(double distance, List<Double> distances) {
		int pos = -1;
		for (int i = 0; i < distances.size(); i++) {
			if (distance < distances.get(i)) {
				distances.add(i, distance);
				pos = i;
				break;
			}
		}
		if (pos == -1 && distances.size() < argK) {
			distances.add(distance);
			pos = distances.size() - 1;
		}
		if (distances.size() == argK + 1)
			distances.remove(argK);
		return pos;
	}

	private static boolean checkIfAmongKNearestNeighbor(double distance, List<Double> distances) {
		if (distances.size() < argK)
			return true;
		for (int i = 0; i < argK; i++) {
			if (distance < distances.get(i))
				return true;
		}
		return false;
	}

	private static double getDistance(List<Double> pattern1, List<Double> pattern2) {
		double distance = 0;
		for (int i = 0; i < numOfAttributes; i++) {
			distance = distance + Math.pow((pattern1.get(i) - pattern2.get(i)), 2);
		}
		distance = Math.sqrt(distance);
		return distance;
	}

	private static List<List<Double>> trainingPhase(List<List<Double>> trainingExamples) {
		List<List<Double>> normalizedTrainingExamples;
		normalizedTrainingExamples = normaliztion(trainingExamples);
		return normalizedTrainingExamples;
	}

	private static List<List<Double>> normaliztion(List<List<Double>> examples) {
		// the List<Double> contains the list of values of one attribute and
		// will be used only for the purpose of calculating mean and std
		List<List<Double>> attributedExamples = getAttributedExamples(examples);
		List<Double> means = new ArrayList<Double>();
		List<Double> stds = new ArrayList<Double>();
		List<List<Double>> normalizedExamples = new ArrayList<List<Double>>();

		for (int attribute = 0; attribute < numOfAttributes; attribute++) {
			double mean = getMean(attributedExamples.get(attribute));
			double std = getStd(mean, attributedExamples.get(attribute));
			means.add(mean);
			stds.add(std);
		}

		for (int i = 0; i < examples.size(); i++) {
			List<Double> normalizedExample = new ArrayList<Double>();
			List<Double> pattern = getPattern(examples.get(i));
			int myClass = getClass(examples.get(i));

			for (int attribute = 0; attribute < numOfAttributes; attribute++) {
				double value = pattern.get(attribute);
				normalizedExample.add(getNormalizedValue(value, means.get(attribute), stds.get(attribute)));
			}
			normalizedExample.add((double) myClass);
			normalizedExamples.add(normalizedExample);
		}
		return normalizedExamples;
	}

	private static List<List<Double>> getAttributedExamples(List<List<Double>> examples) {
		List<List<Double>> attributedExamples = new ArrayList<List<Double>>();
		for (int attribute = 0; attribute < numOfAttributes; attribute++) {
			List<Double> list = new ArrayList<Double>();
			attributedExamples.add(list);
		}

		for (int i = 0; i < examples.size(); i++) {
			List<Double> pattern = getPattern(examples.get(i));
			for (int attribute = 0; attribute < numOfAttributes; attribute++) {
				attributedExamples.get(attribute).add(pattern.get(attribute));
			}
		}
		return attributedExamples;
	}

	private static double getNormalizedValue(double value, double mean, double std) {
		if (std == 0)
			return 0;
		return (value - mean) / std;
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
		String fileName = System.getProperty("user.dir") + "/" + testFile;
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
		if (args.length != 3) {
			System.out.println("Invalid number of arguments");
			System.exit(1);
		}
		trainingFile = args[0];
		testFile = args[1];
		argK = Integer.parseInt(args[2]);
	}
}
