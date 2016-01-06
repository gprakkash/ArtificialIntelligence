import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Random;

public class DecisionTree {

	private static String trainingFile;
	private static String testFile;
	private static String option;
	private static int numOfAttributes = 0;
	private static int numOfClasses = 0;
	private static int examplesToBePruned = 50;
	private static int numOfThreshold = 50;

	public static void main(String[] args) {

		readArguments(args);
		List<Node> decisionTrees = null;
		decisionTrees = trainingPhase();
		classificationPhase(decisionTrees);

	}

	private static List<Node> trainingPhase() {
		List<Node> decisionTrees = new ArrayList<Node>();
		Map<List<Double>, Integer> trainingExamples = new LinkedHashMap<List<Double>, Integer>();
		readFile(trainingExamples, trainingFile);
		/*
		 * System.out.println("No. of training objects: " +
		 * trainingExamples.size());
		 */
		numOfClasses = findNumberOfClasses(trainingExamples);
		List<Double> myDefault = generateDistribution(trainingExamples);

		if (option.equals("optimized") || option.equals("randomized")) {
			Node decisionTree = learnDecisionTree(trainingExamples, numOfAttributes, myDefault, option);
			printDecisionTree(decisionTree, 0);
			decisionTrees.add(decisionTree);
		} else if (option.equals("forest3")) {
			for (int i = 0; i < 3; i++) {
				Node decisionTree = learnDecisionTree(trainingExamples, numOfAttributes, myDefault, "randomized");
				decisionTrees.add(decisionTree);
				printDecisionTree(decisionTree, i);
			}
		} else if (option.equals("forest15")) {
			for (int i = 0; i < 15; i++) {
				Node decisionTree = learnDecisionTree(trainingExamples, numOfAttributes, myDefault, "randomized");
				decisionTrees.add(decisionTree);
				printDecisionTree(decisionTree, i);
			}
		} else {
			System.out.println("Invalid value of option parameter");
			System.exit(1);
		}
		return decisionTrees;
	}

	private static void classificationPhase(List<Node> decisionTrees) {
		Map<List<Double>, Integer> testExamples = new LinkedHashMap<List<Double>, Integer>();
		readFile(testExamples, testFile);
		/* System.out.println("No. of test objects: " + testExamples.size()); */
		// numOfClasses = findNumberOfClasses(testExamples);
		classify(testExamples, decisionTrees);

	}

	private static void classify(Map<List<Double>, Integer> testExamples, List<Node> decisionTrees) {
		List<Double> distribution = null;
		int exampleID = 0;
		int predictedClass = -1;
		int trueClass = -1;
		double accuracy = 0.0;
		double accuracySum = 0.0;

		for (Map.Entry<List<Double>, Integer> pair : testExamples.entrySet()) {
			distribution = getAverageClassDistribution(pair.getKey(), decisionTrees);
			predictedClass = predictClass(distribution);
			trueClass = pair.getValue();
			accuracy = getAccuracy(predictedClass, trueClass, distribution);
			accuracySum = accuracySum + accuracy;
			System.out.printf("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n", exampleID, predictedClass, trueClass,
					accuracy);
			exampleID++;
		}
		System.out.printf("classification accuracy=%6.4f\n", accuracySum / (double) testExamples.size());
	}

	private static List<Double> getAverageClassDistribution(List<Double> testPattern, List<Node> decisionTrees) {
		List<Double> distribution = null;
		for (int i = 0; i < decisionTrees.size(); i++) {
			List<Double> tempDistribution = getClasssDistribution(testPattern, decisionTrees.get(i));
			if (i == 0)
				distribution = new ArrayList<Double>(tempDistribution);
			else {
				for (int j = 0; j < distribution.size(); j++)
					distribution.set(j, distribution.get(j) + tempDistribution.get(j));
			}
		}
		return distribution;
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

	private static List<Double> getClasssDistribution(List<Double> pattern, Node decisionTree) {
		// base case
		if (decisionTree.getAttribute() == -1) {
			return decisionTree.getClassDistribution();
		}
		// recursive case
		if (pattern.get(decisionTree.getAttribute()) < decisionTree.getThreshold())
			return getClasssDistribution(pattern, decisionTree.getLeftNode());
		else
			return getClasssDistribution(pattern, decisionTree.getRightNode());
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

	private static void printDecisionTree(Node root, int treeID) {
		int nodeID = 0;
		Queue<Node> q = new LinkedList<Node>();
		if (root == null)
			return;
		q.add(root);
		while (!q.isEmpty()) {
			Node n = (Node) q.remove();
			++nodeID;
			System.out.printf("tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n", treeID, nodeID, n.getAttribute(),
					n.getThreshold(), n.getInformationGain());
			if (n.getLeftNode() != null)
				q.add(n.getLeftNode());
			if (n.getRightNode() != null)
				q.add(n.getRightNode());
		}
		System.out.println();
	}

	private static Node learnDecisionTree(Map<List<Double>, Integer> examples, int numOfAttributes,
			List<Double> myDefault, String option) {
		if (examples.isEmpty()) {
			Node node = new Node();
			node.setClassDistribution(myDefault);
			return node;
		} else if (allExamplesHaveSameClass(examples)) {
			Node node = new Node();
			node.setClassDistribution(generateDistribution(examples));
			return node;
		} else {
			List<Double> result = null;

			if (option.equals("optimized"))
				result = chooseBestAttributeAndThreshold(examples, numOfAttributes);
			else if (option.equals("randomized"))
				result = chooseRandomAttributeAndBestThreshold(examples, numOfAttributes);

			int attribute = result.get(0).intValue();
			double bestThreshold = result.get(1);

			Node node = new Node();
			Map<List<Double>, Integer> leftExamples = getLeftExamples(examples, attribute, bestThreshold);
			Map<List<Double>, Integer> rightExamples = getRightExamples(examples, attribute, bestThreshold);

			if (leftExamples.size() < examplesToBePruned || rightExamples.size() < examplesToBePruned) {
				node.setClassDistribution(generateDistribution(examples));
				return node;
			} else {
				node.setAttribute(attribute);
				node.setThreshold(bestThreshold);
				node.setLeftNode(
						learnDecisionTree(leftExamples, numOfAttributes, generateDistribution(examples), option));
				node.setRightNode(
						learnDecisionTree(rightExamples, numOfAttributes, generateDistribution(examples), option));
				node.setInformationGain(informationGain(examples, attribute, bestThreshold));
				return node;
			}

		}
	}

	private static List<Double> generateDistribution(Map<List<Double>, Integer> examples) {
		List<Double> distribution = new ArrayList<Double>();
		Map<Integer, Integer> classes = getClassesAndItsCount(examples);
		int numOfExamples = examples.size();
		for (int i = 0; i < numOfClasses; i++) {
			if (classes.get(i) != null)
				distribution.add((double) classes.get(i) / (double) numOfExamples);
			else
				distribution.add((double) 0);
		}
		return distribution;
	}

	private static int findNumberOfClasses(Map<List<Double>, Integer> examples) {
		int maxClass = -1;
		for (Map.Entry<List<Double>, Integer> pair : examples.entrySet()) {
			if (pair.getValue() > maxClass)
				maxClass = pair.getValue();
		}
		return maxClass + 1;
	}

	private static boolean allExamplesHaveSameClass(Map<List<Double>, Integer> examples) {
		Map<Integer, Integer> classes = getClassesAndItsCount(examples);
		if (classes.size() == 1)
			return true;
		else
			return false;
	}

	private static Map<Integer, Integer> getClassesAndItsCount(Map<List<Double>, Integer> examples) {
		Map<Integer, Integer> classes = new HashMap<Integer, Integer>();
		int myClass = -1;
		int count = 0;
		// get the count of each classes from the examples
		for (Map.Entry<List<Double>, Integer> pair : examples.entrySet()) {
			myClass = (Integer) pair.getValue();
			if (classes.containsKey(myClass)) {
				count = classes.get(myClass);
				classes.put(myClass, ++count);
			} else {
				classes.put(myClass, 1);
			}
		}
		return classes;
	}

	private static List<Double> chooseBestAttributeAndThreshold(Map<List<Double>, Integer> examples,
			int numOfAttributes) {
		double maxGain = -1;
		double bestAttribute = -1;
		double bestThreshold = -1;
		double l = 0;
		double m = 0;
		double value = 0;
		double threshold = 0;
		double gain = 0;
		boolean firstIteration = true;
		List<Double> result = new ArrayList<Double>();

		for (int i = 0; i < numOfAttributes; i++) {
			for (Map.Entry<List<Double>, Integer> pair : examples.entrySet()) {
				value = pair.getKey().get(i);
				if (firstIteration) {
					l = value;
					m = value;
					firstIteration = false;
				} else {
					if (l > value)
						l = value;
					if (m < value)
						m = value;
				}
			}
			firstIteration = true;

			for (int k = 1; k <= numOfThreshold; k++) {
				threshold = l + k * (m - l) / (numOfThreshold + 1);
				gain = informationGain(examples, i, threshold);
				if (gain > maxGain) {
					maxGain = gain;
					bestAttribute = i;
					bestThreshold = threshold;
				}
			}
		}
		result.add(bestAttribute);
		result.add(bestThreshold);
		return result;
	}

	private static List<Double> chooseRandomAttributeAndBestThreshold(Map<List<Double>, Integer> examples,
			int numOfAttributes) {
		double maxGain = -1;
		int randomAttribute = -1;
		double bestThreshold = -1;
		double l = 0;
		double m = 0;
		double value = 0;
		double threshold = 0;
		double gain = 0;
		boolean firstIteration = true;
		Random rand = new Random();
		List<Double> result = new ArrayList<Double>();

		randomAttribute = rand.nextInt(numOfAttributes);
		for (Map.Entry<List<Double>, Integer> pair : examples.entrySet()) {
			value = pair.getKey().get(randomAttribute);
			if (firstIteration) {
				l = value;
				m = value;
				firstIteration = false;
			} else {
				if (l > value)
					l = value;
				if (m < value)
					m = value;
			}
		}

		for (int k = 1; k <= numOfThreshold; k++) {
			threshold = l + k * (m - l) / (numOfThreshold + 1);
			gain = informationGain(examples, randomAttribute, threshold);
			if (gain > maxGain) {
				maxGain = gain;
				bestThreshold = threshold;
			}
		}
		result.add((double) randomAttribute);
		result.add(bestThreshold);
		return result;
	}

	private static double informationGain(Map<List<Double>, Integer> examples, int attribute, double threshold) {

		Map<List<Double>, Integer> leftExamples = getLeftExamples(examples, attribute, threshold);
		Map<List<Double>, Integer> rightExamples = getRightExamples(examples, attribute, threshold);
		double size = examples.size();
		double sizeOfLeft = leftExamples.size();
		double sizeOfRight = rightExamples.size();
		double informationGain = entropy(examples) - (sizeOfLeft / size) * entropy(leftExamples)
				- (sizeOfRight / size) * entropy(rightExamples);
		return informationGain;
	}

	private static Map<List<Double>, Integer> getLeftExamples(Map<List<Double>, Integer> examples, int attribute,
			double threshold) {
		Map<List<Double>, Integer> subExamples = new LinkedHashMap<List<Double>, Integer>();

		for (Map.Entry<List<Double>, Integer> pair : examples.entrySet()) {
			if (pair.getKey().get(attribute) < threshold)
				subExamples.put(pair.getKey(), pair.getValue());
		}
		return subExamples;
	}

	private static Map<List<Double>, Integer> getRightExamples(Map<List<Double>, Integer> examples, int attribute,
			double threshold) {
		Map<List<Double>, Integer> subExamples = new LinkedHashMap<List<Double>, Integer>();

		for (Map.Entry<List<Double>, Integer> pair : examples.entrySet()) {
			if (pair.getKey().get(attribute) >= threshold)
				subExamples.put(pair.getKey(), pair.getValue());
		}
		return subExamples;
	}

	private static double entropy(Map<List<Double>, Integer> examples) {
		int positiveExamples = 0;
		double calculatedEntropy = 0;
		double prob = 0;
		double totalNoOfExamples = examples.size();
		Map<Integer, Integer> classes = getClassesAndItsCount(examples);

		for (Map.Entry<Integer, Integer> pair : classes.entrySet()) {
			positiveExamples = (Integer) pair.getValue();
			prob = positiveExamples / totalNoOfExamples;
			calculatedEntropy += (-prob) * log2(prob);
		}
		return calculatedEntropy;
	}

	private static double log2(double prob) {
		return Math.log(prob) / Math.log(2);
	}

	private static void readFile(Map<List<Double>, Integer> examples, String testFile) {

		String fileName = System.getProperty("user.dir") + "/" + testFile;// check
		String line = null;
		String[] values = null;
		List<Double> pattern = null;
		int myClass = -1;

		try {
			FileReader fileReader = new FileReader(fileName);

			BufferedReader bufferedReader = new BufferedReader(fileReader);

			while ((line = bufferedReader.readLine()) != null) {
				pattern = new ArrayList<Double>();
				line = line.trim();
				values = line.split("\\s+");
				for (int i = 0; i < values.length; i++) {
					if (i != values.length - 1)
						pattern.add(Double.parseDouble(values[i]));
					else if (i == values.length - 1)
						myClass = Integer.parseInt(values[i]);
				}
				examples.put(pattern, myClass);
			}
			numOfAttributes = pattern.size();
			bufferedReader.close();
		} catch (FileNotFoundException ex) {
			System.out.println("Unable to open file '" + fileName + "'");
			System.exit(1);
		} catch (IOException ex) {
			System.out.println("Error reading file '" + fileName + "'");
			System.exit(1);
		}
	}

	private static void readArguments(String[] args) {
		trainingFile = args[0];
		testFile = args[1];
		option = args[2];
	}

}
