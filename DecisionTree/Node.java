import java.util.List;

public class Node {

	private int attribute = -1;// -1 for leaf node
	private double threshold = -1;
	private double informationGain = -1;
	private List<Double> classDistribution = null;
	private Node leftNode = null;
	private Node rightNode = null;

	public int getAttribute() {
		return attribute;
	}

	public void setAttribute(int attribute) {
		this.attribute = attribute;
	}

	public double getThreshold() {
		return threshold;

	}

	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	public double getInformationGain() {
		return informationGain;
	}

	public void setInformationGain(double informationGain) {
		this.informationGain = informationGain;
	}

	public List<Double> getClassDistribution() {
		return classDistribution;
	}

	public void setClassDistribution(List<Double> classDistribution) {
		this.classDistribution = classDistribution;
	}

	public Node getLeftNode() {
		return leftNode;
	}

	public void setLeftNode(Node leftNode) {
		this.leftNode = leftNode;
	}

	public Node getRightNode() {
		return rightNode;
	}

	public void setRightNode(Node rightNode) {
		this.rightNode = rightNode;
	}

	@Override
	public String toString() {
		return "Node \n[attribute=" + attribute + ", \nthreshold=" + threshold + ", \ninformationGain=" + informationGain
				+ ", \nclassDistribution=" + classDistribution + ", \nleftNode=" + leftNode + ", \nrightNode=" + rightNode
				+ "]";
	}
}
