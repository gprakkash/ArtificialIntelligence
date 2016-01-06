import java.util.List;

public class MixtureOfGaussians {
	List<Gaussian> mixture;
	List<Double> weights;
	
	//indicates the probability that the ith Gaussian generated the jth value
	List<List<Double>> probij;

	public List<List<Double>> getProbij() {
		return probij;
	}

	public void setProbij(List<List<Double>> probij) {
		this.probij = probij;
	}

	public List<Double> getWeights() {
		return weights;
	}

	public void setWeights(List<Double> weights) {
		this.weights = weights;
	}

	public List<Gaussian> getMixture() {
		return mixture;
	}

	public void setMixture(List<Gaussian> mixture) {
		this.mixture = mixture;
	}

	@Override
	public String toString() {
		return "MixtureOfGaussians \n[mixture=\n" + mixture + "\nweights=" + weights + "]\n";
	}

}
