import java.util.ArrayList;
import java.util.List;

public class Bin {
	private List<Double> distribution = null;

	public Bin() {
		this.distribution = new ArrayList<Double>();
	}

	public List<Double> getDistribution() {
		return distribution;
	}

	public void setDistribution(List<Double> distribution) {
		this.distribution = distribution;
	}

	@Override
	public String toString() {
		return "\nBin [distribution=" + distribution + "]";
	}

}
