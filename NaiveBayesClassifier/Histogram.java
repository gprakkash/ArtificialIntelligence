import java.util.ArrayList;
import java.util.List;

public class Histogram {
	private List<Bin> bins = null;

	public Histogram() {
		this.bins = new ArrayList<Bin>();
	}

	public List<Bin> getBins() {
		return bins;
	}

	public void setBins(List<Bin> bins) {
		this.bins = bins;
	}

	@Override
	public String toString() {
		return "Histogram [bins=" + bins + "]";
	}

}
