/*
 * A class that represents the results and statistics for a models performance
 */
package linearperceptronclassifiers;

/**
 *
 * @author imac5
 */
public class Result {
    private final String data;
    private final String classifierName;
    private final int accuracy;
    private final double timing;
    
    /**
     * Constructor for a result object
     * @param data
     * @param classifierName
     * @param accuracy
     * @param timing 
     */
    public Result(String data, String classifierName, int accuracy, double timing){
        this.data = data;
        this.classifierName = classifierName;
        this.accuracy = accuracy;
        this.timing = timing;
    }

    public String getData() {
        return data;
    }

    public String getClassifierName() {
        return classifierName;
    }

    public int getAccuracy() {
        return accuracy;
    }

    public double getTiming() {
        return timing;
    }
}
