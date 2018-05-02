/*
 * A class that represents the results and statistics for a models performance
 */
package linearperceptronclassifiers;

/**
 *
 * @author imac5
 */
public class Result {
    private final String classifierName;
    private final String datasetName;
    private final int numInstances;
    private final int numAttributes;
    private final double averageAccuracy;
    private final double accuracy;
    private final double balancedAccuracy;
    private final double recall;
    private final double precision;
    private final double fMeasure;
    private final long timing;
    
    
    /**
     * Constructor for a result object
     * @param datasetName
     * @param numInstances
     * @param numAttributes
     * @param classifierName
     * @param averageAccuracy
     * @param accuracy
     * @param balancedAccuracy
     * @param recall
     * @param precision
     * @param timing 
     * @param fMeasure 
     */
    public Result(String classifierName, String datasetName, int numInstances, 
                  int numAttributes, double averageAccuracy,long timing){
        this.classifierName = classifierName;
        this.datasetName = datasetName;
        this.numInstances = numInstances;
        this.numAttributes = numAttributes;
        this.averageAccuracy = averageAccuracy;
        this.accuracy = 0;
        this.balancedAccuracy = 0;
        this.recall = 0;
        this.precision = 0;
        this.fMeasure = 0;
        this.timing = timing;
    }
    
    /**
     * Constructor for a result object
     * @param datasetName
     * @param numInstances
     * @param numAttributes
     * @param classifierName
     * @param averageAccuracy
     * @param accuracy
     * @param balancedAccuracy
     * @param recall
     * @param precision
     * @param timing 
     * @param fMeasure 
     */
    public Result(String classifierName, String datasetName, int numInstances, 
                  int numAttributes, double averageAccuracy, double accuracy,
                  double balancedAccuracy, double recall, double precision,
                  double fMeasure, long timing){
        this.classifierName = classifierName;
        this.datasetName = datasetName;
        this.numInstances = numInstances;
        this.numAttributes = numAttributes;
        this.averageAccuracy = averageAccuracy;
        this.accuracy = accuracy;
        this.balancedAccuracy = balancedAccuracy;
        this.recall = recall;
        this.precision = precision;
        this.fMeasure = fMeasure;
        this.timing = timing;
    }

    

    public String getClassifierName() {
        return classifierName;
    }
    
    public String getDatasetName() {
        return datasetName;
    }

    public int getNumInstances() {
        return numInstances;
    }

    public int getNumAttributes() {
        return numAttributes;
    }

    public double getAverageAccuracy() {
        return averageAccuracy;
    }
    
    public double getAccuracy() {
        return accuracy;
    }
    
    public double getBalancedAccuracy() {
        return balancedAccuracy;
    }

    public double getRecall() {
        return recall;
    }

    public double getPrecision() {
        return precision;
    }

    public double getfMeasure() {
        return fMeasure;
    }

    public long getTiming() {
        return timing;
    }
    
    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        s.append(classifierName).append(", ");
        s.append(datasetName).append(", ");
        s.append(numInstances).append(", ");
        s.append(numAttributes).append(", ");
        s.append(accuracy).append(", ");
        s.append(timing).append("\n");
        return s.toString();
    }
}
