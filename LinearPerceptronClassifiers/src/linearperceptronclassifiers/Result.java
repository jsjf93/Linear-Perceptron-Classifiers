/*
 * A class that represents the results and statistics for a models performance
 */
package linearperceptronclassifiers;

/**
 *
 * @author Joshua Foster
 */
public class Result {
    private final String classifierName;
    private final String datasetName;
    private final int numInstances;
    private final int numAttributes;
    private final double accuracy;
    
    
    /**
     * Constructor for a result object
     * @param datasetName
     * @param numInstances
     * @param numAttributes
     * @param classifierName
     * @param accuracy
     */
    public Result(String classifierName, String datasetName, int numInstances, 
                  int numAttributes, double accuracy){
        this.classifierName = classifierName;
        this.datasetName = datasetName;
        this.numInstances = numInstances;
        this.numAttributes = numAttributes;
        this.accuracy = accuracy;
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
    
    public double getAccuracy() {
        return accuracy;
    }
    
    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        s.append(classifierName).append(", ");
        s.append(datasetName).append(", ");
        s.append(numInstances).append(", ");
        s.append(numAttributes).append(", ");
        s.append(accuracy).append("\n");
        return s.toString();
    }
}
