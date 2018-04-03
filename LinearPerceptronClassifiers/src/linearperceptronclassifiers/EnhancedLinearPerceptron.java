/*
 * An implementation of the standard online Linear Peceptron classifer and the
 * offline Perceptron algorithm

 * REMEMBER TO CHECK TERNARY OPERATION AND DELETE COMMENTED CODE
 * Might be able to reduce code a bit by not passing w[] to methods
 * It implements the Weka Classifier interface.
 */
package linearperceptronclassifiers;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * @author Joshua Foster
 */
public class EnhancedLinearPerceptron implements Classifier{
    private double[] w; // Variable for weights
    private final double bias;
    private final static int ETA = 1; // Variable for learning rate
    private final int MAX_ITERATIONS = 100;
    private final boolean STANDARDISE_FLAG;
    private double[] means;
    private double[] stdDev;
    
    /**
     *  Default constructor for the LinearPerceptron classifier
     */
    public EnhancedLinearPerceptron() {
        this.w = new double[]{1, 1};
        this.bias = 0;
        this.STANDARDISE_FLAG = false;
    }
    
    /**
     * A constructor that takes a value for bias
     * @param bias
     */
    public EnhancedLinearPerceptron(double bias) {
        this.w = new double[]{1, 1};
        this.bias = bias;
        this.STANDARDISE_FLAG = false;
    }
    
    /**
     * A constructor that allows the user to specify if instances 
     * should be standardised using standardiseFlag
     * @param standardiseFlag 
     */
    public EnhancedLinearPerceptron(boolean standardiseFlag) {
        this.w = new double[]{1, 1};
        this.bias = 0;
        this.STANDARDISE_FLAG = standardiseFlag;
    }
    
    /**
     * A constructor that takes a value for bias, and allows the user to specify
     * if instances should be standardised using standardiseFlag
     * @param bias 
     * @param standardiseFlag 
     */
    public EnhancedLinearPerceptron(double bias, boolean standardiseFlag) {
        this.w = new double[]{1, 1};
        this.bias = bias;
        this.STANDARDISE_FLAG = standardiseFlag;
    }

    @Override
    public void buildClassifier(Instances train) 
            throws Exception {
        // To do: check that instances are continuous and not discrete
        
        
        // Standardise if flag = true
        if(STANDARDISE_FLAG){
            getMeans(train);
            getStandardDeviation(train);
            train = standardiseAttributes(train);
        }
        // Run the on-line Perceptron training algorithm
        perceptronTraining(train);
        
        // Run the off-line Perceptron training algorithm
        
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        // Standardise attributes if flag = true
        if(STANDARDISE_FLAG){
            for(int i = 0; i < instance.numAttributes(); i++){
                instance.setValue(i, instance.value(i) - means[i] / stdDev[i]);
            }
        }
        // Get the class of the instance
        double y = calculateY(w, instance);
        System.out.println("Pred: " + y + ". Actual: " + instance.value(2));
        return y;
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
    /**
     * Return the y value (either 1 or 0) using a threshold of 0
     * @param w the weights
     * @param i the current instance in the training data
     * @return 1 or -1
     */
    private static int calculateY(double[] w, Instance i){
        double x = i.value(0);
        double y = i.value(1);
        double calc = w[0] * x + w[1] * y;
        return(calc >= 0) ? 1 : -1;
    }
    
    /**
     * Return the y value (either 1 or 0) using a threshold of 0.
     * Overloaded to allow the inclusion of the bias term.
     * @param w the weights
     * @param i the current instance in the training data
     * @param bias the bias variable
     * @return 1 or -1
     */
    private static int calculateY(double[] w, Instance i, double bias){
        double x = i.value(0);
        double y = i.value(1);
        double calc = w[0] * x + w[1] * y + bias;
        return(calc >= 0) ? 1 : -1;
    }
    
    /**
     * Returns the new weights for the iteration
     * @param w the current weights
     * @param i the current instance
     * @param y the output
     * @return an array containing the two updated weights
     */
    private static double[] calculateWeights(double[] w, Instance i, double y){
        double tw[] = new double[2]; // temporary weights
        tw[0] = w[0] + (0.5*ETA) * (i.value(2) - y) * i.value(0);
        tw[1] = w[1] + (0.5*ETA) * (i.value(2) - y) * i.value(1);
        return tw;       
    }
    
    /**
     * An implementation of the on-line perceptron training algorithm
     * @param train
     */
    private void perceptronTraining(Instances train){
        // Initialise count; used to check that a full pass has been made 
        // through the dataset without a change in y or w
        int count = 0;
        int iterations = 0;
        
        do{
            // Loop through each row of the data
            for(int i = 0; i < train.numInstances(); i++){
                // Calculate y
                double y = calculateY(w, train.instance(i));
                // Calculate new weights and store in temporary array
                double[] tw = calculateWeights(w, train.instance(i), y);
                // Check that weights haven't changed
                //if(tempWeights[0] == w[0] && tempWeights[1] == w[1]) count++;
                //else count = 0;
                count = (tw[0]==w[0] && tw[1]==w[1]) ? count+1 : count*0;
                // Assign temporary weights to w[]
                w[0] = tw[0];
                w[1] = tw[1];
                // Increment the number of iterations
                iterations++;
            }
        } while(count < train.numInstances() && iterations <= MAX_ITERATIONS);
    }
    
    
    
    private void gradientDescentTraining(Instances train){
        // Initialise count; qused to check that a full pass has been made 
        // through the dataset without a change in y or w
        int count = 0;
        int iterations = 0;
        
        do{
            // Initialise delta w to zeroes
            double dw[] = new double[]{0,0};
            for(int i = 0; i < train.numInstances(); i++){
                Instance in = train.instance(i);
                // Calculate y
                double y = calculateY(w, in);
                // Calculate deltaW
                for(int j = 0; j < dw.length; j++){
                    dw[j] = dw[j] + (0.5*ETA) *  (in.value(2) - y) * in.value(0);
                }
            }
            // Update weights
            for(int j = 0; j < dw.length; j++){
                w[j] = w[j] + dw[j];
            }
        } while(count < train.numInstances() && iterations <= MAX_ITERATIONS);
    }
    
    
    private void getMeans(Instances train){
        means = new double[train.numAttributes()];
        // Loop through each column (attribute)
        for(int i = 0; i < train.numAttributes(); i++){
            // Get the sum of the attribute for each instance
            for(int j = 0; j < train.numInstances(); j++){
                means[i] += train.get(j).value(i);
            }
            // Divide by the number of instances to get the attribute mean
            means[i] /= train.numInstances();
        }
    }
    
    private void getStandardDeviation(Instances train){
        stdDev = new double[train.numAttributes()];
        // Calculate variance
        // Loop through each column (attribute)
        for(int i = 0; i < train.numAttributes(); i++){
            // Loop through each instance value of attribute i
            for(int j = 0; j < train.numInstances(); j++){
                stdDev[i] += (train.get(j).value(i) - means[i]) * 
                             (train.get(j).value(i) - means[i]);
            }
            // Get the variance for stdDev[i]
            stdDev[i] /= train.numInstances()-1;
        }
        // Calculate standard deviation
        for(int i = 0; i < stdDev.length; i++){
            stdDev[i] = Math.sqrt(stdDev[i]);
        }
    }
    
    private Instances standardiseAttributes(Instances train){
        // Loop through each column (attribute)
        for(int i = 0; i < train.numAttributes(); i++){
            // Loop through each instance value of attribute i
            for(int j = 0; j < train.numInstances(); j++){
                train.get(j).setValue(i, (train.get(j).value(i) - means[i]) / 
                        stdDev[i]);
            }
        }
        return train;
    }
}
