/*
 * An implementation of the standard online Linear Peceptron classifer and the
 * offline Perceptron algorithm

 * REMEMBER TO CHECK TERNARY OPERATION AND DELETE COMMENTED CODE
 * Might be able to reduce code a bit by not passing w[] to methods
 * It implements the Weka Classifier interface.
 */
package linearperceptronclassifiers;

import weka.classifiers.Classifier;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Joshua Foster
 */
public class EnhancedLinearPerceptron implements Classifier{
    private double[] w; // Variable for weights
    private double bias;
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
        this.w = new double[]{-0.5999067136984317, 1.929693104359004};
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
    public void buildClassifier(Instances train) throws Exception {
        // To do: check that instances are continuous and not discrete
        
        
        // Standardise if flag = true
        if(STANDARDISE_FLAG){
            train = standardise(train);
        }

        //System.out.println(train);
        //System.out.println("");
        
        // Run the on-line Perceptron training algorithm
        perceptronTraining(train);
        
        // Run the off-line Perceptron training algorithm
        
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double y;
        // Standardise attributes if flag = true
        if(STANDARDISE_FLAG){
            //for(int i = 0; i < instance.numAttributes()-1; i++){
            //    instance.setValue(i, (instance.value(i) - means[i]) / stdDev[i]);
           // }
            double x1 = (instance.value(0) - means[0]) / stdDev[0];
            double x2 = (instance.value(1) - means[1]) / stdDev[1];
            double calc = w[0] * x1 + w[1] * x2 + bias;
            y = (calc >= 0) ? 1 : -1;
            //System.out.println(instance);
        }
        else {
            y = calculateY(instance);
        }
        
        
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
     * @param i the current instance in the training data
     * @return 1 or -1
     */
    private int calculateY(Instance i){
        double x1 = i.value(0);
        double x2 = i.value(1);
        double calc = w[0] * x1 + w[1] * x2 + bias;
        //System.out.println(calc);
        return(calc >= 0) ? 1 : -1;
    }
    
    
    /**
     * Returns the new weights for the iteration
     * @param i the current instance
     * @param y the output
     * @return an array containing the two updated weights
     */
    private double[] calculateWeights(Instance i, double y){
        double tw[] = new double[2]; // temporary weights
        System.out.println("Calculation: ");
        System.out.println("tw[0] = " + w[0] + " + 0.5 * (" + i.value(2) + " - " + y + ") * " + i.value(0));
        tw[0] = w[0] + (0.5*ETA) * (i.value(2) - y) * i.value(0);
        tw[1] = w[1] + (0.5*ETA) * (i.value(2) - y) * i.value(1);
        //bias = (0.5*ETA) * (i.value(2) - y);
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
                double y = calculateY(train.instance(i));
                //System.out.println(y);
                // Calculate new weights and store in temporary array
                double[] tw = calculateWeights(train.instance(i), y);
                //System.out.println("tw: " + tw[0] + ", " + tw[1]);
                // Check that weights haven't changed
                count = (tw[0] == w[0] && tw[1] == w[1]) ? count+1 : count*0;
                // Assign temporary weights to w[]
                w[0] = tw[0];
                w[1] = tw[1];
                // Increment the number of iterations
                iterations++;
                //System.out.println("Count: " + count);
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
                double y = calculateY(in);
                // Calculate deltaW
                for(int j = 0; j < dw.length; j++){
                    dw[j] = dw[j] + (0.5*ETA) * (in.value(2) - y) * in.value(0);
                }
            }
            // Update weights
            for(int j = 0; j < dw.length; j++){
                w[j] = w[j] + dw[j];
            }
        } while(count < train.numInstances() && iterations <= MAX_ITERATIONS);
    }
    
    /**
     * Obtains the mean of each attribute column of the Instances provided
     * @param train 
     */
    private void getMeans(Instances train){
        means = new double[train.numAttributes()-1];
        // Loop through each column (attribute)
        for(int i = 0; i < train.numAttributes()-1; i++){
            // Get the sum of the attribute for each instance
            for(int j = 0; j < train.numInstances(); j++){
                means[i] += train.get(j).value(i);
            }
            // Divide by the number of instances to get the attribute mean
            means[i] /= train.numInstances();
        }
    }
    
    /**
     * Obtains the standard deviation of each attribute column of the Instances
     * provided
     * @param train 
     */
    private void getStandardDeviation(Instances train){
        stdDev = new double[train.numAttributes()-1];
        // Calculate variance
        // Loop through each column (attribute)
        for(int i = 0; i < train.numAttributes()-1; i++){
            // Loop through each instance value of attribute i
            for(int j = 0; j < train.numInstances(); j++){
                stdDev[i] += (train.get(j).value(i) - means[i]) * 
                             (train.get(j).value(i) - means[i]);
            }
            // Get the variance for stdDev[i]
            stdDev[i] /= train.numInstances();
        }
        // Calculate standard deviation
        for(int i = 0; i < stdDev.length; i++){
            stdDev[i] = Math.sqrt(stdDev[i]);
        }
    }
    
    /**
     * Standardises each attribute value using the means and standard deviations
     * calculated in the getMeans() and getStandardDeviation() methods
     * @param train
     * @return the instances with standardised attributes
     */
    private Instances standardiseAttributes(Instances train){
        // Loop through each column (attribute)
        for(int i = 0; i < train.numAttributes()-1; i++){
            // Loop through each instance value of attribute i
            for(int j = 0; j < train.numInstances(); j++){
                train.get(j).setValue(i, (train.get(j).value(i) - means[i]) / 
                        stdDev[i]);
            }
        }
        return train;
    }
    
    private Instances standardise(Instances train){
        means = new double[train.numAttributes()-1];
        stdDev = new double[train.numAttributes()-1];
        
        for(int i = 0; i < train.numAttributes()-1; i++){
            AttributeStats attributeStats = train.attributeStats(i);
            means[i] = attributeStats.numericStats.mean;
            stdDev[i] = attributeStats.numericStats.stdDev;
            for(int j = 0; j < train.numInstances(); j++){
                train.get(j).setValue(i, (train.get(j).value(i) - means[i]) / 
                        stdDev[i]);
            }
        }
        return train;
    }
}
