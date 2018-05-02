/*
 * An implementation of the standard online LinearPeceptron classifer.
 * It implements the Weka Classifier interface.
 */
package linearperceptronclassifiers;

import java.util.Arrays;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Joshua Foster
 */
public class LinearPerceptron extends AbstractClassifier{
    private double[] w; // Variable for weights
    private final double bias;
    private final int ETA = 1; // Variable for learning rate
    private final int MAX_ITERATIONS = 1000;
    
    /**
     *  Default constructor for the LinearPerceptron classifier
     */
    public LinearPerceptron() {
        this.w = new double[]{1, 1};
        this.bias = 0;
    }
    
    /**
     * A constructor that takes a value for bias
     * @param bias 
     */
    public LinearPerceptron(double bias) {
        this.w = new double[]{1, 1};
        this.bias = bias;
    }

    @Override
    public void buildClassifier(Instances train) throws Exception {
        // To do: check that instances are continuous and not discrete
        
        // Initialise weights
        w = new double[train.numAttributes()-1];
        Arrays.fill(w, 1);
        // Initialise count; used to check that a full pass has been made 
        // through the dataset without a change in y or w
        int count = 0;
        int iterations = 0;
        
        do{
            // Loop through each row of the data
            for(int i = 0; i < train.numInstances(); i++){
                // Determine y
                double y = calculateY(train.instance(i));
                // Calculate new weights
                double[] tw = calculateWeights(train.instance(i), y);
                // Check that weights haven't changed
                count = (Arrays.equals(tw, w)) ? count+1 : 0;
                // Assign temporary weights to w[]
                w = tw;
                // End the algorithm if a full pass has been made 
                if(count == train.numInstances()) {
                    break;
                }
                // Increment the number of iterations
                iterations++;
            }
        } while(count < train.numInstances() && iterations <= MAX_ITERATIONS);
    }

    @Override
    public double classifyInstance(Instance i) throws Exception {
        // Get the class of the instance
        return calculateY(i);
    }

//    @Override
//    public double[] distributionForInstance(Instance instnc) throws Exception {
//        throw new UnsupportedOperationException("Not supported yet.");
//    }
//
//    @Override
//    public Capabilities getCapabilities() {
//        throw new UnsupportedOperationException("Not supported yet.");
//    }
    
    /**
     * Return the y value (either 1 or 0) using a threshold of 0
     * @param w the weights
     * @param i the current instance in the training data
     * @return 1 or -1
     */
    private int calculateY(Instance i){
        // Calculate y
        double calc = 0;
        for (int j = 0; j < i.numAttributes()-1; j++) {
            calc += w[j] * i.value(j);
        }
        calc += bias;
        return(calc >= 0) ? 1 : 0;
        
    }
    
    /**
     * Returns the new weights for the iteration
     * @param i the current instance
     * @param y the output
     * @return an array containing the two updated weights
     */
    private double[] calculateWeights(Instance i, double y){
        double tw[] = new double[w.length];
        for(int j = 0; j < w.length; j++){
            tw[j] = w[j] + (0.5*ETA) * (i.classValue() - y) * i.value(j);
        }
        return tw;       
    }
    
}
