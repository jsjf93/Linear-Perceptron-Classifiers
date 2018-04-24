/*
 * An implementation of the standard online Linear Peceptron classifer and the
 * offline Perceptron algorithm

 * REMEMBER TO CHECK TERNARY OPERATION AND DELETE COMMENTED CODE
 * Might be able to reduce code a bit by not passing w[] to methods
 * It implements the Weka Classifier interface.
 */
package linearperceptronclassifiers;

import java.util.Arrays;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Joshua Foster
 */
public class EnhancedLinearPerceptron extends AbstractClassifier{
    private double[] w; // Variable for weights
    private double bias;
    private final int ETA = 1; // Variable for learning rate
    private final int MAX_ITERATIONS = 1000;
    private boolean STANDARDISE_FLAG;
    private boolean ONLINE_RULE;
    private boolean MODEL_SELECTION;
    private double[] means;
    private double[] stdDev;
    
    /**
     *  Default constructor for the LinearPerceptron classifier
     */
    public EnhancedLinearPerceptron() {
        //this.w = new double[]{1, 1};
        this.bias = 0;
        this.STANDARDISE_FLAG = false;
        this.ONLINE_RULE = true;
        this.MODEL_SELECTION = false;
    }
    
    /**
     * A constructor that takes a value for bias
     * @param bias
     */
    public EnhancedLinearPerceptron(double bias) {
        //this.w = new double[]{1, 1};
        this.bias = bias;
        this.STANDARDISE_FLAG = false;
        this.ONLINE_RULE = true;
        this.MODEL_SELECTION = false;
    }
    
    /**
     * A constructor that allows the user to specify if model selection should
     * be used
     * @param modelSelection 
     */
    public EnhancedLinearPerceptron(boolean modelSelection) {
        //this.w = new double[]{1, 1};
        this.bias = 0;
        this.MODEL_SELECTION = modelSelection;
    }
    
    /**
     * A constructor that takes a value for bias, and allows the user to specify
     * if instances should be standardised using standardiseFlag
     * @param bias 
     * @param standardiseFlag 
     * @param onlineRule 
     */
    public EnhancedLinearPerceptron(double bias, boolean standardiseFlag, boolean onlineRule) {
        //this.w = new double[]{1, 1};
        this.bias = bias;
        this.STANDARDISE_FLAG = standardiseFlag;
        this.ONLINE_RULE = onlineRule;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        Instances newInstances;
        // To do: check that attributes are continuous and not discrete
        
        // Initialise weights
        w = new double[instances.numAttributes()-1];
        Arrays.fill(w, 1);
        // Could try duplicating instances
        if(MODEL_SELECTION){
            // Create classifiers and Evaluation objects
            EnhancedLinearPerceptron online = new EnhancedLinearPerceptron();
            Evaluation evalOnline = new Evaluation(instances);
            EnhancedLinearPerceptron offline = new EnhancedLinearPerceptron(0, false, true);
            Evaluation evalOffline = new Evaluation(instances);
            // Number of folds for crossvalidation
            int folds = (instances.numInstances() > 10) ? 10 : instances.numInstances();
            // Cross-validate
            evalOnline.crossValidateModel(online, instances, folds, new Random(1));
            evalOffline.crossValidateModel(offline, instances, folds, new Random(1));
            //evalOnline.evaluateModel(online, instances);
            System.out.println("Online correct: " + evalOnline.pctCorrect());
            System.out.println("Offline correct: " + evalOffline.pctCorrect());
            System.out.println("Online error: " + evalOnline.errorRate());
            System.out.println("Offline error: " + evalOffline.errorRate());
            
            if(evalOnline.errorRate() <= evalOffline.errorRate()){
                perceptronTraining(instances);
            }
            else{
                gradientDescentTraining(instances);
            }
        }
        else{
            // Standardise attributes if flag is set to true, assign train to
            // instances otherwise
            newInstances = (STANDARDISE_FLAG) ? standardiseAttributes(instances) : instances;
            // Run the on-line perceptron algorithm if flag is true, otherwise use
            // the off-line algorithm
            if(ONLINE_RULE){
                perceptronTraining(newInstances);
            }
            else{
                gradientDescentTraining(newInstances);
            }
        }
        
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double y;
        // Standardise attributes if flag = true
        if(STANDARDISE_FLAG){
            //double x1 = (instance.value(0) - means[0]) / stdDev[0];
            //double x2 = (instance.value(1) - means[1]) / stdDev[1];
            //double calc = w[0] * x1 + w[1] * x2 + bias;
            //y = (calc >= 0) ? 1 : -1;
            
            double calc = 0;
            for(int i = 0; i < instance.numAttributes()-1; i++){
                calc += w[i] * instance.value(i);
            }
            y = (calc >= 0) ? 1 : 0;
        }
        else{
            y = calculateY(instance);
        }
        //System.out.println("Pred: " + y + ". Actual: " + instance.value(2));
        return y;
    }

//    @Override
//    public double[] distributionForInstance(Instance instnc) throws Exception {
//        //throw new UnsupportedOperationException("Not supported yet.");
//        try{
//            System.out.println("Needs doing");
//        }
//        catch(Exception e){
//            System.out.println("Needs doing too");
//        }
//        return null;
//    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NOMINAL_CLASS);
        result.setMinimumNumberInstances(2);

        return result;
    }
    
    /**
     * Return the y value (either 1 or -1) using a threshold of 0
     * @param i the current instance in the training data
     * @return 1 or -1
     */
    private int calculateY(Instance i){
        //double x1 = i.value(0);
        //double x2 = i.value(1);
        //double calc = w[0] * x1 + w[1] * x2 + bias;
        //System.out.println(calc);
        //return(calc >= 0) ? 1 : -1;
        
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
        //double tw[] = new double[2]; // temporary weights
        //tw[0] = w[0] + (0.5*ETA) * (i.value(2) - y) * i.value(0);
        //tw[1] = w[1] + (0.5*ETA) * (i.value(2) - y) * i.value(1);
        //bias = bias + (0.5*ETA) * (i.value(2) - y);
        //return tw;   
        double tw[] = new double[w.length];
        for(int j = 0; j < w.length; j++){
            tw[j] = w[j] + (0.5*ETA) * (i.classValue() - y) * i.value(j);
        }
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
                // Calculate new weights and store in temporary array
                double[] tw = calculateWeights(train.instance(i), y);
                // Check that weights haven't changed
                //count = (tw[0] == w[0] && tw[1] == w[1]) ? count+1 : count*0;
                count = (Arrays.equals(tw, w)) ? count+1 : 0;
                // Assign temporary weights to w[]
                w = tw;
                //w[0] = tw[0];
                //w[1] = tw[1];
                // Increment the number of iterations
                if(count == train.numInstances()) break;
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
            double dw[] = new double[train.numAttributes()-1];
            Arrays.fill(dw, 0);
            for(int i = 0; i < train.numInstances(); i++){
                Instance in = train.instance(i);
                // Calculate y
                double y = calculateY(in);
                // Calculate deltaW
                for(int j = 0; j < dw.length; j++){
                    dw[j] = dw[j] + (0.5*ETA) * (in.classValue() - y) * in.value(j);
                }
            }
            // Update weights
            double[] tw = new double[train.numAttributes()-1]; // temporary weights
            for(int j = 0; j < dw.length; j++){
                tw[j] = w[j] + dw[j];
            }
            // Check that weights haven't changed
            //count = (tw[0] == w[0] && tw[1] == w[1]) ? count+1 : count*0;
            count = (Arrays.equals(tw, w)) ? count+1 : 0;
            // Assign temporary weights to w[]
            w = tw;
            //w[0] = tw[0];
            //w[1] = tw[1];
            // Increment the number of iterations
            if(count == train.numInstances()) break;
            iterations++;
        } while(count < train.numInstances() && iterations <= MAX_ITERATIONS);
    }
    
    private Instances standardiseAttributes(Instances train){
        Instances standardised = new Instances(train);
        means = new double[train.numAttributes()-1];
        stdDev = new double[train.numAttributes()-1];
        
        for(int i = 0; i < standardised.numAttributes()-1; i++){
            AttributeStats attributeStats = standardised.attributeStats(i);
            means[i] = attributeStats.numericStats.mean;
            stdDev[i] = attributeStats.numericStats.stdDev;
            for(int j = 0; j < standardised.numInstances(); j++){
                standardised.get(j).setValue(i, (standardised.get(j).value(i) 
                        - means[i]) / stdDev[i]);
            }
        }
        return standardised;
    }
}
