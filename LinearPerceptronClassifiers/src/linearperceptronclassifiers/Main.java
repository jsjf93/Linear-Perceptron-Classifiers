/**
 * The Main class of the program. Calls instances of the classifiers
 */
package linearperceptronclassifiers;

import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * @author Joshua Foster
 */
public class Main {

    public static void main(String[] args) throws Exception {
        // Load data
        //Instances train = loadData("data.arff");
        //Instances test = loadData("data.arff");
        Instances train = loadData("ForML/bank.arff");
        Instances test = loadData("ForML/bank.arff");
        /*int trainSize = (int) Math.round(bank.numInstances() * 0.8);
        int testSize = bank.numInstances() - trainSize;
        Instances train = new Instances(bank, 0, trainSize);
        Instances test = new Instances(bank, trainSize, testSize);*/
        
        
        System.out.println("Linear Perceptron: \n");
        // Create instance of LinearPerceptron classifer
        LinearPerceptron linearPerceptron = new LinearPerceptron();
        // Build classifier
        linearPerceptron.buildClassifier(train);
        // Classify instances
        testModel(linearPerceptron, test);
        
        System.out.println("\nEnhanced Linear Perceptron (online, no standardisation): \n");
        // Create instance of EnhancedLinearPerceptron classifer
        double bias = 0;
        boolean standardise = false;
        boolean onlineRule = true;
        EnhancedLinearPerceptron eln = new EnhancedLinearPerceptron(bias, standardise, onlineRule);
        // Build classifier
        eln.buildClassifier(train);
        // Classify instances
        testModel(eln, test);
        
        System.out.println("\nEnhanced Linear Perceptron (online, standardisation): \n");
        // Create instance of EnhancedLinearPerceptron classifer
        bias = -0.5;
        standardise = true;
        onlineRule = true;
        EnhancedLinearPerceptron eln1 = new EnhancedLinearPerceptron(bias, standardise, onlineRule);
        // Build classifier
        eln1.buildClassifier(train);
        // Classify instances
        testModel(eln1, test);
        
        System.out.println("\nEnhanced Linear Perceptron (off-line): \n");
        // Create instance of EnhancedLinearPerceptron classifer
        bias = 0;
        standardise = false;
        onlineRule = false;
        EnhancedLinearPerceptron eln2 = new EnhancedLinearPerceptron(bias, standardise, onlineRule);
        // Build classifier
        eln2.buildClassifier(train);
        // Classify instances
        testModel(eln2, test);
        
        System.out.println("\nModel Selection\n");
        EnhancedLinearPerceptron modelSelection = new EnhancedLinearPerceptron(true);
        modelSelection.buildClassifier(train);
        
        testModel(modelSelection, test);
    }
    
    /**
     * A method that takes a file path as a String and reads the data into an
     * Instances object.
     * @param path
     * @return i
     */
    public static Instances loadData(String path){
        Instances i = null;
        try{
            FileReader fr = new FileReader(path);
            i = new Instances(fr);
            i.setClassIndex(i.numAttributes()-1);
        }
        catch(Exception e){
            System.out.println("Unable to read file. Exception: " + e);
        }
        return i;
    }
    
    
    public static void testModel(Classifier model, Instances test) throws Exception{
        int correct = 0;
        for(int i = 0; i < test.numInstances(); i++){ // for each test isntance
            // if classifier c predicts the class of test instance i correctly
            if(model.classifyInstance(test.instance(i))==test.instance(i).classValue()){
                correct++;   // if correct, add 1 to the count. Do nothing otherwise
            }
        }
        System.out.println(correct+" correct out of " + test.numInstances());
        System.out.println((((double)correct / test.numInstances()) * 100)+"%");
    }
}
