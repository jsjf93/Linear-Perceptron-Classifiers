/**
 * The Main class of the program. Calls instances of the classifiers
 */
package linearperceptronclassifiers;

import java.io.FileReader;
import weka.core.Instances;

/**
 * @author Joshua Foster
 */
public class Main {

    public static void main(String[] args) throws Exception {
        // Load data
        Instances train = loadData("data.arff");
        Instances test = loadData("RandomTestData.arff");
        
        System.out.println("Linear Perceptron: \n");
        // Create instance of LinearPerceptron classifer
        LinearPerceptron linearPerceptron = new LinearPerceptron();
        // Build classifier
        linearPerceptron.buildClassifier(train);
        // Classify instances
        for(int i = 0; i < train.numInstances(); i++){
            linearPerceptron.classifyInstance(train.instance(i));
        }
        
        System.out.println("\nEnhanced Linear Perceptron: \n");
        // Create instance of EnhancedLinearPerceptron classifer
        double bias = -1;
        boolean standardise = true;
        EnhancedLinearPerceptron eln = new EnhancedLinearPerceptron(bias, standardise);
        // Build classifier
        eln.buildClassifier(train);
        // Classify instances
        for(int i = 0; i < train.numInstances(); i++){
            eln.classifyInstance(train.instance(i));
        }
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
}
