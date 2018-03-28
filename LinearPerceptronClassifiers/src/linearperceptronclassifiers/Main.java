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
        // Create instance of LinearPerceptron classifer
        LinearPerceptron linearPerceptron = new LinearPerceptron();
        // Build classifier
        linearPerceptron.buildClassifier(train);
        // Classify instances
        for(int i = 0; i < train.numInstances(); i++){
            linearPerceptron.classifyInstance(train.instance(i));
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
        }
        catch(Exception e){
            System.out.println("Unable to read file. Exception: " + e);
        }
        return i;
    }
}
