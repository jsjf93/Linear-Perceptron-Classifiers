/**
 * The Main class of the program. Calls instances of the classifiers
 */
package linearperceptronclassifiers;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 * @author Joshua Foster
 */
public class Main {

    public static void main(String[] args) throws Exception {
        // Load data
        //Instances train = loadData("data.arff");
        //Instances test = loadData("data.arff");
        Instances dataset = loadData("ForML/blood.arff");
        //Instances test = loadData("ForML/blood.arff");
        
        double average = 0, correct = 0;
        
        
        System.out.println("Linear Perceptron:");
        // Create instance of LinearPerceptron classifer
        LinearPerceptron linearPerceptron = new LinearPerceptron();
        // Build classifier
        linearPerceptron.buildClassifier(dataset);
        // Classify instances
        //testModel(linearPerceptron, test);
        correct = testModel(linearPerceptron, dataset);
        correct = (correct / dataset.numInstances()) * 100;
        System.out.printf("Correct: %.2f%%\n", correct);
        
        System.out.println("\nEnhanced Linear Perceptron (online, no standardisation):");
        // Create instance of EnhancedLinearPerceptron classifer
        double bias = 0;
        boolean standardise = false;
        boolean onlineRule = true;
        EnhancedLinearPerceptron eln = new EnhancedLinearPerceptron(bias, standardise, onlineRule);
        // Build classifier
        eln.buildClassifier(dataset);
        // Classify instances
        correct = testModel(eln, dataset);
        correct = (correct / dataset.numInstances()) * 100;
        System.out.printf("Correct: %.2f%%\n", correct);
        
        System.out.println("\nEnhanced Linear Perceptron (online, standardisation):");
        // Create instance of EnhancedLinearPerceptron classifer
        bias = -0.5;
        standardise = true;
        onlineRule = true;
        EnhancedLinearPerceptron eln1 = new EnhancedLinearPerceptron(bias, standardise, onlineRule);
        // Build classifier
        eln1.buildClassifier(dataset);
        // Classify instances
        correct = testModel(eln1, dataset);
        correct = (correct / dataset.numInstances()) * 100;
        System.out.printf("Correct: %.2f%%\n", correct);
        
        System.out.println("\nEnhanced Linear Perceptron (off-line):");
        // Create instance of EnhancedLinearPerceptron classifer
        bias = 0;
        standardise = false;
        onlineRule = false;
        EnhancedLinearPerceptron eln2 = new EnhancedLinearPerceptron(bias, standardise, onlineRule);
        // Build classifier
        eln2.buildClassifier(dataset);
        // Classify instances
        correct = testModel(eln2, dataset);
        correct = (correct / dataset.numInstances()) * 100;
        System.out.printf("Correct: %.2f%%\n", correct);
        
        System.out.println("\nModel Selection:");
        EnhancedLinearPerceptron modelSelection = new EnhancedLinearPerceptron(true);
        modelSelection.buildClassifier(dataset);
        correct = testModel(modelSelection, dataset);
        correct = (correct / dataset.numInstances()) * 100;
        System.out.printf("Correct: %.2f%%\n", correct);

        System.out.println("\nEnsemble:");
        LinearPerceptronEnsemble lpe = new LinearPerceptronEnsemble();
        lpe.buildClassifier(dataset);
        correct = testModel(lpe, dataset);
        correct = (correct / dataset.numInstances()) * 100;
        System.out.printf("Correct: %.2f%%\n", correct);
        
        System.out.println("\nEnsemble (distribution for instance):");
        getEnsembleClassVotes(lpe, dataset);
        
        
        // ClassifyInstance
        System.out.println("\nEvaluation:");
        for (int i = 0; i < dataset.numInstances(); i++) {
            // Create train and test datasets
            Instances[] split = generateDatasets(dataset);
            // Build classifier
            eln1.buildClassifier(split[0]);
            // Test the model
            correct = testModel(eln1, split[1]);
            correct = (correct / split[1].numInstances()) * 100;
            //System.out.printf("Correct: %.2f%%\n", correct);
            average += correct;
        }
        average /= dataset.numInstances();
        System.out.printf("Average: %.2f%%\n", average);
        
        
        
        
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
    
    
    public static int testModel(Classifier model, Instances test) throws Exception{
        int correct = 0;
        for(int i = 0; i < test.numInstances(); i++){ // for each test instance
            // if classifier c predicts the class of test instance i correctly
            if(model.classifyInstance(test.instance(i))==test.instance(i).classValue()){
                correct++;   // if correct, add 1 to the count
            }
        }
        //System.out.println(correct+" correct out of " + test.numInstances());
        //System.out.println((((double)correct / test.numInstances()) * 100)+"%");
        return correct;
    }
    
    public static void getEnsembleClassVotes(Classifier model, Instances test) throws Exception{
        for (int i = 0; i < test.numInstances(); i++) {
            double[] dist = model.distributionForInstance(test.instance(i));
            //System.out.println("Distribution: " + dist[0] + "/" + dist[1]);
            //System.out.printf("Distribution (1, 0): %.0f / %.0f \n", dist[0], dist[1]);
        }
    }
    
    public static Instances[] generateDatasets(Instances dataset){
        Instances[] split = new Instances[2];
        // Shuffle dataset
        Collections.shuffle(dataset);
        // Set the train/test split sizes
        int trainSize = (int)Math.round(dataset.numInstances() * 0.5);
        int testSize = dataset.numInstances() - trainSize;
        // Create train/test sets
        split[0] = new Instances(dataset, 0, trainSize);
        split[1] = new Instances(dataset, trainSize, testSize);
        return split;
    }
    
    
    public static void writeCSV(String fileName){
        
    }
}
