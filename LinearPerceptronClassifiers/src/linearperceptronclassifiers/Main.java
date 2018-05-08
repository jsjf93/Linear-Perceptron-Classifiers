/**
 * The Main class of the program. Calls instances of the classifiers
 */
package linearperceptronclassifiers;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.VotedPerceptron;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Joshua Foster
 */
public class Main {

    public static void main(String[] args) throws Exception {
        // Load data
        Instances dataset = loadData("ForML/acute-inflammation.arff");
        
        double average = 0, correct = 0;
        
        System.out.println("*** Part 2 ***");
        System.out.println("Linear Perceptron:");
        // Create instance of LinearPerceptron classifer
        LinearPerceptron linearPerceptron = new LinearPerceptron();
        // Build classifier
        linearPerceptron.buildClassifier(dataset);
        // Classify instances
        //testModel(linearPerceptron, test);
        correct = testModel(linearPerceptron, dataset);
        correct = (correct / dataset.numInstances()) * 100;
        
        Evaluation e = new Evaluation(dataset);
        e.crossValidateModel(linearPerceptron, dataset, 5, new Random(1));
        //System.out.println(e.fMeasure(0));
        
        System.out.printf("Correct: %.2f%%\n", correct);
        
        System.out.println("\nEnhanced Linear Perceptron (online, no "
                + "standardisation):");
        // Create instance of EnhancedLinearPerceptron classifer
        double bias = 0;
        boolean standardise = false;
        boolean onlineRule = true;
        EnhancedLinearPerceptron eln = new EnhancedLinearPerceptron(bias, 
                standardise, onlineRule);
        // Build classifier
        eln.buildClassifier(dataset);
        // Classify instances
        correct = testModel(eln, dataset);
        correct = (correct / dataset.numInstances()) * 100;
        System.out.printf("Correct: %.2f%%\n", correct);
        
        System.out.println("\nEnhanced Linear Perceptron (online, "
                + "standardisation):");
        // Create instance of EnhancedLinearPerceptron classifer
        bias = -0.5;
        standardise = true;
        onlineRule = true;
        EnhancedLinearPerceptron eln1 = new EnhancedLinearPerceptron(bias, 
                standardise, onlineRule);
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
        EnhancedLinearPerceptron eln2 = new EnhancedLinearPerceptron(bias, 
                standardise, onlineRule);
        // Build classifier
        eln2.buildClassifier(dataset);
        // Classify instances
        correct = testModel(eln2, dataset);
        correct = (correct / dataset.numInstances()) * 100;
        System.out.printf("Correct: %.2f%%\n", correct);
        
        System.out.println("\nModel Selection:");
        EnhancedLinearPerceptron modelSelection = new 
                EnhancedLinearPerceptron(true);
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
        
        System.out.println("*** Part 3 ***");
        
        // Retrieve all the two class ARFFs
        List<String> pathList = walkDirTree("arffstoprocess");
        // Create list of classifier names
        String[] classifierNames = {
            "Enhanced Perceptron(Online)", "Enhanced Perceptron(Offline)",
            "Enhanced Perceptron(Online/Standard)",
            "Enhanced Perceptron(Offline/Standard)",
            "Enhanced Perceptron(Model Selection)",
            "Ensemble (Enhanced/Online/Standard)", "Multilayer Perceptron",
            "SMO", "VotedPeceptron"};
        // Create a List of each classifier to be used
        List<Classifier> classifierList;
        // Declare variables for results
        List<Result> results = new ArrayList<>();
        
        /*
            Find the average accuracy for each classifier on each dataset
        */
        System.out.println("Generating results.csv. Please wait...");
        // Go through each ARFF
        for(String file : pathList){
            dataset = loadData(file);
            classifierList = createClassifiers();
            if(dataset.numClasses() == 2){
                System.out.println("File: " + file);
                // Go through each classifier
                for(int i = 0; i < classifierList.size(); i++){
                    average = 0; correct = 0;
                    System.out.println(" - Classifier: " + classifierNames[i]);
                    // Go through the dataset for the number of instances
                    for(Instance instance : dataset){
                        // Create train and test datasets
                        Instances[] split = generateDatasets(dataset);
                        // Build classifier
                        classifierList.get(i).buildClassifier(split[0]);
                        // Test the model
                        correct = testModel(classifierList.get(i), split[1]);
                        correct = (correct / split[1].numInstances()) * 100;
                        average += correct;
                    }
                    // Calculate average accuracy
                    average /= dataset.numInstances();
                    // Create a Result object                    
                    Result result = new Result(classifierNames[i], file,
                            dataset.numInstances(), dataset.numAttributes(), 
                            average);
                    results.add(result);
                }
            }
        }
        // Write results to CSV
        writeCSV(results);
        
        // Perform timing experiments
        // Select largest file
        dataset = loadData("ForML/breast-cancer-wisc-diag.arff");
        classifierList = createClassifiers();
        // Loop through each classifier
        for(int j = 0; j < classifierList.size(); j++){
            System.out.println(classifierNames[j]);
            // Loop through the datasets increasing the amount of the training
            // data used each time
            for(double i = 0.1; i <= 1; i+=0.1){
                Instances[] split = generateDatasets(dataset);
                int n = (int)Math.round(split[0].size() * i);
                // Create a train set from 0 - n of original train set
                split[0] = new Instances(split[0], 0, n);
                timingTest(classifierList.get(j), split, n,
                        classifierNames[j]);
            }
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
    
    /**
     * Returns the number of correct classifications
     * @param model
     * @param test
     * @return
     * @throws Exception 
     */
    public static int testModel(Classifier model, Instances test) throws 
            Exception{
        int correct = 0;
        for(int i = 0; i < test.numInstances(); i++){ // for each test instance
            // if classifier c predicts the class of test instance i correctly
            if(model.classifyInstance(test.instance(i))==
                    test.instance(i).classValue()){
                correct++;   // if correct, add 1 to the count
            }
        }
        return correct;
    }
    
    /**
     * Outputs the classification votes from the ensemble
     * @param model
     * @param test
     * @throws Exception 
     */
    public static void getEnsembleClassVotes(Classifier model, Instances test) 
            throws Exception{
        for (int i = 0; i < test.numInstances(); i++) {
            double[] dist = model.distributionForInstance(test.instance(i));
            System.out.println("Distribution: " + dist[0] + "/" + dist[1]);
            System.out.printf("Distribution (1, 0): %.0f / %.0f \n", dist[0], 
                    dist[1]);
        }
    }
    
    /**
     * Returns an array consisting of a randomised train and test sets.
     * @param dataset
     * @return 
     */
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
    
    /**
     * Writes a list of classifier accuracy results to a csv file
     * @param results 
     */
    public static void writeCSV(List<Result> results){
        final String FILENAME = "results.csv";
        // File delimiters
        final String DELIMITER = ",";
        final String NEW_LINE = "\n";
        // File Headers
        final String HEADERS = "Classifier,Data,numInstances,numAttributes,"
                + "Accuracy";
        
        FileWriter fw = null;
        
        try{
            fw = new FileWriter(FILENAME);
            fw.append(HEADERS);
            fw.append(NEW_LINE);
            
            // Loop through the Result objects and write to file
            for(Result result : results){
                fw.append(result.getClassifierName());
                fw.append(DELIMITER);
                fw.append(result.getDatasetName());
                fw.append(DELIMITER);
                fw.append(String.valueOf(result.getNumInstances()));
                fw.append(DELIMITER);
                fw.append(String.valueOf(result.getNumAttributes()));
                fw.append(DELIMITER);
                fw.append(String.valueOf(result.getAccuracy()));
                fw.append(NEW_LINE);
            }
            System.out.println("CSV file created. ");
        } 
        catch (IOException e) {
            System.out.println("Error in writeCSV. Exception: " + e);
        } 
        finally{
           try{
               fw.flush();
               fw.close();
           } 
           catch(IOException e){
               System.out.println("Error flushing or closing FileWriter. ");
               System.out.println("Exception: " + e);
           }
        }
    }
    
    /**
     * Returns a list of file paths given a rootfolder
     * @param rootFolder
     * @return
     * @throws Exception 
     */
    public static List<String> walkDirTree(String rootFolder) throws Exception{
        ArrayList<String> pathList = new ArrayList<>();
        Files.walk(Paths.get(rootFolder)).forEach(path ->{
            if(path.toString().contains(".arff")){
                pathList.add(path.toString());
                System.out.println(path.toString());
            }
        });
        return pathList;
    }

    /**
     * Generates a list of classifiers to use for evaluation
     * @return 
     */
    public static List<Classifier> createClassifiers() {
        List<Classifier> classifierList = new ArrayList<>();
        
        // Built classifiers
        //Classifier linearPerceptron = new LinearPerceptron();
        Classifier elp = new EnhancedLinearPerceptron();
        Classifier elpOff = new EnhancedLinearPerceptron(0,false, false);
        Classifier elpSO = new EnhancedLinearPerceptron(-0.5,true, true);
        Classifier elpSOff = new EnhancedLinearPerceptron(-0.5,true, false);
        Classifier modelSelect = new EnhancedLinearPerceptron(true);
        Classifier ensemble_elp = new LinearPerceptronEnsemble();
        // Weka Classifiers
        Classifier mlp = new MultilayerPerceptron();
        Classifier smo = new SMO();
        Classifier vp = new VotedPerceptron();
        
        //classifierList.add(linearPerceptron);
        classifierList.add(elp);
        classifierList.add(elpOff);
        classifierList.add(elpSO);
        classifierList.add(elpSOff);
        classifierList.add(modelSelect);
        classifierList.add(ensemble_elp);
        classifierList.add(mlp);
        classifierList.add(smo);
        classifierList.add(vp);
        
        return classifierList;
    }
    
    /**
     * A timing test for the three searching algorithms written.
     * It records the mean time and the standard deviation to a txt file.
     * @param c
     * @param split
     * @param n
     * @param name
     * @return 
     * @throws java.lang.Exception 
     */
    public static double timingTest(Classifier c,Instances[] split, int n,
            String name) throws Exception{
        
        // Record mean and std deviation of performing an operation reps times
        double sum = 0;
        double sumSquared = 0;
        int reps = split[0].size();
        for(int i = 0; i < reps; i++){
            long t1 = System.nanoTime();
            
            // Build classifier and classify instances
            c.buildClassifier(split[0]);
            for(Instance instance : split[1]){
                c.classifyInstance(instance);
            }
            
            long t2 = System.nanoTime() - t1;
            //Recording it in milli seconds to make it more interpretable
            sum += (double)t2 / 1000000.0;
            sumSquared += (double)(t2 / 1000000.0)*(double)(t2 / 1000000.0);
        }
        double mean = sum / reps;
        double variance = sumSquared / reps - (mean*mean);
        double stdDev = Math.sqrt(variance);
        fileWriter(name, n, mean, stdDev);
        //System.out.println("Mean: " + mean);
        return mean;
    }
    
    /**
     * A method that writes the mean and standard deviation to a text file.
     * @param mean
     * @param stdDev 
     */
    static void fileWriter(String name, int n, double mean, double stdDev){
        
        final String FILENAME = "timing.csv";
        // File delimiters
        final String DELIMITER = ",";
        final String NEW_LINE = "\n";
        // File Headers
        //final String HEADERS = "Classifier,n,mean,std";
        
        FileWriter fw = null;
        
        try{
            fw = new FileWriter(FILENAME, true);
            //fw.append(HEADERS);
            //fw.append(NEW_LINE);
            
            fw.append(name);
            fw.append(DELIMITER);
            fw.append(String.valueOf(n));
            fw.append(DELIMITER);
            fw.append(String.valueOf(mean));
            fw.append(DELIMITER);
            fw.append(String.valueOf(stdDev));
            fw.append(NEW_LINE);
            
            System.out.println("CSV file created. ");
        } 
        catch (IOException e) {
            System.out.println("Error in writeCSV. Exception: " + e);
        } 
        finally{
           try{
               fw.flush();
               fw.close();
           } 
           catch(IOException e){
               System.out.println("Error flushing or closing FileWriter. ");
               System.out.println("Exception: " + e);
           }
        }
        
    }
}
