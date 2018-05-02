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
        //Instances train = loadData("data.arff");
        //Instances test = loadData("data.arff");
        Instances dataset = loadData("ForML/breast-cancer-wisc-diag.arff");
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
        
        Evaluation e = new Evaluation(dataset);
        e.crossValidateModel(linearPerceptron, dataset, 5, new Random(1));
        //System.out.println(e.fMeasure(0));
        
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
//        System.out.println("\nEvaluation:");
//        for (int i = 0; i < dataset.numInstances(); i++) {
//            // Create train and test datasets
//            Instances[] split = generateDatasets(dataset);
//            // Build classifier
//            lpe.buildClassifier(split[0]);
//            // Test the model
//            correct = testModel(lpe, split[1]);
//            correct = (correct / split[1].numInstances()) * 100;
//            //System.out.printf("Correct: %.2f%%\n", correct);
//            average += correct;
//        }
//        average /= dataset.numInstances();
//        System.out.printf("Average: %.2f%%\n", average);
        
        
        System.out.println("\n***********************************************\n");
        
        // Retrieve all the two class ARFFs
        List<String> pathList = walkDirTree("arffstoprocess");
        // Create list of classifier names
//        String[] classifierNames = {"Linear Perceptron", 
//            "Enhanced Perceptron(Online)", "Enhanced Perceptron(Online/Standard)",
//            "Enhanced Perceptron(Offline)","Enhanced Perceptron(Offline/Standard)",
//            "Ensemble (Enhanced/Online/Standard)", "Multilayer Perceptron",
//            "SMO", "VotedPeceptron"};
        String[] classifierNames = {"Linear Perceptron",
            "Enhanced Perceptron(Online)",
            "Enhanced Perceptron(Online/Standard)",
            "Enhanced Perceptron(Offline)",
            "Multilayer Perceptron",
            "SMO", "VotedPeceptron"};
        // Create a List of each classifier to be used
        List<Classifier> classifierList;
        
        // Declare variables for results
        
        List<Result> results = new ArrayList<>();
        //Instances largestARFF = loadData("ForML/ozone.arff");
        
        int count = 1;
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
                    System.out.println(" - Classifier: "+classifierNames[i]);
                    Classifier c = classifierList.get(i);
                    
                    // Go through the dataset for the number of instances
                    for(Instance instance : dataset){
                        // Create train and test datasets
                        Instances[] split = generateDatasets(dataset);
                        // Build classifier
                        c.buildClassifier(split[0]);
                        // Test the model
                        correct = testModel(c, split[1]);
                        correct = (correct / split[1].numInstances()) * 100;
                        average += correct;
                    }
                    // Calculate average accuracy
                    average /= dataset.numInstances();
                    // Cross-validate model to more statistics
                    double accuracy, balancedAccuracy, recall, precision, fMeasure;
                    Evaluation eval = new Evaluation(dataset);
                    eval.crossValidateModel(c, dataset, 10, new Random(1));
                    //System.out.println(eval.toSummaryString());;
                    
                    //System.out.println("Error rate"+eval.errorRate());
                    accuracy = (eval.numTruePositives(1) + 
                                eval.trueNegativeRate(1)) /
                                eval.numInstances();
                    balancedAccuracy = (eval.truePositiveRate(1) + 
                                        eval.trueNegativeRate(1)) / 2;
                    recall = eval.recall(1);
                    precision = eval.precision(1);
                    fMeasure = eval.fMeasure(1);
                    
                    // Start timing test for single pass of buildClassifier
                    long startTime = System.nanoTime();
                    c.buildClassifier(dataset);
                    long endTime = System.nanoTime();
                    long duration = endTime - startTime;
//                    Result result = new Result(classifierNames[i], file,
//                                        dataset.numInstances(),
//                                        dataset.numAttributes(),
//                                        average, duration);
                    // Create a Result object to store data
                    Result result = new Result(classifierNames[i], file,
                                        dataset.numInstances(),
                                        dataset.numAttributes(),
                                        average, accuracy, balancedAccuracy,
                                        recall, precision, fMeasure, duration);
                    results.add(result); // Add Result to list
                    count++;
                }
            }
        }
        // Write results to CSV
        writeCSV(results);
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
    
    
    public static void writeCSV(List<Result> results){
        final String FILENAME = "results.csv";
        // File delimiters
        final String DELIMITER = ",";
        final String NEW_LINE = "\n";
        // File Headers
        final String HEADERS = "Classifier,Data,numInstances,numAttributes,"
                + "AvgAccuracy(%),Accuracy,BalancedAccuracy,Recall, Precision,"
                + "fMeasure,Time(nano)";
        
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
                fw.append(String.valueOf(result.getAverageAccuracy()));
                fw.append(DELIMITER);
                fw.append(String.valueOf(result.getAccuracy()));
                fw.append(DELIMITER);
                fw.append(String.valueOf(result.getBalancedAccuracy()));
                fw.append(DELIMITER);
                fw.append(String.valueOf(result.getRecall()));
                fw.append(DELIMITER);
                fw.append(String.valueOf(result.getPrecision()));
                fw.append(DELIMITER);
                fw.append(String.valueOf(result.getfMeasure()));
                fw.append(DELIMITER);
                fw.append(String.valueOf(result.getTiming()));
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

    public static List<Classifier> createClassifiers() {
        List<Classifier> classifierList = new ArrayList<>();
        
        // Built classifiers
        Classifier linearPerceptron = new LinearPerceptron();
        Classifier elp = new EnhancedLinearPerceptron();
        Classifier elpOS = new EnhancedLinearPerceptron(-0.5,true, true);
        Classifier elpOff = new EnhancedLinearPerceptron(0,false, false);
        //Classifier elpOffS = new EnhancedLinearPerceptron(-0.5,true, false);
        //Classifier ensemble_elp = new LinearPerceptronEnsemble();
        // Weka Classifiers
        Classifier mlp = new MultilayerPerceptron();
        Classifier smo = new SMO();
        Classifier vp = new VotedPerceptron();
        
        //classifierList.add(linearPerceptron);
        classifierList.add(elp);
        classifierList.add(elpOS);
        classifierList.add(elpOff);
//        classifierList.add(elpOffS);
        //classifierList.add(ensemble_elp);
        classifierList.add(mlp);
        classifierList.add(smo);
        classifierList.add(vp);
        
        return classifierList;
    }
}
