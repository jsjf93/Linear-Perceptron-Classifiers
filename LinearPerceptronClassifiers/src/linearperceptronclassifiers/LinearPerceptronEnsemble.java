/*
 * An implementation of the Linear Perceptron that conists of an ensemble of
 * LinearPerceptron classifiers.


// TO DO: Could just stick the class value at the end of attributes array
// instead of double arraycopy
 */
package linearperceptronclassifiers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;
import weka.classifiers.AbstractClassifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Joshua Foster
 */
public class LinearPerceptronEnsemble extends AbstractClassifier{
    private final List<LinearPerceptron> ensembleList;
    private final int[][] attributesList;
    private final int ENSEMBLE_SIZE;
    private final double PROPORTION;
    private double[] voteProportions;
    
    /**
     * Default constructor for the Ensemble
     */
    public LinearPerceptronEnsemble(){
        ensembleList = new ArrayList<>();
        ENSEMBLE_SIZE = 50;
        PROPORTION = 0.5;
        attributesList = new int[ENSEMBLE_SIZE][];
        //voteProportions = new double[2];
    }
    
    /**
     * A constructor that allows a user to define the size of the Ensemble and
     * set the proportion of attributes to be selected for the Ensemble subsets
     * @param size 
     * @param proportion 
     */
    public LinearPerceptronEnsemble(int size, int proportion){
        ensembleList = new ArrayList<>();
        ENSEMBLE_SIZE = size;
        PROPORTION = proportion;
        attributesList = new int[ENSEMBLE_SIZE][];
        //voteProportions = new double[2];
    }
    
    /**
     * A method that randomly selects 50% of the attributes and stores their
     * indexes in the attributesList according to the given index that
     * corresponds to the classifier in the same numbered position of the
     * ensembleList
     * @param i - the instances
     * @param index - the index of the classifier
     */
    private void selectAttributes(Instances i, int index){
        // Set number of attributes to be used
        int numAttr = (int)Math.ceil(PROPORTION * (i.numAttributes()-1));
        // Create ordered array of positions (0 - i.numAttributes-1)
        int[] positions = IntStream.range(0, i.numAttributes()-1).toArray();
        // Shuffle positions
        shuffle(positions);
        // Copy positions into attributesList at the given index
        int[] attributes = new int[numAttr+1];
        int[] classIndex = {i.classIndex()};
        System.arraycopy(positions, 0, attributes, 0, numAttr);
        System.arraycopy(classIndex, 0, attributes, numAttr, 1);
        //print(attributes);
        attributesList[index] = attributes;
    }
    
    /**
     * Implementation of Fisher-Yates shuffle algorithm
     * @param array - original array
     */
    private void shuffle(int[] array){
        Random r = new Random();
        int index;
        for (int i = array.length-1; i > 0; i--) {
            index = r.nextInt(i+1);
            int temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }
    
    /**
     * This method first creates a subset of i that has the necessary attributes
     * removed. It then shuffles the the subset and removes the second half of
     * the Instance
     * @param i - the original instances
     * @param index - corresponds to a place in the attributesList and
     *                ensembleList
     * @return subset
     */
    private Instances createSubset(Instances i, int index){
        // Filter out unused attributes
        Instances subset = filterAttributes(i, index);
        // Randomise the subset
        Collections.shuffle(subset);
        // Get the number of instances to remove
        int num = subset.numInstances() - (int)Math.ceil(PROPORTION * 
                (i.numInstances()));
        for(int j = subset.numInstances()-1; j>= num; j--){
            subset.remove(j);
        }
        return subset;
    }
    
    /**
     * A method that filters out the unused attributes and creates a subset
     * @param i
     * @param index
     * @return 
     */
    private Instances filterAttributes(Instances i, int index){
        Instances subset = null;
        int[] attributes = attributesList[index];
        
        Remove attributeFilter = new Remove();
        try{
            attributeFilter.setAttributeIndicesArray(attributes);
            attributeFilter.setInvertSelection(true);
            attributeFilter.setInputFormat(i);
            subset = Filter.useFilter(i, attributeFilter);
        } 
        catch (Exception e) {
            System.out.println("Unable to filter attributes. Caught exception: " 
                    + e);
        }
        
        return subset;
    }
    
    /**
     * A method that builds each of the classifiers in the ensembleList
     * @param train - the training data
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances train) throws Exception {
        for (int i = 0; i < ENSEMBLE_SIZE; i++) {
            // Select attributes for each subset
            selectAttributes(train, i);
            // Create instance subsets
            Instances subset = createSubset(new Instances(train), i); 
            // Create instance of classifier
            LinearPerceptron perceptron = new LinearPerceptron();
            // Build classifier
            perceptron.buildClassifier(subset);
            // Add perceptron to ensemble
            ensembleList.add(perceptron);
        }
    }
    
    /**
     * Gets the classification of an instance from each of the classifiers in
     * the ensembleList and returns the classification with the most votes
     * @param i
     * @return 
     */
    @Override
    public double classifyInstance(Instance i){
        double[] classifications = new double[ENSEMBLE_SIZE];
        voteProportions = new double[2];
        // Go through each classifier 
        for(int j = 0; j < ENSEMBLE_SIZE; j++){
            // Retrieve the attributes relating to the current classifier
            int[] attributes = attributesList[j];
            Arrays.sort(attributes);
            Instance instance = new DenseInstance(i);
            // Remove unwanted attributes
            // a starts at the last element before the class index
            for(int a = i.numAttributes()-2; a >= 0; a--){
                boolean flag = false;
                for(int b = 0; b < attributes.length-1; b++){
                    if(a == attributes[b]){
                        flag = true;
                        break;
                    }
                }
                if(!flag) instance.deleteAttributeAt(a); 
            }
            // Classify
            try{
                classifications[j] = 
                        ensembleList.get(j).classifyInstance(instance);
            }
            catch(Exception e){
                System.out.println("Could not classify instance. Exception: " + 
                        e);
            }
        }
        // Get votes
        for (double classification : classifications) {
            if (classification == 1) voteProportions[0]++; 
            else voteProportions[1]++; 
        }
        return (voteProportions[0] >= voteProportions[1]) ? 1 : 0;
    } 
    
    /**
     * Returns the vote proportions obtained by classifyInstance()
     * @param i
     * @return 
     */
    @Override
    public double[] distributionForInstance(Instance i){
        classifyInstance(i);
        return voteProportions;
    }
}
