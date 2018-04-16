/*
 * An implementation of the Linear Perceptron that conists of an ensemble of
 * LinearPerceptron classifiers.
 */
package linearperceptronclassifiers;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Joshua Foster
 */
public class LinearPerceptronEnsemble {
    private List<LinearPerceptron> ensembleList;
    private int ENSEMBLE_SIZE;
    
    public LinearPerceptronEnsemble(){
        ensembleList = new ArrayList<>();
        ENSEMBLE_SIZE = 50;
    }
}
