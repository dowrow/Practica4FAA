package practica4faa;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Profesores FAA
 */
public class Practica4FAA {
    
    public static void main(String[] args) throws Exception {
        Random rnd = new Random(System.currentTimeMillis());
        Instances data = DataSource.read("data/credit-g.arff");
        data.setClassIndex(data.numAttributes() - 1);      // The class is the last datum of each line
        data.randomize(rnd);
        Evaluation eval;
        
        MultilayerPerceptron perceptron = new MultilayerPerceptron();
        perceptron.setHiddenLayers("3");                   // Perceptron parameter: nr. of neurons per layer, separated by commas
        perceptron.setTrainingTime(500);                   // Nr. epochs
        eval = new Evaluation(data);
        eval.crossValidateModel(perceptron, data, 5, rnd); // 5-fold
        System.out.println("Perceptron\n----------" + eval.toSummaryString());
        
        IBk knn = new IBk();
        knn.setKNN(10);                                    // Nr. neighbors
        eval = new Evaluation(data);
        eval.crossValidateModel(knn, data, 5, rnd);        // 5-fold
        System.out.println("Nearest neighbors\n-----------------" + eval.toSummaryString());
        
        NaiveBayes nb = new NaiveBayes();
        eval = new Evaluation(data);
        eval.crossValidateModel(nb, data, 5, rnd);         // 5-fold
        System.out.println("Naive Bayes\n-----------" + eval.toSummaryString());
        
        Logistic logistic = new Logistic();
        logistic.setMaxIts(500);                           // Nr. epochs
        eval = new Evaluation(data);
        eval.crossValidateModel(logistic, data, 5, rnd);   // 5-fold
        System.out.println("Logistic regression\n-------------------" + eval.toSummaryString());

        // Logistic regression again, but running the folds one by one
        double error = 0;
        int nFolds = 5;
        for (int fold = 0; fold < nFolds; fold++) {
            Instances training = data.trainCV(nFolds, fold);
            Instances test = data.testCV(nFolds, fold);
            logistic.buildClassifier(training);
            eval = new Evaluation(training);
            eval.evaluateModel(logistic, test);
            error += eval.errorRate();
        }
        System.out.println("Logistic regression\n-------------------\nAvg. error rate:" + error / nFolds);
    }    
}