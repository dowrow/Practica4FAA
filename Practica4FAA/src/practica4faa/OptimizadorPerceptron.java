/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package practica4faa;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

/**
 *
 * @author dani
 */
public class OptimizadorPerceptron {
    private Instances data;
    private int nFolds;
    private int nFoldsOptimizacion;
    private ArrayList<Integer> mejoresPer;
    private int minPer;        //valor minimo de numero de neuronas
    private int maxPer;        //valor maximo de numero de neuronas
    private int incrementoPer; //incrementos de numero de neuronas
    private MultilayerPerceptron perceptron;
    
    public OptimizadorPerceptron(Instances data, int nFolds){
        this.data = data;
        this.nFolds = nFolds;
        this.nFoldsOptimizacion = this.nFolds - 1;
        this.mejoresPer = new ArrayList<>();
        this.minPer = 1;
        this.maxPer = 30;
        this.incrementoPer = 5;
        this.perceptron = new MultilayerPerceptron();
    }
    
    public void Optimizacion(){
        //recorremos todas las particiones de entrenamiento/test
        for(int i = 0; i< this.nFolds; i++){
            Instances training = data.trainCV(nFolds, i);
            Instances test = data.testCV(nFolds, i);
            //ya tenemos las particiones
            //recorremos dentro de test las particiones
            ArrayList<Integer> mejoresNPerOptim = new ArrayList<>();
            for(int j = 0; j < this.nFoldsOptimizacion; j++){
                Instances trainingTraining = training.trainCV(this.nFoldsOptimizacion, j);
                Instances trainingTest = training.testCV(this.nFoldsOptimizacion, j);
                try {
                    //ya tenemos las particiones de las particiones
                    //ahora debemos recorrer las particiones con los distintos valores
                    // de K
                    Double minErr = 1.0;
                    int mejorNperc = 1;
                    //en este algoritmo el train se puede hacer sin usar k ^^
                    //por lo que entrenamos antes del bucle
                    
                    Evaluation eval = new Evaluation(trainingTraining);
                    for(int percInLayer = this.minPer; percInLayer< this.maxPer; percInLayer += this.incrementoPer){
                        this.perceptron.setHiddenLayers(""+percInLayer);
                        this.perceptron.buildClassifier(trainingTraining);
                        
                        eval.evaluateModel(this.perceptron, trainingTest);
                        Double error = eval.errorRate();
                        
                        if(error < minErr){
                            mejorNperc = percInLayer;
                            minErr = error;
                        }
                    }
                    //System.out.println("K--->"+ mejorK +" Err " + minErr + " Fold "+ i+ "-" + j);
                    mejoresNPerOptim.add(mejorNperc);
                } catch (Exception ex) {
                    Logger.getLogger(OptimizadorVecinosProximos.class.getName()).log(Level.SEVERE, null, ex);
                }
            }//fin bucle j
            try {
                Double minErr = 1.0;
                int mejorPer = 1;
                //ahora que tenemos las mejores K vamos a evaluarlas con el conjunto de test
                //nos quedaremos con la mejor

                Evaluation eval = new Evaluation(training);
                for(Integer nper : mejoresNPerOptim){
                    this.perceptron.setHiddenLayers(""+nper);
                    this.perceptron.buildClassifier(training);
                    eval.evaluateModel(this.perceptron, test);
                    Double error = eval.errorRate();
                    if(error < minErr){
                        mejorPer = nper;
                        minErr = error;
                    }
                }
                System.out.println("Neuronas--->"+ mejorPer +" Err " + minErr + " Fold "+ i);
                this.mejoresPer.add(mejorPer);
            } catch (Exception ex) {
                Logger.getLogger(OptimizadorVecinosProximos.class.getName()).log(Level.SEVERE, null, ex);
            }
        }//fin bucle i
        System.out.println(this.mejoresPer);
    }
}
