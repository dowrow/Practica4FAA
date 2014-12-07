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
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

/**
 *
 * @author dani
 */
public class OptimizadorVecinosProximos {
    private Instances data;
    private IBk knn;
    private int nFolds;
    private int nFoldsOptimizacion;
    private ArrayList<Integer> mejoresK;
    private int mink;        //valor minimo de k
    private int maxK;        //valor maximo de k
    private int incrementoK; //incrementos de k
    
    
    public OptimizadorVecinosProximos(Instances data, int nFolds){
        this.data = data;
        this.nFolds = nFolds;
        this.nFoldsOptimizacion = this.nFolds - 1;
        this.knn = new IBk();
        this.mejoresK = new ArrayList<>();
        this.mink = 1;
        this.maxK = 300;
        this.incrementoK = 5;
    }
    
    public void Optimizacion(){
        //recorremos todas las particiones de entrenamiento/test
        for(int i = 0; i< this.nFolds; i++){
            Instances training = data.trainCV(nFolds, i);
            Instances test = data.testCV(nFolds, i);
            //ya tenemos las particiones
            //recorremos dentro de test las particiones
            ArrayList<Integer> mejoresKOptim = new ArrayList<>();
            for(int j = 0; j < this.nFoldsOptimizacion; j++){
                Instances trainingTraining = training.trainCV(this.nFoldsOptimizacion, j);
                Instances trainingTest = training.testCV(this.nFoldsOptimizacion, j);
                try {
                    //ya tenemos las particiones de las particiones
                    //ahora debemos recorrer las particiones con los distintos valores
                    // de K
                    Double minErr = 1.0;
                    int mejorK = 1;
                    //en este algoritmo el train se puede hacer sin usar k ^^
                    //por lo que entrenamos antes del bucle
                    this.knn.buildClassifier(trainingTraining);
                    Evaluation eval = new Evaluation(trainingTraining);
                    for(int k = this.mink; k < this.maxK; k += this.incrementoK){
                        if(k%2 == 0){
                            this.knn.setKNN(k - 1);
                        }else{
                            this.knn.setKNN(k);
                        }
                        
                        eval.evaluateModel(this.knn, trainingTest);
                        Double error = eval.errorRate();
                        
                        if(error < minErr){
                            if(k%2 == 0){
                                mejorK = k - 1;
                                minErr = error;
                            }else{
                                mejorK = k;
                                minErr = error;
                            }
                        }
                    }
                    //System.out.println("K--->"+ mejorK +" Err " + minErr + " Fold "+ i+ "-" + j);
                    mejoresKOptim.add(mejorK);
                } catch (Exception ex) {
                    Logger.getLogger(OptimizadorVecinosProximos.class.getName()).log(Level.SEVERE, null, ex);
                }
            }//fin bucle j
            try {
                Double minErr = 1.0;
                int mejorK = 1;
                //ahora que tenemos las mejores K vamos a evaluarlas con el conjunto de test
                //nos quedaremos con la mejor
                this.knn.buildClassifier(training);
                Evaluation eval = new Evaluation(training);
                for(Integer k : mejoresKOptim){
                    this.knn.setKNN(k);
                    eval.evaluateModel(this.knn, test);
                    Double error = eval.errorRate();
                    if(error < minErr){
                        mejorK = k;
                        minErr = error;
                    }
                }
                System.out.println("K--->"+ mejorK +" Err " + minErr + " Fold "+ i);
                this.mejoresK.add(mejorK);
            } catch (Exception ex) {
                Logger.getLogger(OptimizadorVecinosProximos.class.getName()).log(Level.SEVERE, null, ex);
            }
        }//fin bucle i
        System.out.println(this.mejoresK);
    }
    
    private double Evaluacion(Instances dataTrain, Instances dataTest, int k) throws Exception{
        Evaluation eval;
        eval = new Evaluation(dataTrain);
        eval.evaluateModel(knn, dataTest);
        return eval.errorRate();
    }
    
}
