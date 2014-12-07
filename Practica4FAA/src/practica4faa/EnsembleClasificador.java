/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package practica4faa;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author dani
 */
public class EnsembleClasificador implements Classifier, Serializable {
    private IBk knn;
    private NaiveBayes nb;
    private Logistic logistic;
    private MultilayerPerceptron perceptron;
    private ArrayList<Classifier> clasificadores;
    private int numClases;
    
    public EnsembleClasificador(){
        this.knn = new IBk();
        this.nb = new NaiveBayes();
        this.logistic = new Logistic();
        this.perceptron = new MultilayerPerceptron();
        
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        this.knn = new IBk();
        this.nb = new NaiveBayes();
        this.logistic = new Logistic();
        this.perceptron = new MultilayerPerceptron();
        
        //KNN
        this.knn.setKNN(10);
        this.knn.buildClassifier(instances);
        //NaiveBayes
        this.nb.buildClassifier(instances);
        //Regresion
        this.logistic.setMaxIts(500);
        this.logistic.buildClassifier(instances);
        //Perceptron de una sola capa
        this.perceptron.setHiddenLayers("0");                //sin capas ocultas
        this.perceptron.setTrainingTime(500);                // Nr. epochs
        this.perceptron.buildClassifier(instances);
        
        //agregamos los clasificadores a la lista
        this.clasificadores = new ArrayList<>();
        this.clasificadores.add(this.nb);
        this.clasificadores.add(this.knn);
        this.clasificadores.add(this.perceptron);
        this.clasificadores.add(this.logistic);
        
        //necesitamos saber cuantas clases hay
        double[] clases = this.nb.distributionForInstance(instances.firstInstance());
        this.numClases = clases.length;
        //System.out.println(this.numClases);
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        HashMap<Double, Integer> repeticionesClase = new HashMap<>();
        
        //clasificamos con cada clasificador
        for(Classifier c : this.clasificadores){
            double clasif = c.classifyInstance(instnc);
            if(repeticionesClase.containsKey(clasif)){
                int rep = repeticionesClase.get(clasif);
                rep++;
                repeticionesClase.put(clasif, rep);
            }else{
                repeticionesClase.put(clasif, 1);
            }
        }
        
        double clase = 0.0;
        int maxRep = 0;
        //buscamos la clase que mas se repite
        for(Double key : repeticionesClase.keySet()){
            int rep;
            if((rep = repeticionesClase.get(key)) > maxRep){
                maxRep = rep;
                clase = key;
            }
        }
        
        //System.out.println(clase);
        return clase;
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        double [] prediccion = new double[this.numClases];
        int clase = (int)this.classifyInstance(instnc);
        for(int i = 0; i < this.numClases; i++){
            if(i == clase){
                prediccion[i] = 1.0;
            }else{
                prediccion[i] = 0.0;
            }
        }
        return prediccion;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
