/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package practica4faa;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author dani
 */
public class EnsembleClasificadorClasificado implements Classifier, Serializable {
    private IBk knn;
    private NaiveBayes nb;
    private MultilayerPerceptron clasifTotal;
    private Logistic logistic;
    private MultilayerPerceptron perceptron;
    private ArrayList<Classifier> clasificadores;
    private int numClases;
    
    public EnsembleClasificadorClasificado(){
        this.knn = new IBk();
        this.nb = new NaiveBayes();
        this.logistic = new Logistic();
        this.perceptron = new MultilayerPerceptron();
        this.clasifTotal = new MultilayerPerceptron();
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
        this.perceptron.setHiddenLayers("10");               //una sola capa oculta
        this.perceptron.setTrainingTime(500);                // Nr. epochs
        this.perceptron.buildClassifier(instances);
        
        //agregamos los clasificadores a la lista
        this.clasificadores = new ArrayList<>();
        this.clasificadores.add(this.nb);
        this.clasificadores.add(this.knn);
        this.clasificadores.add(this.perceptron);
        this.clasificadores.add(this.logistic);
        
        //necesitamos saber cuantas clases hay
   
        Attribute at = instances.classAttribute();
        this.numClases = at.numValues();
        
        //System.out.println(this.numClases);
        
        //ahora vamos a entrenar un nuevo clasificador con los resultados que dan
        //por ejemplo naive bayes
        
        //primero creamos las instancias nuevas
        Attribute Attribute1 = new Attribute("knn");
        Attribute Attribute2 = new Attribute("nb");
        Attribute Attribute3 = new Attribute("log");
        Attribute Attribute4 = new Attribute("per");
         // Declare the class attribute along with its values
        FastVector fvClassVal = new FastVector(this.numClases);
        Enumeration enu = at.enumerateValues();
        while(enu.hasMoreElements()){
            fvClassVal.addElement(enu.nextElement());
        }
        /*for(Object cs : ){
            fvClassVal.addElement(cs+"");
        }*/
        
        Attribute ClassAttribute = new Attribute("clase", fvClassVal);
        
        
        FastVector fvWekaAttributes = new FastVector(5);
        fvWekaAttributes.addElement(Attribute1);    
        fvWekaAttributes.addElement(Attribute2);    
        fvWekaAttributes.addElement(Attribute3);    
        fvWekaAttributes.addElement(Attribute4);   
        fvWekaAttributes.addElement(ClassAttribute);
        // Create an empty training set
        Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, instances.numInstances());         
        // Set class index
        isTrainingSet.setClassIndex(4);
        
        for(Instance instance : instances){
            Instance iExample = new DenseInstance(5);
            Double d0 = this.knn.classifyInstance(instance);
            iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), d0);
            Double d1 = this.nb.classifyInstance(instance);
            iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), d1);
            Double d2 = this.logistic.classifyInstance(instance);
            iExample.setValue((Attribute)fvWekaAttributes.elementAt(2), d2);
            Double d3 = this.perceptron.classifyInstance(instance);
            iExample.setValue((Attribute)fvWekaAttributes.elementAt(3), d3);
            String c = instance.stringValue(instance.classIndex());
            iExample.setValue((Attribute)fvWekaAttributes.elementAt(4), c);
            
            // add the instance
            isTrainingSet.add(iExample);
        }
        //Perceptron de una sola capa
        this.perceptron.setHiddenLayers("10");               //una sola capa oculta
        this.perceptron.setTrainingTime(500);                // Nr. epochs
        this.clasifTotal.buildClassifier(isTrainingSet);
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
        return this.distributionForInstanceUno(instnc);
    }
    
    public double[] distributionForInstanceUno(Instance instnc) throws Exception{
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
    public double[] distributionForInstanceDos(Instance instnc) throws Exception{
        double [] prediccion = new double[this.numClases];
        for(int i = 0; i<this.numClases; i++ ){
            prediccion[i] = 0;
        }
        for(Classifier c : this.clasificadores){
            double [] prClas = c.distributionForInstance(instnc);
            prediccion = this.sumaVectores(prediccion, prClas);
        }
        
        return this.multiplicaVectorPorCTE(prediccion, 1.0/(double)this.clasificadores.size());
    }
    public double[] distributionForInstanceTres(Instance instnc) throws Exception{
        //primero creamos las instancias nuevas
        Attribute Attribute1 = new Attribute("knn");
        Attribute Attribute2 = new Attribute("nb");
        Attribute Attribute3 = new Attribute("log");
        Attribute Attribute4 = new Attribute("per");
        Attribute at = instnc.classAttribute();
        // Declare the class attribute along with its values
        FastVector fvClassVal = new FastVector(this.numClases);
        Enumeration enu = at.enumerateValues();
        while(enu.hasMoreElements()){
            fvClassVal.addElement(enu.nextElement());
        }
        Attribute ClassAttribute = new Attribute("clase", fvClassVal);
        
        FastVector fvWekaAttributes = new FastVector(5);
        fvWekaAttributes.addElement(Attribute1);    
        fvWekaAttributes.addElement(Attribute2);    
        fvWekaAttributes.addElement(Attribute3);    
        fvWekaAttributes.addElement(Attribute4);   
        fvWekaAttributes.addElement(ClassAttribute);
        
        // Create an empty training set
        Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, 1);         
        // Set class index
        isTrainingSet.setClassIndex(4);
        
        Instance iExample = new DenseInstance(5);
        Double d0 = this.knn.classifyInstance(instnc);
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), d0);
        Double d1 = this.nb.classifyInstance(instnc);
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), d1);
        Double d2 = this.logistic.classifyInstance(instnc);
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(2), d2);
        Double d3 = this.perceptron.classifyInstance(instnc);
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(3), d3);
        
        // add the instance
        isTrainingSet.add(iExample);
        iExample = isTrainingSet.firstInstance();
        //double clase = this.clasifTotal.distributionForInstance(iExample);
        //double clase = this.clasifTotal.classifyInstance(iExample);
        return this.clasifTotal.distributionForInstance(iExample);
    }
    private double[] sumaVectores(double[] v1, double[] v2){
        double [] v3 = new double[v1.length];
        for(int i = 0; i < v1.length; i++){
            v3[i] = v1[i] + v2[i];
        }
        return v3;
    }
    
    private double[] multiplicaVectorPorCTE(double[] v1, double cte){
        double [] v3 = new double[v1.length];
        for(int i = 0; i < v1.length; i++){
            v3[i] = v1[i] * cte;
        }
        return v3;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
