/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package practica4faa;

import java.util.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author dani
 */
public class MainOptimizacion {
    public static void main(String[] args) throws Exception {
        Random rnd = new Random(System.currentTimeMillis());
        Instances data = ConverterUtils.DataSource.read("data/car.arff");
        data.setClassIndex(data.numAttributes() - 1);      // The class is the last datum of each line
        data.randomize(rnd);
        
        OptimizadorVecinosProximos optimKNN = new OptimizadorVecinosProximos(data, 5);
        optimKNN.Optimizacion();
        
        OptimizadorPerceptron optimPer = new OptimizadorPerceptron(data, 5);
        optimPer.Optimizacion();
    }
}
