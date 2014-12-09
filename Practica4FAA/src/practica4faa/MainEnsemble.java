/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package practica4faa;

import java.io.File;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author dani
 */
public class MainEnsemble {
    public static void main(String[] args) throws Exception {
        Random rnd = new Random(System.currentTimeMillis());
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File("data/credit-g.arff"));
        Instances structure = loader.getStructure();
        Instances data = loader.getDataSet();
        structure.setClassIndex(structure.numAttributes() - 1);
        
        /*DataSource source = new DataSource("data/balan.arff");
        Instances data = source.getDataSet();*/
        //Instances data = ConverterUtils.DataSource.read("data/balan.arff");
        data.setClassIndex(data.numAttributes() - 1);      // The class is the last datum of each line
        data.randomize(rnd);
        
        //EnsembleClasificador ensem = new EnsembleClasificador();
        EnsembleClasificadorClasificado ensem = new EnsembleClasificadorClasificado();
        
        Evaluation eval;
        eval = new Evaluation(data);
        eval.crossValidateModel(ensem, data, 5, rnd); // 5-fold
        System.out.println("Ensemble\n----------" + eval.toSummaryString());
        
        System.out.println();
    }
}
