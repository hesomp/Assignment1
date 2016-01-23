package com.company;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import java.util.HashMap;

import org.apache.commons.lang3.time.StopWatch;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveRange;
import weka.filters.unsupervised.instance.Resample;


public class Main {

    public static void main(String[] args) throws Exception {

        Instances ds = getAdultDataset();


        ds = resample(ds);



        //split into training and test datasets
        HashMap<String, Instances> datasets = getTrainingandTestInstances(ds);

/*
        System.out.format("The value of i is: %f\n",  decisionTree(datasets, (float).1, false));
        System.out.format("The value of i is: %f\n",  decisionTree(datasets, (float).2, false));
        System.out.format("The value of i is: %f\n",  decisionTree(datasets, (float).3, false));
        System.out.format("The value of i is: %f\n",  decisionTree(datasets, (float).4, false));

      // System.out.format("The value of i is: %f\n", neuralNetwork(datasets));

        System.out.format("The value of i is: %f\n", boostedDecisiontree(datasets));

        System.out.format("The value of i is: %f\n", svm(datasets));
*/
        System.out.format("The value of i is: %f\n", knn(datasets,8));

    }


    private   static HashMap<String, Instances> getTrainingandTestInstances(Instances dataset){

        HashMap<String, Instances> retVal = new HashMap<>();

        dataset.randomize(new java.util.Random(0));

        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
        int testSize = dataset.numInstances() - trainSize;
        Instances train = new Instances(dataset, 0, trainSize);
        Instances test = new Instances(dataset, trainSize, testSize);

        //System.out.println(test);

        retVal.put("train", train);
        retVal.put("test", test);

        return retVal;
    }

    private   static double decisionTree(HashMap<String, Instances> datasets, float confidence, boolean unpruned) throws Exception{

        StopWatch st = new StopWatch();

        J48 model = new J48();

        model.setUnpruned(unpruned);

        model.setConfidenceFactor(confidence);

        Instances train = datasets.get("train");
        Instances test = datasets.get("test") ;


        Evaluation evaluation = new Evaluation(train);

        st.start();

        model.buildClassifier(train);

        st.stop();
        System.out.println("train=" + st.getTime());


        st.reset();
        st.start();
        evaluation.evaluateModel(model, test);

        st.stop();
        System.out.println("test=" + st.getTime());

        FastVector predictions = new FastVector();

        predictions.appendElements(evaluation.predictions());

        double correct= 0;
        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }


        System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));


        return (100 * correct / predictions.size());


    }


    private   static Instances resample(Instances data) throws Exception {


        Resample r = new Resample();
        r.setNoReplacement(true);
        r.setSampleSizePercent(25);
        r.setInputFormat(data);

       return Filter.useFilter(data, r);
    }


    private   static double neuralNetwork(HashMap<String, Instances> datasets) throws Exception{

        MultilayerPerceptron model = new MultilayerPerceptron();



        Instances train = datasets.get("train");
        Instances test = datasets.get("test") ;



        Evaluation evaluation = new Evaluation(train);

        model.buildClassifier(train);

        evaluation.evaluateModel(model, test);

        FastVector predictions = new FastVector();

        predictions.appendElements(evaluation.predictions());

        double correct= 0;
        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }


        System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));

        return (100 * correct / predictions.size());


    }

    private   static double boostedDecisiontree(HashMap<String, Instances> datasets) throws Exception{

        AdaBoostM1 model = new AdaBoostM1();

        Instances train = datasets.get("train");
        Instances test = datasets.get("test") ;

        model.setClassifier(new J48());

        Evaluation evaluation = new Evaluation(train);

        model.buildClassifier(train);

        evaluation.evaluateModel(model, test);

        FastVector predictions = new FastVector();

        predictions.appendElements(evaluation.predictions());

        double correct= 0;
        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }


        System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));

        return (100 * correct / predictions.size());


    }



    private   static double svm(HashMap<String, Instances> datasets) throws Exception{

        SMO model = new SMO();

        Instances train = datasets.get("train");
        Instances test = datasets.get("test") ;

        model.setKernel(new weka.classifiers.functions.supportVector.PolyKernel());

        Evaluation evaluation = new Evaluation(train);

        model.buildClassifier(train);

        evaluation.evaluateModel(model, test);

        FastVector predictions = new FastVector();

        predictions.appendElements(evaluation.predictions());

        double correct= 0;
        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }


        System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));

        return (100 * correct / predictions.size());


    }

    private static double knn(HashMap<String, Instances> datasets, int distance) throws Exception{

        IBk model = new IBk();

        Instances train = datasets.get("train");
        Instances test = datasets.get("test") ;

        model.setKNN(distance);

        Evaluation evaluation = new Evaluation(train);

        model.buildClassifier(train);

        evaluation.evaluateModel(model, test);

        FastVector predictions = new FastVector();

        predictions.appendElements(evaluation.predictions());

        double correct= 0;
        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }


        System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));

        return (100 * correct / predictions.size());


    }


    private static Instances getAdultDataset() throws Exception{

        Instances retVal;
        BufferedReader datafile = readDataFile("adult.arff");


        Instances data = new Instances(datafile);


        data.setClassIndex(data.numAttributes() - 1);

        Remove remove = new Remove();
        remove.setAttributeIndices("3");

        remove.setInputFormat(data);
        retVal = Filter.useFilter(data, remove);

        return retVal;

    }

    private static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }





}
