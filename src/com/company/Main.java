package com.company;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import java.util.HashMap;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Main {

    public static void main(String[] args) throws Exception {

        Instances ds = getAdultDataset();


        //split into training and test datasets
        HashMap<String, Instances> datasets = getTrainingandTestInstances(ds);

/*
        System.out.format("The value of i is: %f\n",  decisionTree(datasets, (float).1, false));
        System.out.format("The value of i is: %f\n",  decisionTree(datasets, (float).2, false));
        System.out.format("The value of i is: %f\n",  decisionTree(datasets, (float).3, false));
        System.out.format("The value of i is: %f\n",  decisionTree(datasets, (float).4, false));
*/

        System.out.format("The value of i is: %f\n", neuralNetwork(datasets));

    }


    public  static HashMap<String, Instances> getTrainingandTestInstances(Instances dataset){

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

    public  static double decisionTree(HashMap<String, Instances> datasets, float confidence, boolean unpruned) throws Exception{

        J48 model = new J48();

        model.setUnpruned(unpruned);

        model.setConfidenceFactor(confidence);

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

        return (100 * correct / predictions.size());


    }


    public  static double neuralNetwork(HashMap<String, Instances> datasets) throws Exception{

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

        return (100 * correct / predictions.size());


    }




    public static Instances getAdultDataset() throws Exception{

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


    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }





}
