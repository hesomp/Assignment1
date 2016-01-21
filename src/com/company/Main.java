package com.company;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class Main {

    public static void main(String[] args) throws Exception {

        Instances ds = getAdultDataset();

        //randomize dataset
        ds.randomize(new java.util.Random(0));

        //split into training and test datasets
        int trainSize = (int) Math.round(ds.numInstances() * 0.8);
        int testSize = ds.numInstances() - trainSize;
        Instances train = new Instances(ds, 0, trainSize);
        Instances test = new Instances(ds, trainSize, testSize);




        J48 decisonTreeModel = new J48();


        Evaluation evaluation = new Evaluation(train);

        decisonTreeModel.buildClassifier(train);
        evaluation.evaluateModel(decisonTreeModel, test);

        FastVector predictions = new FastVector();

        predictions.appendElements(evaluation.predictions());

        double correct= 0;
        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }

        double accuracy = 100 * correct / predictions.size();
         System.out.print(accuracy);


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
