
import java.awt.*;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import java.io.IOException;
import java.util.Date;
import java.util.HashMap;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.Resample;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;


public class Main {

    public static void main(String[] args) throws Exception {

        //split into training and test datasets
        System.out.println("Adult Dataset");
        HashMap<String, Instances> datasets = getTrainingandTestInstances("Adult");
       // RunLearningAlgorithms(datasets);


        datasets.clear();
        System.out.println("Cancer Dataset");
        datasets = getTrainingandTestInstances("Cancer");
        RunLearningAlgorithms(datasets);


    }

    public static void RunLearningAlgorithms( HashMap<String, Instances> datasets) throws Exception {


        decisionTree(datasets, (float)0, true);

        decisionTree(datasets, (float).01, false);
        decisionTree(datasets, (float).1, false);
        decisionTree(datasets, (float).2, false);
        decisionTree(datasets, (float).3, false);
        decisionTree(datasets, (float).4, false);
        decisionTree(datasets, (float).5, false);

        neuralNetwork(datasets, 500, .2, .3);
        neuralNetwork(datasets, 1000, .2, .3);
        neuralNetwork(datasets, 2500, .2, .3);
        neuralNetwork(datasets, 5000, .2, .3);

        neuralNetwork(datasets, 500, .4, .6);
        neuralNetwork(datasets, 1000, .4, .6);
        neuralNetwork(datasets, 2500, .4, .6);
        neuralNetwork(datasets, 5000, .4, .6);

        boostedDecisiontree(datasets, (float).0);
        boostedDecisiontree(datasets, (float).1);
        boostedDecisiontree(datasets, (float).2);
        boostedDecisiontree(datasets, (float).3);
        boostedDecisiontree(datasets, (float).4);
        boostedDecisiontree(datasets, (float).5);


        SVM(datasets, new weka.classifiers.functions.supportVector.PolyKernel());
        SVM(datasets, new weka.classifiers.functions.supportVector.RBFKernel());

        KNN(datasets, 1);
        KNN(datasets, 3);
        KNN(datasets, 5);
    }



    private static void ClassifyandOutput(Classifier model, HashMap<String, Instances> datasets) throws Exception{
        Instances train = datasets.get("train");
        Instances test = datasets.get("test") ;


        //cross validation
        Evaluation eval = new Evaluation(train);
        eval.crossValidateModel(model, train, 10, new Random(1));

        System.out.println(eval.toSummaryString("\nCross Validation Results\n======\n", false));
        System.out.println(eval.toMatrixString());

        //build classifier
        long start = System.currentTimeMillis();
        model.buildClassifier(train);

        long end = System.currentTimeMillis();
        System.out.println("build classifier=" + (end - start));


        Evaluation evalTest = new Evaluation(train);

        start = System.currentTimeMillis();
        evalTest.evaluateModel(model, test);
        end = System.currentTimeMillis();
        System.out.println("eval classifier=" + (end - start));


        System.out.println(evalTest.toSummaryString("\nTesting Results\n======\n", false));
        System.out.println(evalTest.toMatrixString());
    }

    private   static HashMap<String, Instances> getTrainingandTestInstances(String dataset) throws IOException {


        HashMap<String, Instances> retVal = new HashMap<>();
        Instances train;
        Instances test;
        BufferedReader trainfile = null;
        BufferedReader testfile = null;


        if (dataset.equals("Adult")){

             trainfile = readDataFile("datasets/adult_train.arff");
             testfile = readDataFile("datasets/adult_test.arff");

        }else if (dataset.equals("Cancer")){

             trainfile = readDataFile("datasets/cancer_train.arff");
             testfile = readDataFile("datasets/cancer_test.arff");

        }


        train = new Instances(trainfile);
        test = new Instances(testfile);

        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);

        retVal.put("train", train);
        retVal.put("test", test);

        return retVal;
    }

    private static void decisionTree(HashMap<String, Instances> datasets, float confidence, boolean unpruned) throws Exception{

        Instances train = datasets.get("train");
        Instances test = datasets.get("test") ;

        System.out.println("model=J48");
        System.out.println("unpruned=" + unpruned);
        System.out.println("confidence=" + confidence);

        J48 model = new J48();
        model.setUnpruned(unpruned);
        model.setConfidenceFactor(confidence);


        ClassifyandOutput(model, datasets);


    }

    private static Instances resample(Instances data) throws Exception {


        Resample r = new Resample();
        r.setNoReplacement(true);
        r.setSampleSizePercent(5);
        r.setInputFormat(data);

       return Filter.useFilter(data, r);
    }

    private static void neuralNetwork(HashMap<String, Instances> datasets, int trainingTime, double momentum, double learning) throws Exception{

        Instances train = datasets.get("train");
        Instances test = datasets.get("test") ;

        System.out.println("model=MultilayerPerceptron");
        System.out.println("epochs=" + trainingTime);
        System.out.println("momentum=" + momentum);
        System.out.println("learning=" + learning);


        MultilayerPerceptron model = new MultilayerPerceptron();

        model.setTrainingTime(trainingTime);
        model.setMomentum(momentum);
        model.setLearningRate(learning);

        ClassifyandOutput(model, datasets);



    }

    private static void boostedDecisiontree(HashMap<String, Instances> datasets, float confidence) throws Exception{


        System.out.println("model=AdaBoostM1");
        System.out.println("confidence=" + confidence);

        AdaBoostM1 model = new AdaBoostM1();

        J48 classifier = new J48();
        classifier.setConfidenceFactor(confidence);

        model.setClassifier(classifier);

        ClassifyandOutput(model, datasets);


    }

    private static void SVM(HashMap<String, Instances> datasets, Kernel kernal) throws Exception{


        System.out.println("model=SMO");

        SMO model = new SMO();

        System.out.println("kernel=" + kernal);
        model.setKernel(kernal);

        ClassifyandOutput(model, datasets);
    }

    private static void KNN(HashMap<String, Instances> datasets, int distance) throws Exception{

        System.out.println("model=KNN");
        IBk model = new IBk();

        System.out.println("distance=" + distance);
        model.setKNN(distance);

        ClassifyandOutput(model, datasets);


    }

    private static Instances getAdultDataset() throws Exception{

        Instances retVal;
        BufferedReader datafile = readDataFile("datasets/adult.arff");


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
