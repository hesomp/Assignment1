
import java.io.*;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.Resample;


public class Main {

    static PrintWriter _logger= null;


    public static void main(String[] args) throws Exception {

        DateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmss");
        Date date = new Date();

        String filename = "results" + dateFormat.format(date) + ".txt";
        _logger = new PrintWriter(new FileOutputStream(filename), true);

        //split into training and test datasets
        Output("Adult Dataset");
        HashMap<String, Instances> datasets = getTrainingandTestInstances("Adult");
       RunLearningAlgorithms(datasets);


        datasets.clear();
        Output("Cancer Dataset");
        datasets = getTrainingandTestInstances("Cancer");
        RunLearningAlgorithms(datasets);

        _logger.close();

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

        boostedDecisiontree(datasets, (float).01);
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
    
    
    private static void Output(String message){
        _logger.println(message);
        System.out.println(message);
        
        
        
    }



    private static void ClassifyandOutput(Classifier model, HashMap<String, Instances> datasets) throws Exception{
        Instances train = datasets.get("train");
        Instances test = datasets.get("test") ;


        //cross validation
        Evaluation eval = new Evaluation(train);

        long start = System.currentTimeMillis();

        eval.crossValidateModel(model, train, 10, new Random(1));

        long end = System.currentTimeMillis();
        Output("cross train=" + (end - start));

        Output(eval.toSummaryString("\nCross Validation Results\n======\n", false));
        Output(eval.toMatrixString());

        //build classifier
        start = System.currentTimeMillis();
        model.buildClassifier(train);

        end = System.currentTimeMillis();
        Output("build classifier=" + (end - start));


        Evaluation evalTest = new Evaluation(train);

        start = System.currentTimeMillis();
        evalTest.evaluateModel(model, test);
        end = System.currentTimeMillis();
        Output("eval classifier=" + (end - start));


        Output(evalTest.toSummaryString("\nTesting Results\n======\n", false));
        Output(evalTest.toMatrixString());
    }

    private   static HashMap<String, Instances> getTrainingandTestInstances(String dataset) throws IOException {


        HashMap<String, Instances> retVal = new HashMap<>();
        Instances train;
        Instances test;
        BufferedReader trainfile = null;
        BufferedReader testfile = null;


        if (dataset.equals("Adult")){

             trainfile = readDataFile("datasets/adult_train_sample.arff");
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

        Output("model=J48");
        Output("unpruned=" + unpruned);
        Output("confidence=" + confidence);

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

        Output("model=MultilayerPerceptron");
        Output("epochs=" + trainingTime);
        Output("momentum=" + momentum);
        Output("learning=" + learning);


        MultilayerPerceptron model = new MultilayerPerceptron();

        model.setTrainingTime(trainingTime);
        model.setMomentum(momentum);
        model.setLearningRate(learning);

        ClassifyandOutput(model, datasets);



    }

    private static void boostedDecisiontree(HashMap<String, Instances> datasets, float confidence) throws Exception{


        Output("model=AdaBoostM1");
        Output("confidence=" + confidence);

        AdaBoostM1 model = new AdaBoostM1();

        J48 classifier = new J48();
        classifier.setConfidenceFactor(confidence);

        model.setClassifier(classifier);

        ClassifyandOutput(model, datasets);


    }

    private static void SVM(HashMap<String, Instances> datasets, Kernel kernal) throws Exception{


        Output("model=SMO");

        SMO model = new SMO();

        Output("kernel=" + kernal);
        model.setKernel(kernal);

        ClassifyandOutput(model, datasets);
    }

    private static void KNN(HashMap<String, Instances> datasets, int distance) throws Exception{

        Output("model=KNN");
        IBk model = new IBk();

        Output("distance=" + distance);
        model.setKNN(distance);

        ClassifyandOutput(model, datasets);


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
