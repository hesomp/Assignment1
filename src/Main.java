
import java.io.*;

import java.text.DateFormat;
import java.text.SimpleDateFormat;

import java.util.Date;
import java.util.HashMap;

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
import weka.filters.unsupervised.instance.Resample;


public class Main {

    static PrintWriter _logger= null;
    static String _currentDataset = null;
    static String _modelNum= null;

    public static void main(String[] args) throws Exception {


        DateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmss");
        Date date = new Date();

        String filename = "results" + dateFormat.format(date) + ".txt";
        _logger = new PrintWriter(new FileOutputStream(filename), true);

        if (args.length >=2){
            String model = args[1];

            if (args.length == 3){
                _modelNum = args[2];
            }


            _currentDataset = args[0];
            HashMap<String, BufferedReader> files  = getFiles(_currentDataset);
            ResampleAndRun(files, model);


        }else{
            //run all models on both datasets
            _currentDataset = "Cancer";
            HashMap<String, BufferedReader> files  = getFiles(_currentDataset);
            ResampleAndRun(files, "all");

            files.clear();

            _currentDataset = "Adult";
            files  = getFiles(_currentDataset);
            ResampleAndRun(files, "all");

        }



        _logger.close();

    }

    public static void ResampleAndRun(HashMap<String, BufferedReader> files, String model) throws  Exception{

        Instances trainOrig = new Instances(files.get("train"));
        Instances testOrig = new Instances(files.get("test"));


        for (int i = 100; i>0 ; i = i -10 ){

            HashMap<String, Instances[]> datasets;
            Instances train = Resample(trainOrig, i);
            Instances test  = Resample(testOrig, i);

            train.setClassIndex(train.numAttributes() - 1);
            test.setClassIndex(test.numAttributes() - 1);

            datasets = CrossValidate(train, 10);
            datasets.put("testing", new Instances[] {test});
            datasets.put("training", new Instances[] {train});


            if (model.equals("all")){
                RunLearningAlgorithm(datasets, "dt");
                RunLearningAlgorithm(datasets, "bdt");
                RunLearningAlgorithm(datasets, "svm");
                RunLearningAlgorithm(datasets, "knn");
                RunLearningAlgorithm(datasets, "nn");

            }else{
                RunLearningAlgorithm(datasets, model);

            }





        }
    }

    public static void RunLearningAlgorithm(HashMap<String, Instances[]> datasets, String model) throws Exception {


        switch (model.toLowerCase()){
            case "dt":
                decisionTree(datasets, (float)0, true, 2);
                decisionTree(datasets, (float)0, true, 4);

                decisionTree(datasets, (float).01, false,2);
                decisionTree(datasets, (float).1, false,2);
                decisionTree(datasets, (float).2, false,2);
                decisionTree(datasets, (float).3, false,2);
                decisionTree(datasets, (float).4, false,2);
                decisionTree(datasets, (float).5, false,2);

                decisionTree(datasets, (float).01, false,4);
                decisionTree(datasets, (float).1, false,4);
                decisionTree(datasets, (float).2, false,4);
                decisionTree(datasets, (float).3, false,4);
                decisionTree(datasets, (float).4, false,4);
                decisionTree(datasets, (float).5, false,4);

                break;
            case "bdt":

                boostedDecisiontree(datasets, (float).01,2);
                boostedDecisiontree(datasets, (float).1,2);
                boostedDecisiontree(datasets, (float).2,2);
                boostedDecisiontree(datasets, (float).3,2);
                boostedDecisiontree(datasets, (float).4,2);
                boostedDecisiontree(datasets, (float).5,2);


                boostedDecisiontree(datasets, (float).01,4);
                boostedDecisiontree(datasets, (float).1,4);
                boostedDecisiontree(datasets, (float).2,4);
                boostedDecisiontree(datasets, (float).3,4);
                boostedDecisiontree(datasets, (float).4,4);
                boostedDecisiontree(datasets, (float).5,4);

                break;
            case "svm":

                if(_modelNum.equals(null)){
                    SVM(datasets, new weka.classifiers.functions.supportVector.PolyKernel());
                    SVM(datasets, new weka.classifiers.functions.supportVector.RBFKernel());
                }else if (_modelNum.equals("1")){
                    SVM(datasets, new weka.classifiers.functions.supportVector.PolyKernel());
                }else if (_modelNum.equals("2")){
                    SVM(datasets, new weka.classifiers.functions.supportVector.RBFKernel());
                }

                break;
            case "knn":

                KNN(datasets, 1);
                KNN(datasets, 3);
                KNN(datasets, 5);
                KNN(datasets, 7);
                break;
            case "nn":

                if(_modelNum.equals(null)){
                    neuralNetwork(datasets, "");
                    neuralNetwork(datasets, "a, 2, 5, 6");
                }else if (_modelNum.equals("1")){
                    neuralNetwork(datasets, "");
                }else if (_modelNum.equals("2")){
                    neuralNetwork(datasets, "a, 2, 5, 6");
                }




            break;



        }


    }
    
    
    private static void Output(String message){
        _logger.println(message);
        System.out.println(message);

    }

    private static Instances Resample(Instances dataset, int percentage) throws Exception {

        Resample r = new Resample();
        r.setNoReplacement(true);
        r.setSampleSizePercent(percentage);
        r.setInputFormat(dataset);

        return Filter.useFilter(dataset, r);

    }



    public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
        Instances[][] split = new Instances[2][numberOfFolds];

        for (int i = 0; i < numberOfFolds; i++) {
            split[0][i] = data.trainCV(numberOfFolds, i);
            split[1][i] = data.testCV(numberOfFolds, i);
        }

        return split;
    }


    private static void ClassifyandOutput(Classifier model, HashMap<String, Instances[]> datasets, String desc) throws Exception{


        Instances[] trainingSplits = datasets.get("trainingSplits");
        Instances[] testingSplits = datasets.get("testingSplits") ;
        Instances testing = datasets.get("testing")[0];
        Instances training = datasets.get("training")[0];

        double cvPercentageSum = 0;

        //cross validation

        for (int i = 0; i < trainingSplits.length; i++) {
            long buildTime= 0;
            long evalTime = 0;
            Evaluation eval = new Evaluation(trainingSplits[i]);


            model.buildClassifier(trainingSplits[i]);

            eval.evaluateModel(model, testingSplits[i]);

            cvPercentageSum = cvPercentageSum + eval.pctCorrect();

        }

        //testing

        long buildTime= 0;
        long evalTime = 0;
        Evaluation eval = new Evaluation(training);

        long start = System.currentTimeMillis();

        model.buildClassifier(training);

        long end = System.currentTimeMillis();
        buildTime =(end - start);


        start = System.currentTimeMillis();
        eval.evaluateModel(model, testing);
        end = System.currentTimeMillis();
        evalTime =  (end - start);


        double numLeaves = 0;
        double treeSize = 0;

        if (model.getClass().getName().contains("J48")){
           numLeaves =  ((J48)model).measureNumLeaves();
            treeSize =  ((J48)model).measureTreeSize();
        }


        Output(_currentDataset + "," + desc + "," + training.numInstances() + "," + eval.numInstances() + "," + buildTime + "," +
                evalTime + "," + eval.pctCorrect() + "," + (cvPercentageSum/trainingSplits.length) + "," + numLeaves + "," + treeSize);

        //Output(eval.toMatrixString());

    }


    private static  HashMap<String, BufferedReader> getFiles(String dataset) throws IOException{

        HashMap<String, BufferedReader> retVal = new HashMap<>();

        BufferedReader trainfile = null;
        BufferedReader testfile = null;

        Instances train;
        Instances test;

        if (dataset.toLowerCase().equals("adult")){

            trainfile = readDataFile("datasets/adult_train.arff");
            testfile = readDataFile("datasets/adult_test.arff");

        }else if (dataset.toLowerCase().equals("cancer")){

            trainfile = readDataFile("datasets/cancer_train.arff");
            testfile = readDataFile("datasets/cancer_test.arff");

        }


        retVal.put("train", trainfile);
        retVal.put("test", testfile);

        return retVal;


    }

    private static HashMap<String, Instances[]> CrossValidate(Instances train, int folds) throws IOException {
        HashMap<String, Instances[]> retVal = new HashMap<>();


        // Do 10-split cross validation
        Instances[][] split = crossValidationSplit(train, folds);

        // Separate split into training and testing arrays
        Instances[] trainingSplits = split[0];
        Instances[] testingSplits = split[1];


        retVal.put("trainingSplits", trainingSplits);
        retVal.put("testingSplits", testingSplits);



        return retVal;
    }




    private static void decisionTree(HashMap<String, Instances[]> datasets, float confidence, boolean unpruned, int minInstancePerLeaf) throws Exception{


        J48 model = new J48();
        model.setUnpruned(unpruned);
        model.setConfidenceFactor(confidence);
        model.setMinNumObj(minInstancePerLeaf);

        String desc = model.getClass().getName()  + "," + unpruned + "," + confidence + "," + minInstancePerLeaf;

        ClassifyandOutput(model, datasets, desc);

    }



    private static void neuralNetwork(HashMap<String, Instances[]> datasets, String hiddenLayers) throws Exception{


        MultilayerPerceptron model = new MultilayerPerceptron();
        if (!hiddenLayers.equals("")){
            model.setHiddenLayers(hiddenLayers);
        }


        String desc = model.getClass().getName() + "," + model.getHiddenLayers().replace(",", "|") + ",N/A,N/A";
        ClassifyandOutput(model, datasets, desc);

    }

    private static void boostedDecisiontree(HashMap<String, Instances[]> datasets, float confidence, int minInstancePerLeaf) throws Exception{



        AdaBoostM1 model = new AdaBoostM1();

        J48 classifier = new J48();
        classifier.setConfidenceFactor(confidence);
        classifier.setMinNumObj(minInstancePerLeaf);

        model.setClassifier(classifier);

        String desc = model.getClass().getName() + "," + confidence + "," + minInstancePerLeaf + ", N/A";

        ClassifyandOutput(model, datasets, desc);


    }

    private static void SVM(HashMap<String, Instances[]> datasets, Kernel kernal) throws Exception{

        SMO model = new SMO();
        model.setKernel(kernal);

        String desc = model.getClass().getName() + "," + kernal.getClass().getName() + ", N/A, N/A";

        ClassifyandOutput(model, datasets, desc);
    }

    private static void KNN(HashMap<String, Instances[]> datasets, int distance) throws Exception{


        IBk model = new IBk();


        model.setKNN(distance);
        String desc = model.getClass().getName() + "," + distance + ",N/A, N/A";
        ClassifyandOutput(model, datasets, desc);


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
