package com.parakweet.emailintent;

import java.text.DecimalFormat;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class WekaEmailIntentClassifier {

	public static void main(String[] args) throws Exception {
		
		if (args.length != 2) {
			System.out.println("Usage: WekaSpeechActClassifier <train_set_input_file> <test_set_input_file>");
			System.exit(0);
		}
		
		String arffFileTrain = args[0];
		String arffFileTest = args[1];

		LibSVM wekaClassifier = new LibSVM();
		wekaClassifier.setOptions(new String[] {"-B", "-H"});

		Instances preparedData = (Instances) SerializationHelper.read(arffFileTrain);
		Instances preparedTest = (Instances) SerializationHelper.read(arffFileTest);
		
		System.out.println("Reading train set and test set done!");

		System.out.print("\nTraining...");
		wekaClassifier.buildClassifier(preparedData);
		
		System.out.println("\nTraining...done!");
		
		Evaluation evalTrain = new Evaluation(preparedData);
		evalTrain.evaluateModel(wekaClassifier, preparedData);

		DecimalFormat formatter = new DecimalFormat("#0.0");
		
		System.out.println("\nEvaluating on trainSet...");
		System.out.println(evalTrain.toSummaryString());
		
		System.out.println("\nResult on trainSet...");
		System.out.println("Precision:" + formatter.format(100*evalTrain.precision(0)) + "%" +
				" - Recal: " + formatter.format(100*evalTrain.recall(0)) + "%" +
				" - F1: " + formatter.format(evalTrain.fMeasure(0)) + "%");
		
		Evaluation eval = new Evaluation(preparedTest);
		eval.evaluateModel(wekaClassifier, preparedTest);

		System.out.println("\nEvaluating on testSet...");
		System.out.println(eval.toSummaryString());
		
		System.out.println("\nResult on testSet...");
		System.out.println("Precision:" + formatter.format(100*eval.precision(0)) + "%" +
				" - Recal: " + formatter.format(100*eval.recall(0)) + "%" +
				" - F1: " + formatter.format(100*eval.fMeasure(0)) + "%");

		System.out.println("True positive rate: " + formatter.format(100*eval.truePositiveRate(0)) + "%" + 
				" - True negative rate: "  + formatter.format(100*eval.trueNegativeRate(0)) + "%");
		System.out.println("Accuracy: " + formatter.format(100*((eval.truePositiveRate(0) + eval.trueNegativeRate(0)) / 2)) + "%");
		
		System.out.println("\nDone!");
	}
}
