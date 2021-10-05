package application;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.ResourceBundle;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.jfoenix.controls.JFXComboBox;
import com.jfoenix.controls.JFXRadioButton;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleGroup;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SampleController implements Initializable {

	@FXML
	private JFXRadioButton radioFile;

	@FXML
	private ToggleGroup group;

	@FXML
	private JFXRadioButton radioFolder;

	@FXML
	private Button btnChoseFile;

	@FXML
	private Label lblChoseFile;

	@FXML
	private Label lblChoseHL;

	@FXML
	private JFXComboBox<String> cmbChoseHL;

	@FXML
	private Label hl1;

	@FXML
	private JFXComboBox<String> hl1Activation;

	@FXML
	private TextField hl1Neuron;

	@FXML
	private Label hl2;

	@FXML
	private JFXComboBox<String> hl2Activation;

	@FXML
	private TextField hl2Neuron;

	@FXML
	private Label hl3;

	@FXML
	private JFXComboBox<String> hl3Activation;

	@FXML
	private TextField hl3Neuron;

	@FXML
	private Label hl4;

	@FXML
	private JFXComboBox<String> hl4Activation;

	@FXML
	private TextField hl4Neuron;

	@FXML
	private Label hl5;

	@FXML
	private JFXComboBox<String> hl5Activation;

	@FXML
	private TextField hl5Neuron;

	@FXML
	private Label hl6;

	@FXML
	private JFXComboBox<String> hl6Activation;

	@FXML
	private TextField hl6Neuron;

	@FXML
	private Label hl7;

	@FXML
	private JFXComboBox<String> hl7Activation;

	@FXML
	private TextField hl7Neuron;

	@FXML
	private Label hlOutpu;

	@FXML
	private JFXComboBox<String> hlOutputActivation;

	@FXML
	private Button btnChoseFolder1;
	
	public static String sonuc;

	@FXML
	void fonkBtnChoseFolder(ActionEvent event) {

		DirectoryChooser dc = new DirectoryChooser();
		sd = dc.showDialog(Main.stage);
		System.out.println(sd.getAbsolutePath());
		lblChoseFile.setText("Dosya secildi");
		
		lblChoseHL.setVisible(true);
		cmbChoseHL.setVisible(true);
		
		
	}

	@FXML
	private JFXComboBox<String> lossFunction;

	ArrayList<String> listHLCount = new ArrayList<>();
	ArrayList<String> listActivationFonk = new ArrayList<>();
	ArrayList<String> listLossFonk = new ArrayList<>();
	FileChooser fileChooser = new FileChooser();
	File selectedFile;
	String secilen, fileName, getParentFileName, uzanti;
	int satirSayi = 0;
	@FXML
	private Label lblDosyaUzantisi;
	public static int FEATURES_COUNT, CLASSES_COUNT;
	String path = null;
	static File sd;
	static String secilenHLSayisi = null;

	@FXML
	void fonkBtnChoseFile(ActionEvent event) {
		selectedFile = fileChooser.showOpenDialog(Main.stage);
		fileChooser.setTitle("Select File");
		if (selectedFile != null) {

			try {
				fileName = selectedFile.getCanonicalPath();
			} catch (IOException e) {

				e.printStackTrace();
			}
			secilen = selectedFile.toString();
			getParentFileName = selectedFile.getParent();

			int ifade = secilen.lastIndexOf(".") + 1;
			uzanti = secilen.substring(ifade);
			path = secilen.substring(0, ifade);
			System.out.println("yol: " + path);
			if (uzanti.equals("csv")) {

				lblChoseFile.setText("CSV Dosyasi Secildi");
				System.out.println(secilen);
			}

		}
		try {
			DataSource source = new DataSource(secilen);
			Instances data = source.getDataSet();

			if (data.classIndex() == -1) {
				data.setClassIndex(data.numAttributes() - 1);
			}
			int datasayisi = data.numAttributes();
			CLASSES_COUNT = data.numAttributes() - 2;

			FEATURES_COUNT = datasayisi - 1;

		} catch (Exception e) {

			e.printStackTrace();
		}

		File f = new File(secilen);

		BufferedReader read = null;
		try {
			read = new BufferedReader(new FileReader(f));
			try {
				String satir = read.readLine();
				while (satir != null) {
					if (satir.length() > 0) {
						satirSayi++;
					}
					satir = read.readLine();
				}
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		System.out.println("satir sayisi: " + satirSayi);
		System.out.println("CLASSES_COUNT : " + CLASSES_COUNT);
		System.out.println("FEATURES_COUNT : " + FEATURES_COUNT);

		lblChoseHL.setVisible(true);
		cmbChoseHL.setVisible(true);

	}

	@FXML
	void fonkCmbChoseHL(ActionEvent event) {

		secilenHLSayisi = cmbChoseHL.getSelectionModel().getSelectedItem().toString().trim();
		if (secilenHLSayisi == "1") {
			hl1.setVisible(true);
			hl1Activation.setVisible(true);
			hl1Neuron.setVisible(true);

			hl2.setVisible(false);
			hl2Activation.setVisible(false);
			hl2Neuron.setVisible(false);

			hl3.setVisible(false);
			hl3Activation.setVisible(false);
			hl3Neuron.setVisible(false);

			hl4.setVisible(false);
			hl4Activation.setVisible(false);
			hl4Neuron.setVisible(false);

			hl5.setVisible(false);
			hl5Activation.setVisible(false);
			hl5Neuron.setVisible(false);

			hl6.setVisible(false);
			hl6Activation.setVisible(false);
			hl6Neuron.setVisible(false);

			hl7.setVisible(false);
			hl7Activation.setVisible(false);
			hl7Neuron.setVisible(false);

			hlOutpu.setVisible(true);
			hlOutputActivation.setVisible(true);
			lossFunction.setVisible(true);

		}
		if (secilenHLSayisi == "2") {
			hl1.setVisible(true);
			hl1Activation.setVisible(true);
			hl1Neuron.setVisible(true);

			hl2.setVisible(true);
			hl2Activation.setVisible(true);
			hl2Neuron.setVisible(true);

			hl3.setVisible(false);
			hl3Activation.setVisible(false);
			hl3Neuron.setVisible(false);

			hl4.setVisible(false);
			hl4Activation.setVisible(false);
			hl4Neuron.setVisible(false);

			hl5.setVisible(false);
			hl5Activation.setVisible(false);
			hl5Neuron.setVisible(false);

			hl6.setVisible(false);
			hl6Activation.setVisible(false);
			hl6Neuron.setVisible(false);

			hl7.setVisible(false);
			hl7Activation.setVisible(false);
			hl7Neuron.setVisible(false);

			hlOutpu.setVisible(true);
			hlOutputActivation.setVisible(true);
			lossFunction.setVisible(true);

		}

		if (secilenHLSayisi == "3") {
			hl1.setVisible(true);
			hl1Activation.setVisible(true);
			hl1Neuron.setVisible(true);

			hl2.setVisible(true);
			hl2Activation.setVisible(true);
			hl2Neuron.setVisible(true);

			hl3.setVisible(true);
			hl3Activation.setVisible(true);
			hl3Neuron.setVisible(true);

			hl4.setVisible(false);
			hl4Activation.setVisible(false);
			hl4Neuron.setVisible(false);

			hl5.setVisible(false);
			hl5Activation.setVisible(false);
			hl5Neuron.setVisible(false);

			hl6.setVisible(false);
			hl6Activation.setVisible(false);
			hl6Neuron.setVisible(false);

			hl7.setVisible(false);
			hl7Activation.setVisible(false);
			hl7Neuron.setVisible(false);

			hlOutpu.setVisible(true);
			hlOutputActivation.setVisible(true);
			lossFunction.setVisible(true);

		}

		if (secilenHLSayisi == "4") {
			hl1.setVisible(true);
			hl1Activation.setVisible(true);
			hl1Neuron.setVisible(true);

			hl2.setVisible(true);
			hl2Activation.setVisible(true);
			hl2Neuron.setVisible(true);

			hl3.setVisible(true);
			hl3Activation.setVisible(true);
			hl3Neuron.setVisible(true);

			hl4.setVisible(true);
			hl4Activation.setVisible(true);
			hl4Neuron.setVisible(true);

			hl5.setVisible(false);
			hl5Activation.setVisible(false);
			hl5Neuron.setVisible(false);

			hl6.setVisible(false);
			hl6Activation.setVisible(false);
			hl6Neuron.setVisible(false);

			hl7.setVisible(false);
			hl7Activation.setVisible(false);
			hl7Neuron.setVisible(false);

			hlOutpu.setVisible(true);
			hlOutputActivation.setVisible(true);
			lossFunction.setVisible(true);

		}

		if (secilenHLSayisi == "5") {
			hl1.setVisible(true);
			hl1Activation.setVisible(true);
			hl1Neuron.setVisible(true);

			hl2.setVisible(true);
			hl2Activation.setVisible(true);
			hl2Neuron.setVisible(true);

			hl3.setVisible(true);
			hl3Activation.setVisible(true);
			hl3Neuron.setVisible(true);

			hl4.setVisible(true);
			hl4Activation.setVisible(true);
			hl4Neuron.setVisible(true);

			hl5.setVisible(true);
			hl5Activation.setVisible(true);
			hl5Neuron.setVisible(true);

			hl6.setVisible(false);
			hl6Activation.setVisible(false);
			hl6Neuron.setVisible(false);

			hl7.setVisible(false);
			hl7Activation.setVisible(false);
			hl7Neuron.setVisible(false);

			hlOutpu.setVisible(true);
			hlOutputActivation.setVisible(true);
			lossFunction.setVisible(true);

		}

		if (secilenHLSayisi == "6") {
			hl1.setVisible(true);
			hl1Activation.setVisible(true);
			hl1Neuron.setVisible(true);

			hl2.setVisible(true);
			hl2Activation.setVisible(true);
			hl2Neuron.setVisible(true);

			hl3.setVisible(true);
			hl3Activation.setVisible(true);
			hl3Neuron.setVisible(true);

			hl4.setVisible(true);
			hl4Activation.setVisible(true);
			hl4Neuron.setVisible(true);

			hl5.setVisible(true);
			hl5Activation.setVisible(true);
			hl5Neuron.setVisible(true);

			hl6.setVisible(true);
			hl6Activation.setVisible(true);
			hl6Neuron.setVisible(true);

			hl7.setVisible(false);
			hl7Activation.setVisible(false);
			hl7Neuron.setVisible(false);

			hlOutpu.setVisible(true);
			hlOutputActivation.setVisible(true);
			lossFunction.setVisible(true);

		}

		if (secilenHLSayisi == "7") {
			hl1.setVisible(true);
			hl1Activation.setVisible(true);
			hl1Neuron.setVisible(true);

			hl2.setVisible(true);
			hl2Activation.setVisible(true);
			hl2Neuron.setVisible(true);

			hl3.setVisible(true);
			hl3Activation.setVisible(true);
			hl3Neuron.setVisible(true);

			hl4.setVisible(true);
			hl4Activation.setVisible(true);
			hl4Neuron.setVisible(true);

			hl5.setVisible(true);
			hl5Activation.setVisible(true);
			hl5Neuron.setVisible(true);

			hl6.setVisible(true);
			hl6Activation.setVisible(true);
			hl6Neuron.setVisible(true);

			hl7.setVisible(true);
			hl7Activation.setVisible(true);
			hl7Neuron.setVisible(true);

			hlOutpu.setVisible(true);
			hlOutputActivation.setVisible(true);
			lossFunction.setVisible(true);

		}

	}

	String sec1, sec2, sec3, sec4, sec5, sec6, sec7, secOut, secLoss;
	MultiLayerNetwork model;
	LossFunction ls;
	int nIn = 0, nOut;
	 long t0 = System.currentTimeMillis();
	
	

	static boolean fileSecili = true;

	@FXML
	void fonkRadioFile(ActionEvent event) {
		if (radioFile.isSelected()) {
			btnChoseFile.setVisible(true);
			lblChoseFile.setVisible(true);
			fileSecili = true;
			btnChoseFolder1.setVisible(false);
		}

	}

	@FXML
	void fonkRadioFolder(ActionEvent event) {
		if (radioFolder.isSelected()) {
			btnChoseFile.setVisible(false);
			lblChoseFile.setVisible(true);
			fileSecili = false;
			
			btnChoseFolder1.setVisible(true);
		}

	}

	@FXML
	void fonkSave(ActionEvent event) {

	}

	@FXML
	void fonkSettings(ActionEvent event) {
		try {
			URL url = new File("C:\\Users\\baran\\eclipse-workspace\\DerinOgrenme\\src\\application\\Settings.fxml").toURI().toURL();
			Parent fxml = FXMLLoader.load(url);
			Stage stage = new Stage();
			Scene scene = new Scene(fxml);
			stage.setTitle("Settings");
			stage.setScene(scene);
			stage.show();
		} catch (IOException e) {

			System.out.println("Settings" + " Hata :" + e.getMessage());
		}
	}

	@FXML
	void fonkStart(ActionEvent event) {
		
		if(fileSecili==true) {
			
		
		String data = secilen;

		try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
			recordReader.initialize(new FileSplit(new File(data)));

			DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, satirSayi, FEATURES_COUNT,
					CLASSES_COUNT);
			DataSet allData = iterator.next();
			allData.shuffle(123);

			DataNormalization normalizer = new NormalizerStandardize();
			normalizer.fit(allData);
			normalizer.transform(allData);

			SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
			DataSet trainingData = testAndTrain.getTrain();
			DataSet testingData = testAndTrain.getTest();

			csvDosya(trainingData, testingData);
			
			try {
				URL url = new File("C:\\Users\\baran\\eclipse-workspace\\Bitir\\src\\application\\Goster.fxml").toURI().toURL();
				Parent fxml = FXMLLoader.load(url);
				Stage stage = new Stage();
				Scene scene = new Scene(fxml);
				stage.setTitle("Settings");
				stage.setScene(scene);
				stage.show();
			} catch (IOException e) {

				System.out.println("Settings" + " Hata :" + e.getMessage());
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
		}
		
		else {
			  t0 = System.currentTimeMillis();
		        //System.out.print(RESOURCES_FOLDER_PATH + "/training");
		        DataSetIterator dataSetIterator;
				try {
					dataSetIterator = getDataSetIterator(sd.getAbsolutePath() + "/training", SettingsController.train);
					buildModel(dataSetIterator);
					
					
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				
				

		        
		}
		
	
		
		
	}

	@FXML
	void fonkhl1Activation(ActionEvent event) {
		sec1 = hl1Activation.getSelectionModel().getSelectedItem().toString().trim();
		System.out.println("sec1: " + sec1);

	}

	@FXML
	void fonkhl2Activation(ActionEvent event) {
		sec2 = hl2Activation.getSelectionModel().getSelectedItem().toString().trim();
		System.out.println("sec2: " + sec2);
	}

	@FXML
	void fonkhl3Activation(ActionEvent event) {
		sec3 = hl3Activation.getSelectionModel().getSelectedItem().toString().trim();
		System.out.println("sec3: " + sec3);
	}

	@FXML
	void fonkhl4Activation(ActionEvent event) {
		sec4 = hl4Activation.getSelectionModel().getSelectedItem().toString().trim();
		System.out.println("sec4: " + sec4);
	}

	@FXML
	void fonkhl5Activation(ActionEvent event) {
		sec5 = hl5Activation.getSelectionModel().getSelectedItem().toString().trim();
		System.out.println("sec5: " + sec5);
	}

	@FXML
	void fonkhl6Activation(ActionEvent event) {
		sec6 = hl6Activation.getSelectionModel().getSelectedItem().toString().trim();
		System.out.println("sec6: " + sec6);
	}

	@FXML
	void fonkhl7Activation(ActionEvent event) {
		sec7 = hl7Activation.getSelectionModel().getSelectedItem().toString().trim();
		System.out.println("sec7: " + sec7);
	}

	@FXML
	void fonkhlOutputActivation(ActionEvent event) {
		secOut = hlOutputActivation.getSelectionModel().getSelectedItem().toString().trim();
		System.out.println("secOut: " + secOut);
	}

	@FXML
	void fonklossFunction(ActionEvent event) {
		secLoss = lossFunction.getSelectionModel().getSelectedItem().toString().trim();
		System.out.println("secLoss: " + secLoss);

	}

	@Override
	public void initialize(URL arg0, ResourceBundle arg1) {
		// TODO Auto-generated method stub
		cmbDoldur();
		System.out.println(hlFonk("TANH"));

	}

	public void cmbDoldur() {
		listHLCount.add("1");
		listHLCount.add("2");
		listHLCount.add("3");
		listHLCount.add("4");
		listHLCount.add("5");
		listHLCount.add("6");
		listHLCount.add("7");

		listActivationFonk.add("RELU");
		listActivationFonk.add("SOFTMAX");
		listActivationFonk.add("TANH");
		listActivationFonk.add("SIGMOID");
		listActivationFonk.add("CUBE");
		listActivationFonk.add("IDENTITY");
		listActivationFonk.add("ELU");
		listActivationFonk.add("HARDSIGMOID");
		listActivationFonk.add("SOFTPLUS");
		listActivationFonk.add("SELU");
		listActivationFonk.add("SWISH");

		listLossFonk.add("POISSON");
		listLossFonk.add("MEAN_ABSOLUTE_ERROR");
		listLossFonk.add("MSE");
		listLossFonk.add("NEGATIVELOGLIKELIHOOD");
		listLossFonk.add("SQUARED_LOSS");

		cmbChoseHL.getItems().addAll(listHLCount);
		hl1Activation.getItems().addAll(listActivationFonk);
		hl2Activation.getItems().addAll(listActivationFonk);
		hl3Activation.getItems().addAll(listActivationFonk);
		hl4Activation.getItems().addAll(listActivationFonk);
		hl5Activation.getItems().addAll(listActivationFonk);
		hl6Activation.getItems().addAll(listActivationFonk);
		hl7Activation.getItems().addAll(listActivationFonk);
		hlOutputActivation.getItems().addAll(listActivationFonk);
		lossFunction.getItems().addAll(listLossFonk);

	}

	public Activation hlFonk(String x) {
		Activation act = null;
		if (x.equals("RELU")) {
			act = Activation.RELU;
		} else if (x.equals("SOFTMAX")) {
			act = Activation.SOFTMAX;
		} else if (x.equals("TANH")) {
			act = Activation.TANH;
		} else if (x.equals("SIGMOID")) {
			act = Activation.SIGMOID;
		} else if (x.equals("CUBE")) {
			act = Activation.CUBE;
		} else if (x.equals("IDENTITY")) {
			act = Activation.IDENTITY;
		} else if (x.equals("ELU")) {
			act = Activation.ELU;
		} else if (x.equals("HARDSIGMOID")) {
			act = Activation.HARDSIGMOID;
		} else if (x.equals("SOFTPLUS")) {
			act = Activation.SOFTPLUS;
		} else if (x.equals("SELU")) {
			act = Activation.SELU;
		} else if (x.equals("SWISH")) {
			act = Activation.SWISH;
		} else {

			Alert mesaj = new Alert(AlertType.ERROR);
			mesaj.setHeaderText("Please make your chose!!");
			mesaj.show();
		}
		System.out.println(act);

		return act;
	}

	public void csvDosya(DataSet trainingData, DataSet testData) {
		System.out.println("secilen hdden sayi: " + secilenHLSayisi);

		if (secilenHLSayisi.equals("1")) {
			model = new MultiLayerNetwork(bir());
		} else if (secilenHLSayisi.equals("2")) {
			model = new MultiLayerNetwork(iki());
		} else if (secilenHLSayisi.equals("3")) {
			model = new MultiLayerNetwork(uc());
		} else if (secilenHLSayisi.equals("4")) {
			model = new MultiLayerNetwork(dort());
		} else if (secilenHLSayisi.equals("5")) {
			model = new MultiLayerNetwork(bes());
		} else if (secilenHLSayisi.equals("6")) {
			model = new MultiLayerNetwork(alti());
		} else if (secilenHLSayisi.equals("7")) {
			model = new MultiLayerNetwork(yedi());
		}
		
		
		model.init();
		model.fit(trainingData); // eðitiyor
		model.setEpochCount(SettingsController.epoch);
		/*
		 * try { model.save(new File("C:\\\\Users\\\\baran\\\\Downloads\\\\iriss.h5"));
		 * } catch (IOException e) { // TODO Auto-generated catch block
		 * e.printStackTrace(); }
		 */

		INDArray output = model.output(testData.getFeatureMatrix());

		Evaluation eval = new Evaluation(3);
		eval.eval(testData.getLabels(), output);
		System.out.println(eval.stats());
		sonuc=eval.stats();

	}

	
	
	
	
	
	
	
	
	
	
	public MultiLayerConfiguration bir() {
		if(fileSecili==false) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width;
			CLASSES_COUNT=SettingsController.classcount;
		}
		if(fileSecili==false && SettingsController.isRgb==true) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width*3;
			CLASSES_COUNT=SettingsController.classcount;
		}
		
		MultiLayerConfiguration configuration = null;
		// secLoss

		if (secLoss.equals("MSE")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl1Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();
		} else if (secLoss.equals("POISSON")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new OutputLayer.Builder(LossFunctions.LossFunction.POISSON).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl1Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();
		} else if (secLoss.equals("MEAN_ABSOLUTE_ERROR")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl1Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();
		} else if (secLoss.equals("NEGATIVELOGLIKELIHOOD")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl1Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();
		} else if (secLoss.equals("SQUARED_LOSS")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl1Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		}

		return configuration;

	}

	public MultiLayerConfiguration iki() {
		MultiLayerConfiguration configuration = null;
		
		if(fileSecili==false) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width;
			CLASSES_COUNT=SettingsController.classcount;
		}
		// secLoss
		if(fileSecili==false && SettingsController.isRgb==true) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width*3;
			CLASSES_COUNT=SettingsController.classcount;
		}
		if (secLoss.equals("MSE")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl2Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();
		}

		else if (secLoss.equals("POISSON")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new OutputLayer.Builder(LossFunctions.LossFunction.POISSON).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl2Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("MEAN_ABSOLUTE_ERROR")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl2Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("NEGATIVELOGLIKELIHOOD")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl2Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("SQUARED_LOSS")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl2Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		}

		return configuration;

	}

	public MultiLayerConfiguration uc() {
		MultiLayerConfiguration configuration = null;
		// secLoss
		
		if(fileSecili==false) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width;
			CLASSES_COUNT=SettingsController.classcount;
		}
		if(fileSecili==false && SettingsController.isRgb==true) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width*3;
			CLASSES_COUNT=SettingsController.classcount;
		}

		if (secLoss.equals("MSE")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl3Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();
		}

		else if (secLoss.equals("POISSON")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new OutputLayer.Builder(LossFunctions.LossFunction.POISSON).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl3Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("MEAN_ABSOLUTE_ERROR")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl3Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("NEGATIVELOGLIKELIHOOD")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl3Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("SQUARED_LOSS")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl3Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		}

		return configuration;

	}

	public MultiLayerConfiguration dort() {
		MultiLayerConfiguration configuration = null;
		// secLoss
		
		if(fileSecili==false) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width;
			CLASSES_COUNT=SettingsController.classcount;
		}
		if(fileSecili==false && SettingsController.isRgb==true) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width*3;
			CLASSES_COUNT=SettingsController.classcount;
		}

		if (secLoss.equals("MSE")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl4Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();
		}

		else if (secLoss.equals("POISSON")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new OutputLayer.Builder(LossFunctions.LossFunction.POISSON).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl4Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("MEAN_ABSOLUTE_ERROR")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl4Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("NEGATIVELOGLIKELIHOOD")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl4Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("SQUARED_LOSS")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl4Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		}

		return configuration;

	}

	public MultiLayerConfiguration bes() {
		MultiLayerConfiguration configuration = null;
		// secLoss
		
		if(fileSecili==false) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width;
			CLASSES_COUNT=SettingsController.classcount;
		}
		if(fileSecili==false && SettingsController.isRgb==true) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width*3;
			CLASSES_COUNT=SettingsController.classcount;
		}

		if (secLoss.equals("MSE")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl5Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();
		}

		else if (secLoss.equals("POISSON")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new OutputLayer.Builder(LossFunctions.LossFunction.POISSON).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl5Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("MEAN_ABSOLUTE_ERROR")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl5Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("NEGATIVELOGLIKELIHOOD")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl5Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("SQUARED_LOSS")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl5Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		}

		return configuration;

	}

	public MultiLayerConfiguration alti() {
		MultiLayerConfiguration configuration = null;
		// secLoss
		
		if(fileSecili==false) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width;
			CLASSES_COUNT=SettingsController.classcount;
		}
		if(fileSecili==false && SettingsController.isRgb==true) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width*3;
			CLASSES_COUNT=SettingsController.classcount;
		}

		if (secLoss.equals("MSE")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl5Neuron.getText())).activation(hlFonk(sec6))
									.nOut(Integer.parseInt(hl6Neuron.getText())).build())
					.layer(6,
							new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl6Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();
		}

		else if (secLoss.equals("POISSON")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl5Neuron.getText())).activation(hlFonk(sec6))
									.nOut(Integer.parseInt(hl6Neuron.getText())).build())
					.layer(6,
							new OutputLayer.Builder(LossFunctions.LossFunction.POISSON).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl6Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("MEAN_ABSOLUTE_ERROR")) {

			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl5Neuron.getText())).activation(hlFonk(sec6))
									.nOut(Integer.parseInt(hl6Neuron.getText())).build())
					.layer(6,
							new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl6Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("NEGATIVELOGLIKELIHOOD")) {

			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl5Neuron.getText())).activation(hlFonk(sec6))
									.nOut(Integer.parseInt(hl6Neuron.getText())).build())
					.layer(6,
							new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl6Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("SQUARED_LOSS")) {

			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl5Neuron.getText())).activation(hlFonk(sec6))
									.nOut(Integer.parseInt(hl6Neuron.getText())).build())
					.layer(6,
							new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl6Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		}

		return configuration;

	}

	public MultiLayerConfiguration yedi() {
		MultiLayerConfiguration configuration = null;
		// secLoss
		
		if(fileSecili==false) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width;
			CLASSES_COUNT=SettingsController.classcount;
		}
		if(fileSecili==false && SettingsController.isRgb==true) {
			FEATURES_COUNT=SettingsController.height*SettingsController.width*3;
			CLASSES_COUNT=SettingsController.classcount;
		}

		if (secLoss.equals("MSE")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl5Neuron.getText())).activation(hlFonk(sec6))
									.nOut(Integer.parseInt(hl6Neuron.getText())).build())
					.layer(6,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl6Neuron.getText())).activation(hlFonk(sec7))
									.nOut(Integer.parseInt(hl7Neuron.getText())).build())
					.layer(7,
							new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl7Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();
		}

		else if (secLoss.equals("POISSON")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl5Neuron.getText())).activation(hlFonk(sec6))
									.nOut(Integer.parseInt(hl6Neuron.getText())).build())
					.layer(6,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl6Neuron.getText())).activation(hlFonk(sec7))
									.nOut(Integer.parseInt(hl7Neuron.getText())).build())
					.layer(7,
							new OutputLayer.Builder(LossFunctions.LossFunction.POISSON).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl7Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("MEAN_ABSOLUTE_ERROR")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl5Neuron.getText())).activation(hlFonk(sec6))
									.nOut(Integer.parseInt(hl6Neuron.getText())).build())
					.layer(6,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl6Neuron.getText())).activation(hlFonk(sec7))
									.nOut(Integer.parseInt(hl7Neuron.getText())).build())
					.layer(7,
							new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl7Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("NEGATIVELOGLIKELIHOOD")) {
			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl5Neuron.getText())).activation(hlFonk(sec6))
									.nOut(Integer.parseInt(hl6Neuron.getText())).build())
					.layer(6,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl6Neuron.getText())).activation(hlFonk(sec7))
									.nOut(Integer.parseInt(hl7Neuron.getText())).build())
					.layer(7,
							new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
									.activation(hlFonk(secOut)).nIn(Integer.parseInt(hl7Neuron.getText()))
									.nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		} else if (secLoss.equals("SQUARED_LOSS")) {

			configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
					.updater(new Nesterovs(0.1, 0.9)).l2(SettingsController.learningRate).list()
					.layer(0,
							new DenseLayer.Builder().nIn(FEATURES_COUNT).activation(hlFonk(sec1))
									.nOut(Integer.parseInt(hl1Neuron.getText())).build())
					.layer(1,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl1Neuron.getText())).activation(hlFonk(sec2))
									.nOut(Integer.parseInt(hl2Neuron.getText())).build())
					.layer(2,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl2Neuron.getText())).activation(hlFonk(sec3))
									.nOut(Integer.parseInt(hl3Neuron.getText())).build())
					.layer(3,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl3Neuron.getText())).activation(hlFonk(sec4))
									.nOut(Integer.parseInt(hl4Neuron.getText())).build())
					.layer(4,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl4Neuron.getText())).activation(hlFonk(sec5))
									.nOut(Integer.parseInt(hl5Neuron.getText())).build())
					.layer(5,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl5Neuron.getText())).activation(hlFonk(sec6))
									.nOut(Integer.parseInt(hl6Neuron.getText())).build())
					.layer(6,
							new DenseLayer.Builder().nIn(Integer.parseInt(hl6Neuron.getText())).activation(hlFonk(sec7))
									.nOut(Integer.parseInt(hl7Neuron.getText())).build())
					.layer(7,
							new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(hlFonk(secOut))
									.nIn(Integer.parseInt(hl7Neuron.getText())).nOut(CLASSES_COUNT).build())
					.backprop(SettingsController.backProp).pretrain(SettingsController.pretrain).build();

		}

		return configuration;

	}

	
	public void buildModel(DataSetIterator dsi) throws IOException {

        int rngSeed = 123;
        int nEpochs = 2;

        System.out.print("Build Model...");
  
        if (secilenHLSayisi.equals("1")) {
			model = new MultiLayerNetwork(bir());
		} else if (secilenHLSayisi.equals("2")) {
			model = new MultiLayerNetwork(iki());
		} else if (secilenHLSayisi.equals("3")) {
			model = new MultiLayerNetwork(uc());
		} else if (secilenHLSayisi.equals("4")) {
			model = new MultiLayerNetwork(dort());
		} else if (secilenHLSayisi.equals("5")) {
			model = new MultiLayerNetwork(bes());
		} else if (secilenHLSayisi.equals("6")) {
			model = new MultiLayerNetwork(alti());
		} else if (secilenHLSayisi.equals("7")) {
			model = new MultiLayerNetwork(yedi());
		}
        
        model.init();
        //Print score every 500 interaction
        model.setListeners(new ScoreIterationListener(500));

        System.out.print("Train Model...");
        model.fit(dsi);

        //Evaluation
        DataSetIterator testDsi = getDataSetIterator(sd.getAbsoluteFile()+"/testing", SettingsController.test);
        System.out.print("Evaluating Model...");
        Evaluation eval = model.evaluate(testDsi);
        System.out.print(eval.stats());
        sonuc=eval.stats();

        long t1 = System.currentTimeMillis();
        double t = (double)(t1-t0)/1000.0;
        System.out.print("\n\nTotal time: "+t+" seconds");
    }
	
	
    private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {
        try {
            File folder = new File(folderPath);
            File[] digitFolders = folder.listFiles();
            NativeImageLoader nativeImageLoader ;
            if(SettingsController.isRgb==true) {
            	nativeImageLoader = new NativeImageLoader(SettingsController.height, SettingsController.width,3); //28x28
            }
            else {
            	nativeImageLoader = new NativeImageLoader(SettingsController.height, SettingsController.width);
            }
              //28x28
            
            
            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0,1); //translate image into seq of 0..1 input values
            INDArray input;
            if(SettingsController.isRgb==true) {
            	 input = Nd4j.create(new int[]{nSamples, SettingsController.height*SettingsController.width*3}); //28x28
            }
            else {
            	input = Nd4j.create(new int[]{nSamples, SettingsController.height*SettingsController.width});
            }
            
            INDArray output = Nd4j.create(new int[]{nSamples, SettingsController.classcount});

            int n = 0;
            //scan all 0 to 9 digit subfolders
            for (File digitFolder: digitFolders) {
                int labelDigit = Integer.parseInt(digitFolder.getName());
                File[] imageFiles = digitFolder.listFiles();

                for (File imgFile : imageFiles) {
                    INDArray img = nativeImageLoader.asRowVector(imgFile);
                    scaler.transform(img);
                    input.putRow(n, img);
                    output.put(n, labelDigit, 1.0);
                    n++;
                }
            }//End of For-loop

            //Joining input and output matrices into a dataset
            DataSet dataSet = new DataSet(input, output);
            //Convert the dataset into a list
            List<DataSet> listDataSet = dataSet.asList();
            //Shuffle content of list randomly
            Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));
            int batchSize = 10;

            //Build and return a dataset iterator
            DataSetIterator dsi = new ListDataSetIterator<DataSet>(listDataSet, batchSize);
            return dsi;
        } catch (Exception e) {
            System.out.println(e.getLocalizedMessage());
            return null;
        }
    } 
}