package application;

import java.net.URL;
import java.util.ResourceBundle;

import com.jfoenix.controls.JFXButton;
import com.jfoenix.controls.JFXCheckBox;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.stage.Stage;

public class SettingsController implements Initializable {

	@FXML
	private TextField txtLR;

	@FXML
	private TextField txtBS;

	@FXML
	private TextField txtPS;

	@FXML
	private TextField txtSeed;

	@FXML
	private TextField txtEpoch;

	@FXML
	private JFXCheckBox chkBackProp;

	@FXML
	private JFXCheckBox chkPretrain;

	@FXML
	private TextField txtWidth;

	@FXML
	private TextField txtHeight;

	@FXML
	private TextField txtClassCount;

	@FXML
	private TextField txtTrainingCount;

	@FXML
	private Label lblTraining;

	@FXML
	private Label isrgb;

	@FXML
	private Label lblWidth;

	@FXML
	private Label lblHeigth;

	@FXML
	private Label lblClassCount;

	@FXML
	private TextField txtTestingCount;
	
    @FXML
    private JFXButton idBtn;
	

    @FXML
    private JFXCheckBox chkRgb;

	@FXML
	private Label lblTesting;

	static int batcSize = 10, seed = 12345, epoch = 10, width = 28, height = 28, classcount = 10, train = 60000,
			test = 10000;
	static double learningRate = 0.0001, percentSplit = 0.65;
	static boolean backProp = true, pretrain = false,isRgb=false;

	@FXML
	void fonkOK(ActionEvent event) {
		if (txtLR.getText().equals(null)) {
			learningRate = 0.0001;
		} 
		else {
			learningRate = Double.parseDouble(txtLR.getText());
		}

		if (txtBS.getText().equals(null)) {
			batcSize = 10;
		} else {
			batcSize = Integer.parseInt(txtBS.getText());
		}
		if (txtEpoch.getText().equals(null)) {
			epoch = 10;
		} else {
			epoch = Integer.parseInt(txtEpoch.getText());
		}

		if (txtSeed.getText().equals(null)) {
			seed = 12345;
		} else {
			seed = Integer.parseInt(txtSeed.getText());
		}

		if (txtPS.getText().equals(null)) {
			percentSplit = 0.65;
		} else {
			percentSplit = Double.parseDouble(txtPS.getText());
		}

		if (txtWidth.getText().equals(null)) {
			width = 28;
		} else {
			width = Integer.parseInt(txtWidth.getText());
		}
		if (txtHeight.getText().equals(null)) {
			height = 28;
		} else {
			height = Integer.parseInt(txtHeight.getText());
		}
		if (txtClassCount.getText().equals(null)) {
			classcount = 10;
		} else {
			classcount = Integer.parseInt(txtClassCount.getText());
		}
		
		if (txtTestingCount.getText().equals(null)) {
			test = 10000;
		} else {
			test = Integer.parseInt(txtTestingCount.getText());
		}
		
		if (txtTrainingCount.getText().equals(null)) {
			train = 60000;
		}
		
		else {
			train = Integer.parseInt(txtTrainingCount.getText());
		}
		
		Stage stage = (Stage) idBtn.getScene().getWindow();
		stage.close();

	}

	@FXML
	void fonkchkBackProp(ActionEvent event) {
		if (chkBackProp.isSelected()) {
			backProp = true;
		} else {
			backProp = false;
		}

	}

	@FXML
	void fonkchkPretrain(ActionEvent event) {
		if (chkPretrain.isSelected()) {
			pretrain = true;
		} else {
			pretrain = false;
		}
	}
	
	@FXML
    void fonkchkRgb(ActionEvent event) {
		
		if (chkRgb.isSelected()) {
			isRgb = true;
		} else {
			isRgb = false;
		}
    }

	@Override
	public void initialize(URL arg0, ResourceBundle arg1) {

		if (SampleController.fileSecili == false) {
			lblClassCount.setVisible(true);
			lblHeigth.setVisible(true);
			lblTesting.setVisible(true);
			lblTraining.setVisible(true);
			lblWidth.setVisible(true);
			txtHeight.setVisible(true);
			txtWidth.setVisible(true);
			txtTestingCount.setVisible(true);
			txtTrainingCount.setVisible(true);
			txtClassCount.setVisible(true);
			chkRgb.setVisible(true);
			isrgb.setVisible(true);
			//txtPS.setDisable(true);
		} else {
			lblClassCount.setVisible(false);
			lblHeigth.setVisible(false);
			lblTesting.setVisible(false);
			lblTraining.setVisible(false);
			lblWidth.setVisible(false);
			txtHeight.setVisible(false);
			txtWidth.setVisible(false);
			txtTestingCount.setVisible(false);
			txtTrainingCount.setVisible(false);
			txtClassCount.setVisible(false);
			chkRgb.setVisible(false);
			isrgb.setVisible(false);
		}

		chkBackProp.setSelected(true);

	}

}
