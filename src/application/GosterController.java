package application;

import java.net.URL;
import java.util.ResourceBundle;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.TextArea;

public class GosterController implements Initializable {

    @FXML
    private TextArea txt;

	@Override
	public void initialize(URL arg0, ResourceBundle arg1) {
		txt.setText(SampleController.sonuc);
		
	}

}
