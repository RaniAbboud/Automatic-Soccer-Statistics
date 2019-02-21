Final Classifiers/
	- Team_Classifier.py: the classifier which identifies the team with the ball for on-game frames
	- Frame_Type_Classifier.py: the classifier which identifies if the frame is on-game or off-game
	
	link for the weight files for the final classifiers:
		https://drive.google.com/open?id=13CRgiYysVOdFMu9hmD-UprSCMfl5UYod
	
	/data preparation/
		- make_image_directory.py: code for creating image directories (splitting the data into folders) - used for the Frame_Type_Classifier
		- make_image_directory_gamon.py: code for creating image directories (splitting the data into folders) - used for the Team_Classifier
	/demo generation/
		- generate_demo.py: code for generation the demos
	/transfer/
		- transfer_model.py: the transfer model
		- create_data_dir.py: code for creating image directories for the new data for transfer learning
		- testing_transfer.py: code for testing the transfer model
auxilary code/
	- accuracy.py: code for calculating on-game accuracy and more
/finding correlation between possession and _/
	- corr.py: code for calculating the correlation between ball possession and other important statistics of the match using a specific dataset
/labelling script/
	- label_images.py: our script that helps us label the data manually
/First Approach models [not used in final classifiers]/
	/our CNN model/
		- CNN_model2.py: our CNN model
	/VGG features 4096 model/
		- model4096.py: the model that uses VGG features (4096)
		/data preparation (extracting features)/
			- prepare4096.py: code for preparing the data and extracting featrues for the 4096 model
		/testing/
			- testing4096.py: code for testing the 4096 model
	/VGG features 73728 model/
		- model_smaller2.py: the 73728 model that uses extracted VGG features
		/data preparation (extracting features)/
			- prepare data.py: code for preparing the data and extracting featrues for the 73728 model
			- split_save_data.py: splits the data into train and validation
		/testing/
			- testing.py: code for testing the 73728 model
	/fine tuning VGG model/
		- model.py: fine tuning VGG model
		- model_without_other.py: this is when we were very close to our final classifier, here we are training only with on-game frames
		
	
