# TaiyoAI_Domestic_Molding
This repository contains code to preprocess data for building Domestic Molding Prediction  machine learning models.

## Readme for Data Processing code

### Introduction
This repository contains code to preprocess data for building Domestic Molding Prediction
 machine learning models. The code includes three classes in the `data/data_preprocessing.py` file `BaseDataProcessor`, `DataPreProcessor` and `DataPostProcessor`, that can be used to clean, fill, and process data for use in building models. The repository also contains a `model.py` which contains     `BaseModelBackBone` , `LazyPredictBackBone`, `LazyPredictRegressionBackBone`, `LazyPredictClassificationBackBone` and `LSTMBackBone` that can be used to build the model, Lastly the jupyter notebook `exploratory_data_analysis.ipynb` contains a detailed step by step walk through on how to use the various conponents to train a model.

### Dependencies
You can find all the dependencies in the following `requirements.txt` file.

## Installation

### Class BaseDataProcessor

This class is the base class that includes functions to load, clean and check the data.

load_data: Loads the data from a csv file and saves it to the instance variable data.
get_base_data: Returns the loaded data.
get_all_columns: Returns the list of all columns in the data.
get_all_features: Returns a list of all features in the data, excluding any features specified to be excluded from export.
get_target_feature: Returns the target feature.
clean_column_names: Cleans the column names by replacing spaces, ":", "(", and ")" with "_" and converting the names to lower case. The cleaned names are then applied to the data.
check_columns_exist: Checks if all columns specified in all_columns exist in the loaded data.
init_data: Calls the load_data, clean_column_names, and check_columns_exist functions.
save_data: Saves the processed data to a csv file.
Class DataPreProcessor
This class inherits from the BaseDataProcessor class and adds additional functions to preprocess the data.

fill_empty_spaces_with_NaN: Replaces empty spaces in the data with NaN.
back_to_float: Converts int columns to float.
fill_empty_spaces_with_values: Fills missing values with values calculated using the MICE imputer.
encode_categorical_data: Encodes categorical data using one hot encoding.
normalize_data: Normalizes the data using the min-max scaler.
pipeline_config_normaliser: A pipeline that performs normalization on the data.

Usage
python

`

    import pandas as pd
    from DataProcessing import BaseDataProcessor, DataPreProcessor

    data_file_path = "data.csv"
    all_columns = ["column_1", "column_2", "column_3", ...]
    target_feature = "target_column"
    exclude_feature_from_export = ["column_to_exclude", ...]

    data_processor = DataPreProcessor(data_file_path, all_columns, target_feature, exclude_feature_from_export)
    data = data_processor.get_base_data()
    data_processor.pipeline_config_normaliser.fit_transform(data)
    data_processor.save_data(data, "processed_data.csv")
`

The code above is for two classes, DataPreProcessor and DataPostProcessor, which are used for data pre-processing and post-processing respectively. The DataPreProcessor class is a parent class that performs the following operations:

Processes the data, splitting it into train, validation, and test sets if specified.
Fits a data pipeline to the data, including operations such as filling NaN values and scaling.
Saves the data pipeline to disk.
The DataPostProcessor class is a child class of DataPreProcessor and performs the following operation:

Loads a pre-trained data pipeline from disk.
Transforms the input data using the loaded pipeline.
Both classes take various parameters for data processing, including the input data, split percentages for train/validation/test sets, and the paths for saving the pipeline and split data. The code also includes TODO notes for further development of classes to perform data transformations.

BaseModelBackBone Class
The BaseModelBackBone is an abstract class that defines the basic structure of a model backbone. It acts as a blueprint for the model backbones that are derived from it. The class has the following methods:

__init__: This is the constructor method that is called when a new instance of the class is created. It takes no arguments and has no implementation.

forward: This method takes any number of arguments and passes them to the child class. It raises a NotImplementedError if the method is not overridden in the child class.

predict: This method takes any number of arguments and passes them to the child class. It raises a NotImplementedError if the method is not overridden in the child class.

train: This method takes any number of arguments and passes them to the child class. It raises a NotImplementedError if the method is not overridden in the child class.

evaluate: This method takes any number of arguments and passes them to the child class. It raises a NotImplementedError if the method is not overridden in the child class.

get_evaluation: This method returns the result of the evaluate method.

save_model: This method takes any number of arguments and passes them to the child class. It raises a NotImplementedError if the method is not overridden in the child class.

load_model: This method takes any number of arguments and passes them to the child class. It raises a NotImplementedError if the method is not overridden in the child class.

save_output: This method takes any number of arguments and passes them to the child class. It raises a NotImplementedError if the method is not overridden in the child class.

LazyPredictBackBone Class
The LazyPredictBackBone class is a derived class from BaseModelBackBone class. The class takes an instance of a lazypred_instance and saves it in the lazyobject property of the class. The class has the following methods:

train: This method trains the model using the lazyobject instance and saves the models and predictions in the properties of the class.

evaluate: This method returns the predictions property of the class.

LazyPredictRegressionBackBone Class
The LazyPredictRegressionBackBone class is a derived class from LazyPredictBackBone class. The class creates an instance of LazyRegressor and passes it to the lazypred_instance argument in the constructor of the parent class.

LazyPredictClassificationBackBone Class
The LazyPredictClassificationBackBone class is a derived class from LazyPredictBackBone class. The class creates an instance of LazyClassifier and passes it to the lazypred_instance argument in the constructor of the parent class.

LSTMBackBone Class
The LSTMBackBone class is a subclass of nn.Module from the PyTorch library. The class creates a Long Short-Term Memory (LSTM) model with an input dimension of input_dim, a

Conclusion
This code provides a flexible and convenient way to process and preprocess data for building machine learning models. The code can be easily modified or extended to meet the specific needs of a project.