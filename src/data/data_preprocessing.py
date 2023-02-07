import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import argparse

class OneHotEncoderColumns(BaseEstimator, TransformerMixin):
    """
    A class used to preprocess the data by one-hot encoding specified features.

    Parameters:
        all_feat (list, optional): List of all features to be considered.
            Default: None
        feat_to_dummy (list, optional): List of features to be one-hot encoded.
            Default: None

    Attributes:
        feat_to_dummy (list): List of features to be one-hot encoded.
        all_feat (list): List of all features to be considered.
        one_hot_encoder (OneHotEncoder): The instance of OneHotEncoder used to fit and transform the data.
    """

    def __init__(self, all_feat=None,
                 feat_to_dummy=None):
        self.feat_to_dummy = feat_to_dummy
        self.all_feat = all_feat

    def fit(self, X, y=None):
        """
        Fit the OneHotEncoder to the data.
        """
        return self #do nothing

    def transform(self, X):
        """
        One-hot encode the specified features.

        Parameters:
            X (pandas.DataFrame): The input data.

        Returns:
            pandas.DataFrame: The one-hot encoded data.
        """
        try:
            X = X[self.all_feat]
        except KeyError as exp:
            raise KeyError("Error: One or more columns specified in all_feat not found in input data") from exp

        X = self.one_hot_encoder.transform(X)
        X = pd.DataFrame(X.toarray(), columns=self.one_hot_encoder.get_feature_names(self.feat_to_dummy))
        return X
    
class BaseDataProcessor:
    """
    This class is used for processing the data, it takes file_path, all_columns, target_feature and 
    exclude_feature_from_export as inputs and performs several operations on the data like cleaning 
    column names, filling empty spaces with NaN, converting int columns to float and rounding values.

    Args:
        file_path (str): The file path of the data.
        all_columns (List[str]): The list of all columns in the data.
        target_feature (str): The target feature for which the model is being built.
        exclude_feature_from_export (List[str]): The list of features that should not be exported.
        pipeline_config (Union[None, sklearn.pipeline.Pipeline]): The pipeline configuration.
    """

    def __init__(self, file_path, all_columns=["Date", "Spot/Export Blow Molding", "Spot, Domestic", 
                                            "WTISPLC", "MCOILBRENTEU", "GASREGM", "IMPCH", "EXPCH", 
                                            "PRUBBUSDM", "WPUFD4111", "PCU325211325211", "PCU32611332611301", 
                                            "WPU0915021625","PCU3252132521", "MHHNGSP", "WPU072205011", 
                                            "PCU32611132611115", "PCU32611332611301","PCU32611132611112", 
                                            "Producer Price Index by Industry: Plastics Material and Resins Manufacturing: Thermoplastic Resins and Plastics Materials ",
                                            "Australia _export", "Canada_export", "Saudi_export", "Usa_export", 
                                            "India_export", "Russia_export", "South_Africa_export", "Turkey	Brazil", 
                                            "France_export", "Germeny_export", "United Kingdome_export	China_export",	
                                            "Australia _import", "Canada_import","Saudi_import","Usa_import","India_import", 
                                            "Russia_import", "South_Africa_import", "Turkey_import","Brazil_import", 
                                            "France_import", "Germeny_import", "United Kingdome_import", 
                                            "China_import", "Japan_import", "South_korea_import"], 
                 target_feature="Domestic Market (Contract) Blow Molding, Low",
                 exclude_feature_from_export=["Date"]):
        
        assert isinstance(file_path, str), "file_path should be a string"
        assert isinstance(all_columns, list), "all_columns should be a list of strings"
        assert isinstance(target_feature, str), "target_feature should be a string"
        assert isinstance(exclude_feature_from_export, list) or exclude_feature_from_export is None, "exclude_feature_from_export should be a list or None"

        self.file_path = file_path
        self.all_columns = all_columns
        self.target_feature = target_feature
        self.exclude_feature_from_export = exclude_feature_from_export
        self.data = None
        
        self.init_data()

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def clean_column_names(self):
        """
        This function cleans the column names of the data by replacing spaces, ":", "(", and ")" in the column names with "_".
        The cleaned column names are then saved back to the instance variables `all_columns`, `target_feature`, and `exclude_feature_from_export`.
        The cleaned column names are also applied to the data inplace.

        Returns:
            None
        """
        def clean_column_name(col_name):
            """
            Helper function to clean a single column name.
            """
            col_name = col_name.strip().replace(" ", "_").replace(":", "_").replace("(", "_").replace(")", "_")
            return col_name

        self.all_columns = [clean_column_name(col) for col in self.all_columns]
        self.target_feature = clean_column_name(self.target_feature)
        self.exclude_feature_from_export = [clean_column_name(col) for col in self.exclude_feature_from_export]

        self.all_columns = list(set(self.all_columns + [self.target_feature] + self.exclude_feature_from_export))

        self.data.rename(columns=lambda x: clean_column_name(x), inplace=True)



    def check_columns_exist(self):
        for col in self.all_columns:
            assert col in self.data.columns, f"Column {col} does not exist in the data."

    def init_data(self):
        self.load_data()
        self.clean_column_names()
        self.check_columns_exist()

    def save_data(self, data, file_path):
        """
        Saves the processed data to a csv file.

        Parameters:
        data (pandas.DataFrame): The data to be saved.
        file_path (str): The file path to save the data to.

        Returns:
        None
        """
        data = data.drop(columns=self.exclude_feature_from_export)
        data.to_csv(file_path, index=False)
class DataPreProcessor(BaseDataProcessor):
    def __init__(self, file_path, all_columns=["Date", "Spot/Export Blow Molding", "Spot, Domestic", 
                                            "WTISPLC", "MCOILBRENTEU", "GASREGM", "IMPCH", "EXPCH", 
                                            "PRUBBUSDM", "WPUFD4111", "PCU325211325211", "PCU32611332611301", 
                                            "WPU0915021625","PCU3252132521", "MHHNGSP", "WPU072205011", 
                                            "PCU32611132611115", "PCU32611332611301","PCU32611132611112", 
                                            "Producer Price Index by Industry: Plastics Material and Resins Manufacturing: Thermoplastic Resins and Plastics Materials ",
                                            "Australia _export", "Canada_export", "Saudi_export", "Usa_export", 
                                            "India_export", "Russia_export", "South_Africa_export", "Turkey	Brazil", 
                                            "France_export", "Germeny_export", "United Kingdome_export	China_export",	
                                            "Australia _import", "Canada_import","Saudi_import","Usa_import","India_import", 
                                            "Russia_import", "South_Africa_import", "Turkey_import","Brazil_import", 
                                            "France_import", "Germeny_import", "United Kingdome_import", 
                                            "China_import", "Japan_import", "South_korea_import"], 
                 target_feature="Domestic Market (Contract) Blow Molding, Low",
                 exclude_feature_from_export=["Date"], pipeline_config=None):
        super().__init__(file_path, all_columns, target_feature, exclude_feature_from_export)
        assert isinstance(pipeline_config, Pipeline) or pipeline_config is None, "pipeline_config should be a Pipeline object or None"
        
        self.pipeline_config = Pipeline([
                        ('fill_missing_with_NaN', Fill_Empty_Spaces_With_NaN(all_features=self.get_features(),find_in=self.get_missing_int_features()+self.find_missing_str,with_='NaN')),
                        ('int_column_to_float', Back_To_Float(all_features=self.get_features(), to_encode=self.get_missing_int_features())),
                        ('fill_missing_for_nan', Fill_Empty_Spaces_With_NaN(all_features=self.get_features(),find_in=self.get_missing_int_features()+self.find_missing_str,with_=np.nan)),
                        ('Mice_Imputer', Fill_Empty_Spaces_With_Values(all_features=self.get_features())),
                        ('Round_of_Values', Round_Of_Values(all_feat=self.get_features(),feat_to_round=self.feature_to_dummy))
                        ]) if pipeline_config is None else pipeline_config
        
    def split_data(self, **kwargs):
        """
        This method splits the data into train, test, and validation sets.

        Parameters:
        test_size (float): The proportion of data to be used as the test set. Default is 0.2.
        val_size (float): The proportion of data to be used as the validation set. Default is 0.2.
        random_state (int): The seed used by the random number generator. Default is 0.
        split_data (bool): Determines if the test set should be split into validation and test sets. Default is True.
        save_split_data (bool): Determines if the splits should be saved to disk. Default is True.
        train_save_path (str): The file path to save the train set. Default is "train.csv".
        test_save_path (str): The file path to save the test set. Default is "test.csv".
        val_save_path (str): The file path to save the validation set. Default is "val.csv".

        Returns:
        None
        """
        test_size = kwargs.get("test_size", 0.2)
        val_size = kwargs.get("val_size", 0.2)
        random_state = kwargs.get("random_state", 0)
        split_data = kwargs.get("split_data", True)
        save_splits = kwargs.get("save_split_data", True)
        train_save_path = kwargs.get("train_save_path", "train.csv")
        test_save_path = kwargs.get("test_save_path", "test.csv")
        val_save_path = kwargs.get("val_save_path", "val.csv")

        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Input data is not of type pandas.DataFrame")

        if split_data and test_size and float(test_size) > 0:
            self.train_data, self.val_test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)

            if val_size and float(val_size) > 0:
                self.val_data, self.test_data = train_test_split(self.val_test_data, test_size=0.5, random_state=random_state)
                if test_save_path is not None and save_splits:
                    self.save_data(self.test_data, test_save_path)
                if val_save_path is not None and save_splits:
                    self.save_data(self.val_data, val_save_path)
            else:
                if test_save_path is not None and save_splits:
                    self.save_data(self.val_test_data, test_save_path)
        else:
            if save_splits and train_save_path is not None:
                self.save_data(self.data, train_save_path)


    def fit_pipeline(self, **kwargs):
        """
            Fit the data processing pipeline and optionally save it.
            Parameters:
            save_data_pipeline (bool, optional): Whether to save the pipeline after fitting. Defaults to True.
            pipeline_path (str, optional): The file path to save the pipeline. Defaults to None.
            data (Union[Pandas.DataFrame, numpy.ndarray], optional): The data to fit the pipeline with. Defaults to None.
        """
        
        save_data_pipeline = kwargs.get("save_data_pipeline", True)
        pipeline_path = kwargs.get("pipeline_path", None)
        data = kwargs.get("data", None)
        
        if data is not None and isinstance(data, (pd.DataFrame, np.ndarray)):
            self.pipeline_config.fit(data)
        else:
            self.pipeline_config.fit(self.data)

        if save_data_pipeline:
            self.save_pipeline(self.pipeline_config, pipeline_path)

    def save_pipeline(self, pipeline, pipeline_path=None):
        """
        Save the pipeline to disk.

        Parameters:
        pipeline (scikit-learn Pipeline): The pipeline to save.
        save_path (str, optional): The file path to save the pipeline. Defaults to `self.save_path`.

        """
        assert isinstance(pipeline_path, str), "pipeline_path should be a string"
        assert isinstance(pipeline, Pipeline) or pipeline is None, "pipeline should be a Pipeline object or None"
        assert pipeline.steps, "pipeline should be fitted before saving"
        
        if hasattr(pipeline, 'steps'):
            for step in pipeline.steps:
                if hasattr(step[1], 'coef_'):
                    print(f"Pipeline step {step[0]} has been fit on data.")
                else:
                    print(f"Pipeline step {step[0]} has not been fit on data.")
        else:
            raise ValueError("This is not a scikit-learn pipeline or not a fitted scikit-learn pipeline.")
        
        pipeline_path = pipeline_path if pipeline_path is None else "artifacts/data_pipeline/pipeline_path.pkl"
        if ".pkl" not in pipeline_path:
            pipeline_path = pipeline_path + ".pkl"
        joblib.dump(pipeline, pipeline_path)


    def load_pipeline(self, load_path):
        """
        Load the pipeline from disk.

        Parameters:
        load_path (str, optional): The file path to load the pipeline from. Defaults to None.

        """
        try:
            self.pipeline_config = joblib.load(load_path)
        except Exception as e:
            raise Exception(f'Could not load pipeline, because of error: {e}')
    
    def process_data(self, data=None, **kwargs):
        """
            This method processes the data, splitting it into train, validation, and test sets if specified.
            Parameters:
            data (pandas.DataFrame, optional): The data to be processed. Default is None, in which case the data stored in the object's data attribute is used.
            split_data (bool, optional): Determines if the data should be split into train, validation, and test sets. Default is True.
            save_split_data (bool, optional): Determines if the splits should be saved to disk. Default is True.
            load_data_pipeline (bool, optional): Determines if the data pipeline should be loaded from disk. Default is False.
            fit_data_pipeline (bool, optional): Determines if the data pipeline should be fit to the data. Default is True.
            save_data_pipeline (bool, optional): Determines if the data pipeline should be saved to disk. Default is True.
            test_size (float, optional): The proportion of data to be used as the test set. Default is 0.2.
            val_size (float, optional): The proportion of data to be used as the validation set. Default is 0.2.
            random_state (int, optional): The seed used by the random number generator. Default is 0.
            train_save_path (str, optional): The file path to save the train set. Default is "train.csv".
            test_save_path (str, optional): The file path to save the test set. Default is "test.csv".
            val_save_path (str, optional): The file path to save the validation set. Default is "val.csv".
            pipeline_path (str, optional): The file path to save the data pipeline. Default is "pipeline.pkl".

            Returns:
            None
        """
        
        if isinstance(data, type(None)):
            data = self.data
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data is not of type pandas.DataFrame")
        
        self.data = data
        split_data = kwargs.get("split_data", True)
        fit_data_pipeline = kwargs.get("fit_data_pipeline", True)
        load_data_pipeline = kwargs.get("load_data_pipeline", False)
        
        if split_data:
            self.split_data(**kwargs)
        
        if load_data_pipeline:
            pipeline_path = kwargs.get("pipeline_path", "pipeline.pkl")
            self.load_pipeline(load_path=pipeline_path)
            
        if fit_data_pipeline:
            self.fit_pipeline(data=data, **kwargs)
                
class DataPostProcessor(DataPreProcessor):
    
    def __init__(self, file_path,  pipeline_path):
        super().__init__(file_path)
    
    def process_data(self, data=None, **kwargs):
        if isinstance(data, type(None)):
            data = self.data
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data is not of type pandas.DataFrame")
        
# TODO: write classes to perform transformation
    # Transformations: 
    #       - Fill name
    #       - Split data into train and test
    #       - Shift data target by one time spet forward
    #       - Label target as 1 for uptrend and 0 for downtrend
    #       - One hot encode target
    #       - Create a pipeline to fill nan, shift targets, label and do one hot encode and scaler
    
if __name__ == '__main__':
    from utils.util_functions import split_data_util, process_data_util
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_percentage", type=float, default=0.2, help="Percentage of data to use for test split")
    parser.add_argument("--split_data", type=bool, default=True, help="Boolean flag to specify if the data should be split or not")
    parser.add_argument("--process_data", type=bool, default=True, help="Boolean flag to specify if the data should be processed or not")
    parser.add_argument("--save_datapipelined", type=bool, default=True, help="Boolean flag to specify if the datapipeline should be saved or not")
    parser.add_argument("--transform_data", type=bool, default=True, help="Boolean flag to specify if the data should be transformed or not")
    parser.add_argument("--save_transformed", type=bool, default=True, help="Boolean flag to specify if the transformed data should be saved or not")
    parser.add_argument("--datapipelined_path", type=str, default=None, help="Path to save the datapipeline")
    parser.add_argument("--save_transformed_path", type=str, default=None, help="Path to save the transformed data")
    parser.add_argument("--data_path", type=str, default='data/data.csv', help="Path to the data file")
    parser.add_argument("--left_split_save_path", type=str, default="train_split.csv", help="Path to save the left split of data")
    parser.add_argument("--right_split_save_path", type=str, default="test_split.csv", help="Path to save the right split of data")
    parser.add_argument("--save_split", type=bool, default=False, help="Boolean flag to specify if the split files should be saved or not")

    args = parser.parse_args()

    split_percentage = args.split_percentage
    split_data = args.split_data
    process_data = args.process_data
    save_datapipelined = args.save_datapipelined
    transform_data = args.transform_data
    save_transformed = args.save_transformed
    datapipelined_path = args.datapipelined_path
    save_transformed_path = args.save_transformed_path
    data_path = args.data_path
    left_split_save_path = args.left_split_save_path
    right_split_save_path = args.right_split_save_path
    save_split = args.save_split

    processor = DataPreProcessor(data_path)
    processor.process_data(**args)