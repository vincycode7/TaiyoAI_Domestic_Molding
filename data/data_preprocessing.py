import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
import joblib
import argparse

class Normalize_Data(BaseEstimator, TransformerMixin):
    def __init__(self, all_features,normalise_cols, normaliser=None):
        # Read in data
        self.all_features = all_features
        self.normalise_cols = normalise_cols
        self.normaliser = Normalizer() if not normaliser else normaliser

    def fit(self, X, y=None):
        
        # Assert that all features and fill_in are available in the input data
        try:
            X[self.all_features]
            X[self.normalise_cols]
            X = X.copy()
        except Exception as exp:
            raise exp

        self.normaliser.fit(X[self.normalise_cols].values)
        return self

    def transform(self, X):
        """
        Work on the dataset
        """
        try:
            X[self.all_features]
            X[self.normalise_cols]
            X = X.copy()
        except Exception as exp:
            raise exp
        
        # Transform only the columns in the fill_in list
        X_normaliser = self.normaliser.transform(X[self.normalise_cols].values)
        df_normaliser = pd.DataFrame(X_normaliser, columns=X[self.normalise_cols].columns)
        X.loc[:,self.normalise_cols] = df_normaliser

        # Return the dataframe with the new fill_in values and the other columns in the features list
        return X
class Select_Column(BaseEstimator, TransformerMixin):
    def __init__(self, all_features,select_column):
        #Read in data
        self.all_features = all_features
        self.select_column = select_column
        
    def fit(self, X, y=None):
        
        return self #do nothing

    def transform(self, X):
        try:
            X[self.all_features]
            X = X.copy() 
        except Exception as exp:
            raise exp
        return X[self.select_column]
    
class Shift_Target(BaseEstimator, TransformerMixin):    
    def __init__(self, all_features,
                        target,
                        target_shift_name="target_shift",
                        ):
    
        #Read in data
        self.all_features = all_features
        self.target = target
        self.target_shift_name = target_shift_name

    def fit(self, X, y=None):
        
        return self #do nothing
    
    def transform(self, X):
        X[self.target_shift_name] = X[self.target].shift(-1)
        return X
    
class Label_Shift_Target(BaseEstimator, TransformerMixin):
    def __init__(self, all_features,target,target_shift_name,target_shift_direction_name):
        #Read in data
        self.all_features = all_features
        self.target = target
        self.target_shift_name = target_shift_name
        self.target_shift_direction_name = target_shift_direction_name
        
    def fit(self, X, y=None):
        
        return self #do nothing

    def transform(self, X):
        try:
            X[self.all_features]
            X = X.copy() 
        except Exception as exp:
            raise exp
            
        X[self.target_shift_direction_name] = np.where(X[self.target].astype('float') > X[self.target_shift_name].astype('float'), -1, 1)
        return X

class Fill_Empty_Spaces_With_Values(BaseEstimator, TransformerMixin):
    """
    This class is used to preprocess the data by filling missing values with standard values that
    represent missing values, such as numpy.nan.
    """
    def __init__(self, all_features=[], fill_in=[], imputer=None):
        # Read in data
        self.all_features = all_features
        self.fill_in = fill_in
        self.imputer = IterativeImputer(max_iter=20, random_state=0) if not imputer else imputer

    def fit(self, X, y=None):
        
        # Assert that all features and fill_in are available in the input data
        try:
            X[self.all_features]
            X[self.fill_in]
        except Exception as exp:
            raise exp

        # Fit the imputer on columns in the fill_in list
        X = X.copy()

        self.imputer.fit(X[self.fill_in].values)
        return self

    def transform(self, X):
        """
        Work on the dataset
        """
        try:
            X[self.all_features]
            X = X.copy()
        except Exception as exp:
            raise exp
        
        # Transform only the columns in the fill_in list
        X_imputed = self.imputer.transform(X[self.fill_in].values)
        df_imputed = pd.DataFrame(X_imputed, columns=X[self.fill_in].columns)
        X.loc[:,self.fill_in] = df_imputed

        # Return the dataframe with the new fill_in values and the other columns in the features list
        return X

class Replace_Char(BaseEstimator, TransformerMixin):
    """
    This is a Class Used to Preprocess the data, By
    Replacing Specified Values with New Values
    """
    def __init__(self, all_features=[],
                        find_in=[],
                        find=[","],
                        with_=""
                        ):

        #Read in data
        self.all_features = all_features
        self.find_in = find_in
        self.find = find
        self.with_ = with_

    def fit(self, X, y=None):
        
        return self #do nothing

    def transform(self,X):
        """
            Replace specified values with new values
        """
        try:
            X[self.all_features]
            X = X.copy()
        except Exception as exp:
            raise exp
        
        for find_value in self.find:
            X.loc[:, self.find_in] = X[self.find_in].applymap(lambda x: str(x).replace(find_value, self.with_))
        return X
    
      
class DropNanRows(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data by
        dropping rows with NaN values.
    """
    def __init__(self, all_features=[]):
        self.all_features = all_features
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
            Drop rows with NaN values.
        """
        try:
            X[self.all_features]
            X = X.copy()
        except Exception as exp:
            raise exp
        
        X.dropna(inplace=True)
        return X


class Fill_Empty_Spaces_With_NaN(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data, By
        Filling Missing Values with Standard Values That
        Represents Missing Values, e.g numpy.nan.
    """
    
    def __init__(self, all_features=[],
                        find_in=[],
                                        
                        find=None,
                        with_=None
                        ):

        #Read in data
        self.all_features = all_features
        self.find_in = find_in
        self.find = ['?','? ',' ?',' ? ','',' ','-',None,'None','none','Null','null',np.nan] if not find else find
        self.with_ = np.nan if not with_ else with_

    def fit(self, X, y=None):
        
        return self #do nothing
    
    def transform(self,X):
        """
            Work on the dataset
        """
        
        try:
            X[self.all_features]
            X = X.copy()
        except Exception as exp:
            raise exp
        
        # Loop through columns and values in the dataframe
        for column in self.find_in:
            for i, val in enumerate(X[column]):
                if val in self.find:
                    X.at[i, column] = self.with_
        
        return X
    
class Back_To_Float(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data, By
        encoding N features and filling missing values
        too
    """
    
    def __init__(self, all_features=[],  
                        to_encode=[]):
    
        #Read in data
        self.all_features = all_features
        self.to_encode = to_encode

    def fit(self, X, y=None):
        
        #check if features are present
        return self #do nothing

    def transform(self,X):
        """
            Work on the dataset
        """
        #check if features are present
        try:
            X[self.all_features]
            X = X.copy()
            X.loc[:, self.to_encode] = X[self.to_encode].astype('float')
        except Exception as exp:
            raise exp
        return X
    
class OneHotEncode_Columns(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data by
        one hot encoding of specified features.
    """
    
    def __init__(self, all_feat=[], feat_to_dummy=[]):
        self.feat_to_dummy = feat_to_dummy
        self.all_feat = all_feat

    def fit(self, X, y=None):
        
        try:
            X[self.all_feat]
            X = X.copy()
        except KeyError as exp:
            raise exp
        
        self.one_hot_encoder = OneHotEncoder()
        self.one_hot_encoder.fit(X[self.feat_to_dummy])
        return self

    def transform(self, X):
        try:
            X[self.all_feat]
            X[self.feat_to_dummy]
            X = X.copy()
        except KeyError as exp:
            raise exp

        X_encoded = self.one_hot_encoder.transform(X[self.feat_to_dummy])
        X[list(self.one_hot_encoder.get_feature_names_out())] = X_encoded.toarray()
        return X

    
class BaseDataProcessor:
    """
    This class is used for processing the data, it takes data_file_path, all_columns, target_feature and 
    exclude_feature_from_export as inputs and performs several operations on the data like cleaning 
    column names, filling empty spaces with NaN, converting int columns to float and rounding values.

    Args:
        data_file_path(str): The file path of the data.
        all_columns (List[str]): The list of all columns in the data.
        target_feature (str): The target feature for which the model is being built.
        exclude_feature_from_export (List[str]): The list of features that should not be exported.
    """

    def __init__(self, **kwargs):
        data_file_path= kwargs.get("data_file_path", None)
        _all_columns=["Date", "Spot/Export Blow Molding", "Spot, Domestic", 
                                            "WTISPLC", "MCOILBRENTEU", "GASREGM", "IMPCH", "EXPCH", 
                                            "PRUBBUSDM", "WPUFD4111", "PCU325211325211", "PCU32611332611301", 
                                            "WPU0915021625","PCU3252132521", "MHHNGSP", "WPU072205011", 
                                            "PCU32611132611115", "PCU32611332611301","PCU32611132611112", 
                                            "Producer Price Index by Industry: Plastics Material and Resins Manufacturing: Thermoplastic Resins and Plastics Materials ",
                                            "Australia _export", "Canada_export", "Saudi_export", "Usa_export", 
                                            "India_export", "Russia_export", "South_Africa_export", "Turkey","Brazil", 
                                            "France_export", "Germeny_export", "United Kingdome_export","China_export",	
                                            "Australia _import", "Canada_import","Saudi_import","Usa_import","India_import", 
                                            "Russia_import", "South_Africa_import", "Turkey_import","Brazil_import", 
                                            "France_import", "Germeny_import", "United Kingdome_import", 
                                            "China_import", "Japan_import"]
        all_columns = kwargs.get("all_columns", None) if kwargs.get("all_columns", None) else _all_columns
        _target_feature="Domestic Market (Contract) Blow Molding, Low"
        target_feature = kwargs.get("target_feature", None) if kwargs.get("target_feature", None) else _target_feature
        _exclude_feature_from_export=["Date"]
        exclude_feature_from_export = kwargs.get("exclude_feature_from_export") if kwargs.get("exclude_feature_from_export") else _exclude_feature_from_export
        
        assert isinstance(data_file_path, str), "data_file_path should be a string"
        assert isinstance(all_columns, list), "all_columns should be a list of strings"
        assert isinstance(target_feature, str), "target_feature should be a string"
        assert isinstance(exclude_feature_from_export, list) or exclude_feature_from_export is None, "exclude_feature_from_export should be a list or None"

        self.data_file_path = data_file_path
        self.all_columns = all_columns
        self.target_feature = target_feature
        self.exclude_feature_from_export = exclude_feature_from_export
        self.data = None
        
        self.init_data()

    def load_data(self):
        self.data = pd.read_csv(self.data_file_path)
        
    def get_base_data(self):
        return self.data

    def get_all_columns(self):
        return self.all_columns
  
    def get_all_features(self):
        return list(set([col for col in self.all_columns if (col not in self.exclude_feature_from_export)]))

    def get_target_feature(self):
        return self.target_feature

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
            col_name = col_name.strip().replace(",", "").replace(":", "").replace("(", "").replace(")", "").replace("/", "").strip().replace(" ", "_").lower()
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

    def save_data(self, data, data_file_path):
        """
            Saves the processed data to a csv file.

            Parameters:
            data (pandas.DataFrame): The data to be saved.
            data_file_path (str): The file path to save the data to.

            Returns:
            None
        """
        # data = data.drop(columns=self.exclude_feature_from_export)
        data.to_csv(data_file_path, index=False)
        
class DataPreProcessor(BaseDataProcessor):
    """
    This class is used for processing the data, it takes data_file_path, all_columns, target_feature and 
    exclude_feature_from_export as inputs and performs several operations on the data like cleaning 
    column names, filling empty spaces with NaN, converting int columns to float and rounding values.

    Args:
        data_file_path (str): The file path of the data.
        all_columns (List[str]): The list of all columns in the data.
        target_feature (str): The target feature for which the model is being built.
        exclude_feature_from_export (List[str]): The list of features that should not be exported.
        pipeline_config_feature (Union[None, sklearn.pipeline.Pipeline]): The pipeline configuration.
        pipeline_config_target (Union[None, sklearn.pipeline.Pipeline]): The pipeline configuration.
        pipeline_config_normaliser (Union[None, sklearn.pipeline.Pipeline]): The pipeline configuration.
    """
    def __init__(self,pipeline_config_feature=None,  pipeline_config_target=None, pipeline_config_normaliser=None, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(pipeline_config_feature, Pipeline) or pipeline_config_feature is None, "pipeline_config_feature should be a Pipeline object or None"
        assert isinstance(pipeline_config_target, Pipeline) or pipeline_config_target is None, "pipeline_config_target should be a Pipeline object or None"
        assert isinstance(pipeline_config_normaliser, Pipeline) or pipeline_config_normaliser is None, "pipeline_config_normaliser should be a Pipeline object or None"

        #Initialize Pipeline for features
        self.pipeline_config_feature = Pipeline([
                        ('replace_all_coman_with_empt_space', Replace_Char(all_features=self.get_all_features(),find_in=self.get_all_features(),with_='')),
                        ('fill_missing_with_NaN', Fill_Empty_Spaces_With_NaN(all_features=self.get_all_features(),find_in=self.get_all_features(),with_='NaN')),
                        ('int_column_to_float', Back_To_Float(all_features=self.get_all_features(), to_encode=self.get_all_features())),
                        ('fill_missing_for_nan', Fill_Empty_Spaces_With_NaN(all_features=self.get_all_features(),find_in=self.get_all_features(),with_=np.nan)),
                        ('Mice_Imputer', Fill_Empty_Spaces_With_Values(all_features=self.get_all_features(), fill_in=self.get_all_features())),
                        ('Select_Column', Select_Column(all_features=self.get_all_features(), select_column=self.get_all_features())),
                        ('DropNanRows', DropNanRows(all_features=self.get_all_features())),
                        ]) if pipeline_config_feature is None else pipeline_config_feature
        
        target_shift_name = "target_shift"
        target_shift_direction_name = "target_shift_direction"
        target_shift_direction_one_hot_list_names = [target_shift_direction_name+"_-1", target_shift_direction_name+"_1"]
        
        #Initialize Pipeline for target
        self.pipeline_config_target = Pipeline([
                        ('replace_all_coman_with_empt_space', Replace_Char(all_features=self.get_all_features(),find_in=[self.get_target_feature()],with_='')),
                        ('shift_target', Shift_Target(all_features=self.get_all_features(),target=self.get_target_feature(),target_shift_name=target_shift_name)),
                        ('fill_missing_target', Fill_Empty_Spaces_With_NaN(all_features=self.get_all_features()+[target_shift_name],find_in=[target_shift_name],with_=np.nan)),
                        ('Mice_Imputer_target', Fill_Empty_Spaces_With_Values(all_features=self.get_all_features()+[target_shift_name], fill_in=[target_shift_name])),
                        ('label_shift_target', Label_Shift_Target(
                            all_features=self.get_all_features()+[target_shift_name],
                            target=self.get_target_feature(),
                            target_shift_name=target_shift_name,
                            target_shift_direction_name=target_shift_direction_name)),
                        ('One_Hot_Encode_Shift_Target_Label', OneHotEncode_Columns(all_feat=self.get_all_features()+[target_shift_name], feat_to_dummy=[target_shift_direction_name])),
                        ('Select_Column', Select_Column(all_features=self.get_all_features()+[target_shift_name]+target_shift_direction_one_hot_list_names, select_column=[target_shift_name,target_shift_direction_name]+target_shift_direction_one_hot_list_names))
                        ]) 
        self.target_shift_name = target_shift_name
        self.target_shift_direction_name = target_shift_direction_name
        
        #Initialize feature Normalizer
        self.pipeline_config_normaliser = Pipeline([
                        ('Select_Column', Select_Column(all_features=self.get_all_features(), select_column=self.get_all_features())),
                        ('Normalize', Normalize_Data(all_features=self.get_all_features(), normalise_cols=self.get_all_features()))
                        ]) if pipeline_config_normaliser is None else pipeline_config_normaliser
        
        
    def get_target_shift_direction(self):
        return self.target_shift_direction_name
    
    def get_target_shift_name(self):
        return self.target_shift_name
    
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
            test_size = float(test_size)+float(val_size) if float(val_size) else test_size
            self.train_data, self.val_test_data = train_test_split(self.data, test_size=test_size,shuffle=False)

            if val_size and float(val_size) > 0:
                self.val_data, self.test_data = train_test_split(self.val_test_data, test_size=1/(test_size/float(val_size)),shuffle=False)
                if test_save_path is not None and save_splits:
                    self.save_data(self.test_data, test_save_path)
                if val_save_path is not None and save_splits:
                    self.save_data(self.val_data, val_save_path)
            else:
                if test_save_path is not None and save_splits:
                    self.save_data(self.val_test_data, test_save_path)
            self.save_data(self.train_data, train_save_path)
            self.data = self.train_data
        else:
            if save_splits and train_save_path is not None:
                self.save_data(self.data, train_save_path)


    def fit_pipeline(self, **kwargs):
        """
            Fit the data processing pipeline and optionally save it.
            Parameters:
            save_data_pipeline (bool, optional): Whether to save the pipeline after fitting. Defaults to True.
            data_pipelined_path (str, optional): The file path to save the pipeline. Defaults to None.
            data (Union[Pandas.DataFrame, numpy.ndarray], optional): The data to fit the pipeline with. Defaults to None.
        """
        save_data_pipeline = kwargs.get("save_data_pipeline", True)
        data_pipelined_path = kwargs.get("data_pipelined_path", None)
        data = kwargs.get("data", None)
        
        if data is not None and isinstance(data, (pd.DataFrame, np.ndarray)):
            pass
        else:
            data = self.data
        self.pipeline_config_feature = self.pipeline_config_feature.fit(data)
        self.transformed_X_data = self.pipeline_config_feature.transform(data)
        
        self.pipeline_config_normaliser = self.pipeline_config_normaliser.fit(self.transformed_X_data)
        self.transformed_X_data_norm = self.pipeline_config_normaliser.transform(self.transformed_X_data)
        
        self.pipeline_config_target = self.pipeline_config_target.fit(self.transformed_X_data)
        self.transformed_Y_data = self.pipeline_config_target.transform(self.transformed_X_data)

        if save_data_pipeline:
            self.save_pipeline(pipeline_config_feature=self.pipeline_config_feature,pipeline_config_target=self.pipeline_config_target,pipeline_config_normaliser=self.pipeline_config_normaliser , data_pipelined_path=data_pipelined_path)
            
    def transform_pipeline(self, **kwargs):
        data = kwargs.get("data", None)
        normalise_transformed_feature = kwargs.get("normalise_transformed_feature", True)
        ignore_target_pipeline = kwargs.get("ignore_target_pipeline", True)
        
        if data is not None and isinstance(data, (pd.DataFrame, np.ndarray)):
            pass
        else:
            data = self.data
        self.transformed_X_data = self.pipeline_config_feature.transform(data)
        
        if normalise_transformed_feature:
            self.transformed_X_data_norm = self.pipeline_config_normaliser.transform(self.transformed_X_data)
        
        if not ignore_target_pipeline:
            self.transformed_Y_data = self.pipeline_config_target.transform(self.transformed_X_data)
        
    def save_pipeline(self, pipeline_config_feature,pipeline_config_target,pipeline_config_normaliser , data_pipelined_path=None):
        """
            Save the pipeline to disk.

            Parameters:
            pipeline (scikit-learn Pipeline): The pipeline to save.
            save_path (str, optional): The file path to save the pipeline. Defaults to `self.save_path`.

        """
        assert isinstance(data_pipelined_path, str), f"data_pipelined_path should be a string, but currently of type {type(data_pipelined_path)}"
        assert isinstance(pipeline_config_feature, Pipeline), "pipeline_config_feature should be a Pipeline object or None"
        assert isinstance(pipeline_config_target, Pipeline), "pipeline_config_target should be a Pipeline object or None"
        assert isinstance(pipeline_config_normaliser, Pipeline), "pipeline_config_normaliser should be a Pipeline object or None"
        assert pipeline_config_feature.steps, "pipeline_config_feature should be fitted before saving"
        assert pipeline_config_target.steps, "pipeline_config_target should be fitted before saving"
            
        data_pipelined_path = data_pipelined_path if data_pipelined_path else "artifacts/data_pipeline/data_pipelined_path.pkl"
        if ".pkl" not in data_pipelined_path:
            data_pipelined_path = data_pipelined_path + ".pkl"
        joblib.dump({"pipeline_config_feature":pipeline_config_feature, 
                     "pipeline_config_target":pipeline_config_target,
                     "pipeline_config_normaliser":pipeline_config_normaliser
                     }, 
                    data_pipelined_path)


    def load_pipeline(self, load_path):
        """
            Load the pipeline from disk.

            Parameters:
            load_path (str, optional): The file path to load the pipeline from. Defaults to None.

        """
        try:
            pipeline_config = joblib.load(load_path)
            self.pipeline_config_feature = pipeline_config["pipeline_config_feature"]
            self.pipeline_config_target = pipeline_config["pipeline_config_target"]
            self.pipeline_config_normaliser = pipeline_config["pipeline_config_normaliser"]
        except Exception as e:
            raise Exception(f'Could not load pipeline, because of error: {e}')
    
    def pre_process_data(self, data=None, **kwargs):
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
            data_pipelined_path (str, optional): The file path to save the data pipeline. Default is "pipeline.pkl".

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
            data_pipelined_path = kwargs.get("data_pipelined_path", "pipeline.pkl")
            self.load_pipeline(load_path=data_pipelined_path)
            
        if fit_data_pipeline:
            self.fit_pipeline(data=self.data, **kwargs)
                
class DataPostProcessor(DataPreProcessor):
    
    def __init__(self,  data_pipelined_path, **kwargs):
        super().__init__(**kwargs)
        data_pipelined_path = data_pipelined_path if data_pipelined_path else None
        assert isinstance(data_pipelined_path, str), "data_pipelined_path should be a string"
        
        self.load_pipeline(data_pipelined_path)
    
    def post_process_data(self, data=None, **kwargs):
        
        if isinstance(data, type(None)):
            data = self.data
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data is not of type pandas.DataFrame")
        
        self.transform_pipeline(data=data,**kwargs)
        
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
    from configs.configs import get_data_processor_config
    data_processor_config = get_data_processor_config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_percentage", type=float, default=data_processor_config.get("split_percentage", 0.2), help="Percentage of data to use for test split")
    parser.add_argument("--split_data", type=bool, default=data_processor_config.get("split_data",True), help="Boolean flag to specify if the data should be split or not")
    parser.add_argument("--process_data", type=bool, default=data_processor_config.get("process_data",True), help="Boolean flag to specify if the data should be processed or not")
    parser.add_argument("--save_datapipelined", type=bool, default=data_processor_config.get("save_datapipelined",True), help="Boolean flag to specify if the datapipeline should be saved or not")
    parser.add_argument("--transform_data", type=bool, default=data_processor_config.get("transform_data",True), help="Boolean flag to specify if the data should be transformed or not")
    parser.add_argument("--save_transformed", type=bool, default=data_processor_config.get("save_transformed",True), help="Boolean flag to specify if the transformed data should be saved or not")
    parser.add_argument("--datapipelined_path", type=str, default=data_processor_config.get("datapipelined_path",None), help="Path to save the datapipeline")
    parser.add_argument("--save_transformed_path", type=str, default=data_processor_config.get("save_transformed_path",None), help="Path to save the transformed data")
    parser.add_argument("--data_file_path", type=str, default=data_processor_config.get("data_file_path",'artifacts/data/raw/data.csv'), help="Path to the data file")
    parser.add_argument("--left_split_save_path", type=str, default=data_processor_config.get("left_split_save_path","train_split.csv"), help="Path to save the left split of data")
    parser.add_argument("--right_split_save_path", type=str, default=data_processor_config.get("right_split_save_path","test_split.csv"), help="Path to save the right split of data")
    parser.add_argument("--save_split", type=bool, default=data_processor_config.get("save_split",False), help="Boolean flag to specify if the split files should be saved or not")

    args = parser.parse_args()

    split_percentage = args.split_percentage
    split_data = args.split_data
    process_data = args.process_data
    save_datapipelined = args.save_datapipelined
    transform_data = args.transform_data
    save_transformed = args.save_transformed
    datapipelined_path = args.datapipelined_path
    save_transformed_path = args.save_transformed_path
    data_file_path = args.data_file_path
    left_split_save_path = args.left_split_save_path
    right_split_save_path = args.right_split_save_path
    save_split = args.save_split

    processor = DataPreProcessor(data_file_path)
    processor.pre_process_data(**args)