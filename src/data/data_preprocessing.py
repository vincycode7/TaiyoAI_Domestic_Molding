import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

class OneHotEncoderColumns(BaseEstimator, TransformerMixin):
    """
    A class used to preprocess the data by one-hot encoding specified features.

    Parameters:
        all_feat (list, optional): List of all features to be considered.
            Default: ["make_id", "model_id", "series_id", "is_verified_dealer", "year_of_manufacture", "listingtitle", "conditiontitle", "sailthru_tag"]
        feat_to_dummy (list, optional): List of features to be one-hot encoded.
            Default: ["make_id","model_id","series_id","is_verified_dealer","year_of_manufacture","listingtitle", "conditiontitle", "sailthru_tag"]

    Attributes:
        feat_to_dummy (list): List of features to be one-hot encoded.
        all_feat (list): List of all features to be considered.
        one_hot_encoder (OneHotEncoder): The instance of OneHotEncoder used to fit and transform the data.
    """

    def __init__(self, all_feat=["make_id", "model_id", "series_id", "is_verified_dealer", "year_of_manufacture", "listingtitle", "conditiontitle", "sailthru_tag"],
                 feat_to_dummy=["make_id","model_id","series_id","is_verified_dealer","year_of_manufacture","listingtitle", "conditiontitle", "sailthru_tag"]):
        self.feat_to_dummy = feat_to_dummy
        self.all_feat = all_feat

    def fit(self, X, y=None):
        """
        Fit the OneHotEncoder to the data.
        """
        try:
            X = X[self.all_feat]
        except KeyError as exp:
            raise KeyError("Error: One or more columns specified in all_feat not found in input data") from exp

        self.one_hot_encoder = OneHotEncoder().fit(X)
        return self

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
    
class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.data = data
        self.save_path = save_path
        self.pipeline = None

    def load_data(self, data_path):
        self.data = pd.read_csv(data_path)

    def split_data(self, test_size=0.2, random_state=0):
        self.train_data, self.val_test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)
        self.val_data, self.test_data = train_test_split(self.val_test_data, test_size=0.5, random_state=random_state)

    def fit_pipeline(self, steps):
        self.pipeline = Pipeline(steps)
        self.pipeline.fit(self.train_data)

    def save_pipeline(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        joblib.dump(self.pipeline, save_path)

    def load_pipeline(self, load_path=None):
        if load_path is None:
            load_path = self.save_path
        self.pipeline = joblib.load(load_path)

    def process_data(self, data=None):
        if data is None:
            data = self.data
        return self.pipeline.transform(data)
    
    
# TODO: write classes to perform transformation
    # Transformations: 
    #       - Fill name
    #       - Split data into train and test
    #       - Shift data target by one time spet forward
    #       - Label target as 1 for uptrend and 0 for downtrend
    #       - One hot encode target
    #       - Create a pipeline to fill nan, shift targets, label and do one hot encode and scaler
    
if __name__=='__main__':
    split_data = True
    process_data = True # if process_data is True, then process will run on train samples if no error occures
    save_datapipelined = True
    transform_data = True # if transform_data is True, then transform will run on train samples if no error occures
    save_transformed = True # if save_transform is True, then transformed data is saved in specified path
    datapipelined_path = None
    save_transformed_path = None
    
    data_path = 'data/data.csv'
    processor = DataProcessor(data_path)
    processor.load_data(data_path)
    processor.split_data(test_size=0.2, random_state=0)
    processor.fit_pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
        ('scaler', StandardScaler()),

    ])
    processor.save_pipeline()