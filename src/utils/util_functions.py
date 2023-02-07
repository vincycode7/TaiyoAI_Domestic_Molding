def split_data_util(processor, split_data, split_percentage, save_split=False, left_split_save_path='train_split.csv', right_split_save_path='test_split.csv'):
    """
    Splits the data into train and test sets using the specified split percentage
    
    Parameters:
        processor (DataProcessor): Instance of the DataProcessor class
        split_data (bool): Boolean flag to specify if the data should be split or not
        split_percentage (float): Percentage of data to use for test split
        save_split (bool, optional): Boolean flag to specify if the split should be saved or not. Defaults to False.
        left_split_save_path (str, optional): Path to save the train split data. Defaults to "train_split.csv".
        right_split_save_path (str, optional): Path to save the test split data. Defaults to "test_split.csv".
        
    Returns:
        None
    """
    processor.split_data(test_size=split_percentage, random_state=0, safe_split=safe_split, left_split_save_path=left_split_save_path, right_split_save_path=right_split_save_path)

def process_data_util(processor, process_data, save_datapipelined, transform_data, save_transformed, datapipelined_path=None, save_transformed_path=None):
    """
    Processes the data using the specified pipeline steps
    
    Parameters:
        processor (DataProcessor): Instance of the DataProcessor class
        process_data (bool): Boolean flag to specify if the data should be processed or not
        save_datapipelined (bool): Boolean flag to specify if the datapipeline should be saved or not
        transform_data (bool): Boolean flag to specify if the data should be transformed or not
        save_transformed (bool): Boolean flag to specify if the transformed data should be saved or not
        datapipelined_path (str, optional): Path to save the datapipeline. Defaults to None.
        save_transformed_path (str, optional): Path to save the transformed data. Defaults to None.
        
    Returns:
        None
    """
    if process_data:
        processor.fit_pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
            ('scaler', StandardScaler()),

        ])
        if save_datapipelined:
            processor.save_pipeline(path=datapipelined_path)
        if transform_data:
            X_train_transformed = processor.transform(processor.X_train)
            if save_transformed:
                processor.save_transformed(X_train_transformed, path=save_transformed_path)
            
def get_data_ready(raw_data):
    return

