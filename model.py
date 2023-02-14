from typing import Any
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score,r2_score,mean_absolute_error,roc_auc_score,balanced_accuracy_score,mean_absolute_percentage_error
import torch.nn as nn
from lazypredict.Supervised import LazyRegressor, LazyClassifier
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class BaseModelBackBone:
    def __init__(self):
        pass
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def predict(self, *args, **kwargs):
        raise NotImplementedError
    
    def train(self, *args, **kwargs):
        raise NotImplementedError
    
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_evaluation(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)
    
    def save_model(self, *args, **kwargs):
        raise NotImplementedError
    
    def load_model(self, *args, **kwargs):
        raise NotImplementedError
    
    def save_output(self,*args, **kwargs):
        raise NotImplementedError

class LazyPredictBackBone(BaseModelBackBone):
    def __init__(self,lazypred_instance):
        self.lazyobject = lazypred_instance
        
    def train(self,X_train, X_test, Y_train, Y_test):
        self.models,self.predictions = self.lazyobject.fit(X_train, X_test, Y_train, Y_test)

    def evaluate(self, *args, **kwargs):
        return self.predictions

class LazyPredictRegressionBackBone(LazyPredictBackBone):
    def __init__(self) -> None:
        regressor = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
        super().__init__(lazypred_instance=regressor)

class LazyPredictClassificationBackBone(LazyPredictBackBone):
    def __init__(self) -> None:
        classifier = LazyClassifier(verbose=0,ignore_warnings=False, custom_metric=None)
        super().__init__(lazypred_instance=classifier)
        
# Define the LSTM model
class LSTMBackBone(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
            super(LSTMBackBone, self).__init__()
            self.hidden_dim = hidden_dim
            self.layer_dim = layer_dim
            self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
            # self.relu = nn.ReLU()
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        # c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x)
        # print(out.shape)
        # out = self.fc(self.relu(out))
        out = self.fc(out)

        return out

    def get_evaluation(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.eval_results
    
    def evaluate_func(self, X_val, Y_val, criterion=None):
        criterion = nn.MSELoss() if isinstance(criterion, type(None)) else criterion
        
        # Evaluation on the validation set
        with torch.no_grad():
            y_val_pred = self(torch.Tensor(X_val.values))
            direction = np.where(torch.sigmoid(y_val_pred[:,1:]) >= 0.5, 1, 0)
            loss = criterion(y_val_pred, torch.Tensor(Y_val.values))
            rmse = np.sqrt(mean_squared_error(Y_val.values[:,:1], y_val_pred.detach().numpy()[:,:1]))
            mape = mean_absolute_percentage_error(Y_val.values[:,:1], y_val_pred.detach().numpy()[:,:1])
            r_squared = r2_score(Y_val.values[:,:1], y_val_pred.detach().numpy()[:,:1])
            adjusted_r_squared = 1 - (1-r_squared)*(len(Y_val.values[:,:1])-1)/(len(Y_val.values[:,:1])-X_val.values.shape[1]-1)
            mae = mean_absolute_error(Y_val.values[:,:1], y_val_pred.detach().numpy()[:,:1])
            
            directional_accuracy = accuracy_score(Y_val.values[:,1:],direction)
            f1 = f1_score(Y_val.values[:,1:],direction)
            roc_auc = roc_auc_score(Y_val.values[:,1:],direction)
            balanced_acc = balanced_accuracy_score(Y_val.values[:,1:],direction)  
        
        self.eval_results = {
            "loss":loss,
            "rmse": rmse,
            "mape":mape,
            "r_squared": r_squared,
            "adjusted_r_squared": adjusted_r_squared,
            "directional_accuracy": directional_accuracy,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "balanced_accuracy": balanced_acc,
            "mae":mae
        }
        return self.eval_results
        
    def train(self, X_train, X_val, Y_train, Y_val, num_epochs=950, learning_rate=0.001, batch_size=200):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        train_dataset = TensorDataset(torch.Tensor(X_train.values), torch.Tensor(Y_train.values))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        pbar = tqdm(total=num_epochs, )

        for epoch in range(num_epochs):
            running_loss = 0.0
            for idx, (x_batch, y_batch) in enumerate(train_dataloader, 0):
                y_pred = self(x_batch)
                loss = criterion(y_pred, y_batch)
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Print the average loss for the epoch
            pbar.update(1) 
                       
            # Evaluation on the validation set
            with torch.no_grad():
                evaluation = self.evaluate_func(X_val, Y_val)
                pbar.set_description(f"Epoch: {epoch + 1}/{num_epochs} Train loss: {running_loss / len(train_dataloader)} Loss: {evaluation['loss']:.4f} RMSE: {evaluation['rmse']:.4f} Adjusted R Squared: {evaluation['r_squared']:.4f} R^2: {evaluation['r_squared']:.4f} MAE: {evaluation['mae']:.4f} Directional Accuracy: {evaluation['directional_accuracy']:.4f} F1 Score: {evaluation['f1_score']:.4f} ROC AUC: {evaluation['roc_auc']:.4f} Balanced Accuracy: {evaluation['balanced_accuracy']:.4f}")
                
        pbar.close()
        print("Training complete!")

# # Define the Transformer model
# class TransformersBackBone(nn.Module):
#     def __init__(self, input_size, output_size, n_layers, n_heads, dim_model, dim_feedforward, dropout):
#         super(TransformersBackBone, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.n_heads = n_heads
#         self.dim_model = dim_model
#         self.dim_feedforward = dim_feedforward
#         self.dropout = dropout
        
#         # Multi-head attention layer
#         self.attention = nn.MultiheadAttention(dim_model, n_heads)
        
#         # Position-wise feedforward layer
#         self.feedforward = nn.Sequential(
#             nn.Linear(dim_model, dim_feedforward),
#             nn.ReLU(),
#             nn.Linear(dim_feedforward, dim_model)
#         )
        
#         # Dropout layer
#         self.dropout_layer = nn.Dropout(dropout)
        
#         # Output layer
#         self.output_layer = nn.Linear(dim_model, output_size)
        
#     def forward(self, x):
#         # Pass input through attention layer
#         attention_output, _ = self.attention(x, x, x)
        
#         # Pass output through feedforward layer
#         feedforward_output = self.feedforward(attention_output)
        
#         # Add the outputs and apply activation function
#         output = F.relu(attention_output + feedforward_output)
        
#         # Apply dropout
#         output = self.dropout_layer(output)
        
#         # Pass output through the output layer
#         prediction = self.output_layer(output)
        
#         return prediction
    
#     # Define a function to evaluate the model
#     def evaluate(model, test_inputs, test_labels):
#         with torch.no_grad():
#             # Get predictions
#             test_predictions = model(test_inputs)
            
#             # Calculate root mean squared error
#             rmse = torch.sqrt(torch.mean((test_predictions - test_labels) ** 2))
            
#             # Calculate R-Squared
#             total_sum_of_squares = torch.sum((test_labels - test_labels.mean()) ** 2)
#             residual_sum_of_squares = torch.sum((test_labels - test_predictions) ** 2)
#             r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
            
#             # Calculate Adjusted R-Squared
#             n = test_inputs.shape[0]
#             p = test_inputs.shape[1]
#             adjusted_r_squared = 1 - (1-r_squared) * (n-1) / (n-p-1)
            
#             # Calculate Directional Accuracy
#             direction_predictions = torch.sign(test_predictions)
#             direction_labels = torch.sign(test_labels)
#             directional_accuracy = torch.mean((direction_predictions == direction_labels).float())
            
#             # Convert the outputs to numpy arrays
#             test_predictions = test_predictions.detach().numpy().flatten()
#             test_labels = test_labels.detach().numpy().flatten()
            
#             # Calculate F1 Score
#             f1_score = metrics.f1_score(test_labels, test_predictions)
            
#             # Calculate ROC AUC
#             roc_auc = metrics.roc_auc_score(test_labels, test_predictions)
            
#             # Calculate Balanced Accuracy
#             balanced_accuracy = metrics.balanced_accuracy_score(test_labels, test_predictions)
            
#         return {
#             "rmse": rmse,
#             "r_squared": r_squared,
#             "adjusted_r_squared": adjusted_r_squared,
#             "directional_accuracy": directional_accuracy,
#             "f1_score": f1_score,
#             "roc_auc": roc_auc,
#             "balanced_accuracy": balanced_accuracy
#         }

    # from utils.util_functions import split_data_util, process_data_util
    # from configs.configs import get_data_processor_config
    # data_processor_config = get_data_processor_config()
    # from configs.configs import get_data_postprocessor_config
    # from data.data_preprocessing import DataPostProcessor
    # from model import LSTMBackBone
    # data_postprocessor_config = get_data_postprocessor_config()
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--split_percentage", type=float, default=data_processor_config.get("split_percentage", 0.2), help="Percentage of data to use for test split")
    # parser.add_argument("--split_data", type=bool, default=data_processor_config.get("split_data",True), help="Boolean flag to specify if the data should be split or not")
    # parser.add_argument("--process_data", type=bool, default=data_processor_config.get("process_data",True), help="Boolean flag to specify if the data should be processed or not")
    # parser.add_argument("--save_datapipelined", type=bool, default=data_processor_config.get("save_datapipelined",True), help="Boolean flag to specify if the datapipeline should be saved or not")
    # parser.add_argument("--transform_data", type=bool, default=data_processor_config.get("transform_data",True), help="Boolean flag to specify if the data should be transformed or not")
    # parser.add_argument("--save_transformed", type=bool, default=data_processor_config.get("save_transformed",True), help="Boolean flag to specify if the transformed data should be saved or not")
    # parser.add_argument("--datapipelined_path", type=str, default=data_processor_config.get("datapipelined_path",None), help="Path to save the datapipeline")
    # parser.add_argument("--save_transformed_path", type=str, default=data_processor_config.get("save_transformed_path",None), help="Path to save the transformed data")
    # parser.add_argument("--data_file_path", type=str, default=data_processor_config.get("data_file_path",'artifacts/data/raw/data.csv'), help="Path to the data file")
    # parser.add_argument("--left_split_save_path", type=str, default=data_processor_config.get("left_split_save_path","train_split.csv"), help="Path to save the left split of data")
    # parser.add_argument("--right_split_save_path", type=str, default=data_processor_config.get("right_split_save_path","test_split.csv"), help="Path to save the right split of data")
    # parser.add_argument("--save_split", type=bool, default=data_processor_config.get("save_split",False), help="Boolean flag to specify if the split files should be saved or not")

    # args = parser.parse_args()

    # split_percentage = args.split_percentage
    # split_data = args.split_data
    # process_data = args.process_data
    # save_datapipelined = args.save_datapipelined
    # transform_data = args.transform_data
    # save_transformed = args.save_transformed
    # datapipelined_path = args.datapipelined_path
    # save_transformed_path = args.save_transformed_path
    # data_file_path = args.data_file_path
    # left_split_save_path = args.left_split_save_path
    # right_split_save_path = args.right_split_save_path
    # save_split = args.save_split

    # ### Train Data Pipeline
    # data_postprocessor_config["split_data"] = False
    # data_postprocessor_config["save_split_data"] = False
    # data_postprocessor_config["data_pipelined_path"] = "artifacts/data_pipeline/data_pipeline_with_all_feat.pkl"
    # data_postprocessor_config["data_file_path"] = "artifacts/data/processed/train_split.csv"
    # processor_train = DataPostProcessor(**data_postprocessor_config)
    # processor_train.post_process_data(**data_postprocessor_config)

    # # Test Data Pipeline
    # data_postprocessor_config["split_data"] = False
    # data_postprocessor_config["save_split_data"] = False
    # data_postprocessor_config["data_pipelined_path"] = "artifacts/data_pipeline/data_pipeline_with_all_feat.pkl"
    # data_postprocessor_config["data_file_path"] = "artifacts/data/processed/test_split.csv"
    # processor_test = DataPostProcessor(**data_postprocessor_config)
    # processor_test.post_process_data(**data_postprocessor_config)
    
    # ### Train Data Pipeline
    # data_postprocessor_config["all_columns"] = all_columns_for_filtered_corr
    # data_postprocessor_config["split_data"] = False
    # data_postprocessor_config["save_split_data"] = False
    # data_postprocessor_config["data_pipelined_path"] = "artifacts/data_pipeline/data_pipeline_with_filtered_corr_feat.pkl"
    # data_postprocessor_config["data_file_path"] = "artifacts/data/processed/train_split.csv"
    # processor_train_with_filtered_corr_feat = DataPostProcessor(**data_postprocessor_config)
    # processor_train_with_filtered_corr_feat.post_process_data(**data_postprocessor_config)

    # # Test Data Pipeline
    # data_postprocessor_config["all_columns"] = all_columns_for_filtered_corr
    # data_postprocessor_config["split_data"] = False
    # data_postprocessor_config["save_split_data"] = False
    # data_postprocessor_config["data_pipelined_path"] = "artifacts/data_pipeline/data_pipeline_with_filtered_corr_feat.pkl"
    # data_postprocessor_config["data_file_path"] = "artifacts/data/processed/test_split.csv"
    # processor_test_with_filtered_corr_feat = DataPostProcessor(**data_postprocessor_config)
    # processor_test_with_filtered_corr_feat.post_process_data(**data_postprocessor_config)