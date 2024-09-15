import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import io
from Agents.code_writer import CodeWriter  # Ensure this import is correct
from utils.response_generator import ResponseGenerator  # Ensure this import is correct

class ModelDesigner:
    def __init__(self, model: str = "gpt-4o"):
        self.code_writer = CodeWriter(model=model)
        self.response_generator = ResponseGenerator(model=model)

    def model_designer(self, specification: dict, data_path: str) -> str:
        """
        Generates functional code that trains a prediction model based on provided specifications and data.

        Parameters:
            specification (dict): Model specifications including target features and model type preferences.
            data_path (str): The path to the dataset file, used for both training the model and unit testing.

        Returns:
            str: Generated code for data processing, model training, and saving the trained model.
        """
        demand = (
            "You are a professional model designer and trainer. Write functional code that can train a prediction model. "
            "Choose from Random Forest, XGBoost, OLSR, Neural Network, CNN, and LSTM, selecting the model that best fits "
            "the task with an emphasis on simplicity for stability. Determine appropriate parameters and architecture. "
            "Ensure the code is robust and production-ready."
        )
        
        knowledge_base = (
            f"Understand basic Python syntax, unit testing in Python. Use the data at {data_path} for unit tests. "
            "Ensure the trained model is saved in the same directory with the filename 'pred_model.pkl'. "
            f"Specifications: {specification}"
        )

        # Use the CodeWriter's advanced_writing_v2 function to ensure correctness
        return self.code_writer.advanced_writing_v2(demand, knowledge_base)

if __name__ == "__main__":
    # Example usage of ModelDesigner
    model_designer = ModelDesigner(model="gpt-4o-mini")

    # Example specification and data path
    example_specification = '''
    odel Specifications:
 ### Label to Predict
The label to predict is **TotalLikes**. This is a continuous variable representing the total number of likes a user has received, which can be considered a success metric for user engagement.

### Features to Use
Based on the provided data, the following features can be utilized for the prediction model:

1. **UsageDuration**: This is the duration of usage by the user, measured in some time unit (likely hours or minutes). It has a mean of approximately 3.42 and a standard deviation of 2.28.
   
2. **Age**: The age of the user, which has a mean of approximately 35.42 and a standard deviation of 15.44. This feature may influence user engagement and preferences.

3. **Country**: This feature is currently not usable as it contains no valid entries (all values are NaN). If this data can be obtained or imputed, it could provide valuable demographic insights.

4. **UserId**: While this is a unique identifier for each user, it may not be useful as a feature for prediction since it does not provide any meaningful information about user behavior or characteristics.

### Suggested Model Architectures
Given the nature of the problem (regression), the following model architectures can be considered:

1. **Linear Regression**: A simple model that can provide a baseline for performance. It assumes a linear relationship between the features and the target variable.

2. **Random Forest Regressor**: This ensemble method can capture non-linear relationships and interactions between features. It is robust to overfitting and can handle the small dataset size.

3. **Gradient Boosting Regressor**: Another powerful ensemble method that builds trees sequentially, focusing on correcting the errors of the previous trees. It can provide high accuracy with proper tuning.

4. **Neural Network**: A simple feedforward neural network with one or two hidden layers can be used if the dataset is expanded or if more features are added in the future.

### Parameter Settings
For the suggested models, here are some initial parameter settings:

1. **Linear Regression**: No specific parameters to tune, but ensure to check for multicollinearity.

2. **Random Forest Regressor**:
   - `n_estimators`: 100 (number of trees)
   - `max_depth`: None (allow trees to grow fully)
   - `min_samples_split`: 2 (minimum samples required to split an internal node)
   - `min_samples_leaf`: 1 (minimum samples required to be at a leaf node)

3. **Gradient Boosting Regressor**:
   - `n_estimators`: 100 (number of boosting stages)
   - `learning_rate`: 0.1 (contributes to the model's learning speed)
   - `max_depth`: 3 (depth of the individual trees)
   - `subsample`: 0.8 (fraction of samples to be used for fitting the individual base learners)

4. **Neural Network**:
   - Input layer: 3 neurons (for UsageDuration, Age, and possibly Country if imputed)
   - Hidden layers: 1-2 layers with 10-20 neurons each
   - Activation function: ReLU
   - Output layer: 1 neuron (for TotalLikes)
   - Loss function: Mean Squared Error (MSE)
   - Optimizer: Adam

### Conclusion
The model should be evaluated using metrics such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE) to assess its performance. Cross-validation should also be employed to ensure the model's robustness given the small dataset size.
Cleaned Data Path: /Users/wangxiang/agent_for_prediction/datasets/social-media_cleaned.csv
Data Report:
 --- DATAFRAME SUMMARY ---
Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 57 entries, 0 to 56
Data columns (total 5 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   UserId         57 non-null     int64  
 1   UsageDuraiton  57 non-null     int64  
 2   Age            57 non-null     int64  
 3   Country        0 non-null      float64
 4   TotalLikes     57 non-null     int64  
dtypes: float64(1), int64(4)
memory usage: 2.4 KB

Shape: (57, 5)
Columns: ['UserId', 'UsageDuraiton', 'Age', 'Country', 'TotalLikes']
Missing Values:
{'UserId': 0, 'UsageDuraiton': 0, 'Age': 0, 'Country': 57, 'TotalLikes': 0}

--- NUMERICAL DATA SUMMARY ---
Numerical Describe:
          UserId  UsageDuraiton        Age  Country  TotalLikes
count  57.000000      57.000000  57.000000      0.0   57.000000
mean   31.000000       3.421053  35.421053      NaN    3.596491
std    17.430474       2.283069  15.443384      NaN    2.782982
min     1.000000       1.000000  18.000000      NaN    0.000000
25%    17.000000       2.000000  20.000000      NaN    1.000000
50%    31.000000       3.000000  28.000000      NaN    3.000000
75%    46.000000       5.000000  51.000000      NaN    5.000000
max    62.000000       9.000000  60.000000      NaN   10.000000
Skewness:
{'UserId': -0.03245777487637989, 'UsageDuraiton': 0.9409098335209461, 'Age': 0.2190959734891967, 'Country': nan, 'TotalLikes': 0.6763856891120883}
Outliers Summary:
{'UserId': {'min': 1, 'max': 62, 'q1': 17.0, 'q3': 46.0, 'iqr': 29.0, 'outliers_count': 0}, 'UsageDuraiton': {'min': 1, 'max': 9, 'q1': 2.0, 'q3': 5.0, 'iqr': 3.0, 'outliers_count': 0}, 'Age': {'min': 18, 'max': 60, 'q1': 20.0, 'q3': 51.0, 'iqr': 31.0, 'outliers_count': 0}, 'Country': {'min': nan, 'max': nan, 'q1': nan, 'q3': nan, 'iqr': nan, 'outliers_count': 0}, 'TotalLikes': {'min': 0, 'max': 10, 'q1': 1.0, 'q3': 5.0, 'iqr': 4.0, 'outliers_count': 0}}
Correlations:
                 UserId  UsageDuraiton       Age  Country  TotalLikes
UserId         1.000000      -0.383214 -0.064082      NaN   -0.198786
UsageDuraiton -0.383214       1.000000 -0.615410      NaN    0.370098
Age           -0.064082      -0.615410  1.000000      NaN   -0.464647
Country             NaN            NaN       NaN      NaN         NaN
TotalLikes    -0.198786       0.370098 -0.464647      NaN    1.000000

--- CATEGORICAL DATA SUMMARY ---
Categorical Describe:
No categorical columns available.
String/Categorical Field Summary:
{}
First Data Point: {'UserId': 1.0, 'UsageDuraiton': 2.0, 'Age': 55.0, 'Country': nan, 'TotalLikes': 5.0}
    '''
    example_data_path = "/Users/wangxiang/agent_for_prediction/datasets/social-media_cleaned.csv"

    # Generate model code
    generated_code = model_designer.model_designer(example_specification, example_data_path)
    print("Generated Model Code:\n", generated_code)