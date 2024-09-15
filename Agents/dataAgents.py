import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import io
from Agents.code_writer import CodeWriter  # Corrected import path
from utils.response_generator import ResponseGenerator  # Add this import
import numpy as np

class DataAgents:
    def __init__(self, model: str = "gpt-4o-mini"):
        # Initialize CodeWriter to reuse code generation and debugging functionality
        self.code_writer = CodeWriter(model=model)
        self.ResponseGenerator = ResponseGenerator(model=model)


    def data_cleaner(self, data_path: str) -> dict:
        demand = "Write a code cleaner that inputs a pandas dataframe and outputs the cleaned dataframe."
        knowledge_base = f"""Basic Python syntax, unit testing in Python, use the data in {data_path} for unit test. NOTE. Make sure that the output is {data_path}_cleaned.csv This format is very very important. 
        For the code, do not write: if __name__ == '__main__': Do not use unittest. Instead, use the data_path for testing for realistic performance. Make sure to apply the functions on the dataset itself."""
        
        # Sample code for cleaning
        sample_code = '''
        import pandas as pd
        def clean_dataframe(df):
            df = df.drop_duplicates()
            df = df.dropna()
            df = df.reset_index(drop=True)
            return df
        df = pd.read_csv('{}')
        df_clean = clean_dataframe(df)
        df_clean.to_csv('{}', index=False)
        '''.format(data_path, data_path.replace(".csv", "_cleaned.csv"))

        # Use the CodeWriter's advanced_writing_v2 function to ensure correctness
        return self.code_writer.advanced_writing_v2(demand, knowledge_base + sample_code)
    
    def brutal_data_cleaner(self, data_path: str) -> str:
        """
        A versatile data cleaning function that handles a wide range of common data issues.
        It reads the data, applies multiple cleaning techniques, and writes the cleaned data back to a file.

        Parameters:
            data_path (str): The path to the CSV file that needs to be cleaned.

        Returns:
            str: A summary of the cleaning process, including information about the cleaned data.
        """
        
        try:
            # Load the dataset
            df = pd.read_csv(data_path)

            # Log original shape
            original_shape = df.shape

            # Step 1: Drop duplicate rows
            df = df.drop_duplicates()

            # Step 2: Handle missing values
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].median(), inplace=True)
                elif df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)

            # Step 3: Detect and handle outliers (for numeric columns only)
            for col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            # Step 4: Standardize string columns
            df = df.apply(lambda col: col.str.strip().str.lower() if col.dtype == 'object' else col)

            # Step 5: Create new features (optional)
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col + '_year'] = pd.DatetimeIndex(df[col]).year

            # Step 6: Convert date columns to standard datetime format
            for col in df.columns:
                if 'date' in col.lower() or pd.api.types.is_string_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            # Step 7: Reset the index after cleaning
            df = df.reset_index(drop=True)

            # Save the cleaned dataset
            cleaned_file_path = data_path.replace(".csv", "_cleaned.csv")
            df.to_csv(cleaned_file_path, index=False)

            # Log final shape
            cleaned_shape = df.shape

            # Return the cleaning summary as a string
            cleaning_summary = (
                f"Cleaning Summary:\n"
                f"Original Shape: {original_shape}\n"
                f"Cleaned Shape: {cleaned_shape}\n"
                f"Cleaned File Path: {cleaned_file_path}\n"
                f"Steps Applied:\n"
                f"- Dropped duplicates\n"
                f"- Handled missing values\n"
                f"- Removed outliers using IQR method\n"
                f"- Standardized string columns\n"
                f"- Converted date columns\n"
                f"- Reset index\n"
            )
            return cleaning_summary
        
        except FileNotFoundError:
            return "Error: The file at path {} was not found.".format(data_path)
        
        except pd.errors.EmptyDataError:
            return "Error: The provided file is empty."
        
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

    def data_analyser(self, data_path: str) -> dict: #TODO: Pandas doesn't work. 
        demand = (
            "Write a function that performs exploratory data analysis (EDA) on the provided dataset called perform_eda(). "
            "Utilize Pandas functions like describe(), info(), shape, etc., to generate a comprehensive analytical report. "
            "Include column names and provide a script to execute the function and return a structured quality report. Make sure to apply the functions on the dataset itself. * Use the correct path in testing, instead of unitest."
        )
        knowledge_base = f"Basic Python syntax and unit testing using the dataset from {data_path}. Do not write: if __name__ == '__main__':"

        # Use the CodeWriter's advanced_writing_v2 function to ensure correctness
        return self.code_writer.advanced_writing_v2(demand, knowledge_base)

    def brutal_data_analyser(self, data_path: str) -> str:
        """
        Performs exploratory data analysis (EDA) on the provided dataset.

        Parameters:
            data_path (str): The path to the CSV file that needs to be analyzed.

        Returns:
            str: A summary report of the analysis, including data types, missing values, and correlations.
        """
        try:
            df = pd.read_csv(data_path)
            info_buffer = io.StringIO()
            df.info(buf=info_buffer)
            info = info_buffer.getvalue()
            shape = df.shape
            describe = df.describe(include='all').to_string()
            columns = df.columns.tolist()
            first_data_point = df.iloc[0].to_dict() if not df.empty else {}
            missing_values = df.isnull().sum().to_dict()
            correlations = df.corr().to_string() if not df.empty else "No numeric columns to correlate."
            
            # Create the report as a string
            report = (
                f"--- DATAFRAME SUMMARY ---\n"
                f"Info:\n{info}\n"
                f"Shape: {shape}\n"
                f"Describe:\n{describe}\n"
                f"Columns: {columns}\n"
                f"First Data Point: {first_data_point}\n"
                f"Missing Values: {missing_values}\n"
                f"Correlations:\n{correlations}\n"
            )
            return report
        except Exception as e:
            return f"An error occurred: {str(e)}"
        

    def brutal_data_analyser(self, data_path: str) -> str:
        try:
            # Load the dataset
            df = pd.read_csv(data_path)
            
            # Buffer to capture DataFrame info output
            info_buffer = io.StringIO()
            df.info(buf=info_buffer)
            info = info_buffer.getvalue()

            # Identify categorical and numerical columns
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

            # General dataframe info
            shape = df.shape
            describe_numerical = df.describe(include=['number']).to_string() if numerical_columns else "No numerical columns available."
            describe_categorical = df.describe(include=['object', 'category']).to_string() if categorical_columns else "No categorical columns available."
            columns = df.columns.tolist()
            first_data_point = df.iloc[0].to_dict() if not df.empty else "DataFrame is empty."
            missing_values = df.isnull().sum().to_dict()

            # Enhanced analysis for string/categorical fields
            string_summary = {col: {
                "unique": df[col].nunique(),
                "top": df[col].mode()[0] if not df[col].mode().empty else None,
                "freq": df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0,
                "missing": df[col].isnull().sum(),
                "missing_percentage": (df[col].isnull().mean() * 100)
            } for col in categorical_columns}

            # Correlation analysis for numeric columns
            if numerical_columns:
                numeric_df = df[numerical_columns]  # Ensure only numeric columns are used
                correlations = numeric_df.corr().to_string()
            else:
                correlations = "No numeric columns to correlate."

            # Enhanced handling for skewness and outliers in numerical data
            skewness = {col: df[col].skew() for col in numerical_columns}
            outliers_summary = {col: {
                "min": df[col].min(),
                "max": df[col].max(),
                "q1": df[col].quantile(0.25),
                "q3": df[col].quantile(0.75),
                "iqr": df[col].quantile(0.75) - df[col].quantile(0.25),
                "outliers_count": df[(df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) | 
                                    (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))))].shape[0]
            } for col in numerical_columns}

            report = (
                f"--- DATAFRAME SUMMARY ---\n"
                f"Info:\n{info}\n"
                f"Shape: {shape}\n"
                f"Columns: {columns}\n"
                f"Missing Values:\n{missing_values}\n\n"
                f"--- NUMERICAL DATA SUMMARY ---\n"
                f"Numerical Describe:\n{describe_numerical}\n"
                f"Skewness:\n{skewness}\n"
                f"Outliers Summary:\n{outliers_summary}\n"
                f"Correlations:\n{correlations}\n\n"
                f"--- CATEGORICAL DATA SUMMARY ---\n"
                f"Categorical Describe:\n{describe_categorical}\n"
                f"String/Categorical Field Summary:\n{string_summary}\n"
                f"First Data Point: {first_data_point}\n"
            )
            
            return report

        except pd.errors.EmptyDataError:
            return "The provided file is empty. Please check the data file."
        
        except FileNotFoundError:
            return f"The file at path {data_path} could not be found. Please check the file path."
        
        except pd.errors.ParserError:
            return "There was an error parsing the file. Ensure the file is in a valid format (e.g., CSV)."
        
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

    def demand_analyser(self, demand: str, data_report: str) -> str:
        demand_prompt = (
            "You are a component of a framework that designs machine learning models. "
            "Based on the provided user demand and the data report, describe the label to predict and the features to use. "
            "Include numerical descriptions where applicable. Also, suggest model architectures and parameter settings."
        )
        return self.ResponseGenerator.get_response(demand_prompt, "Demand: "+ demand + "DATA Report:" + data_report)

    def advanced_data_analyser(self, data_path: str):
        demand = (
            "Write a function that performs detailed exploratory data analysis (EDA) on the provided dataset. "
            "Utilize Pandas functions like describe(), info(), shape, and correlations to generate a comprehensive report. "
            "Include creative insights that could help inform model parameter settings and data splitting strategies.  Make sure to apply the functions on the dataset itself."
        )
        knowledge_base = f"Basic Python syntax and unit testing using the dataset from {data_path}. Do not write: if __name__ == '__main__':"

        return self.code_writer.advanced_writing_v2(demand, knowledge_base)
    
    def data_block(self, parent_path, file_name, demand, model='gpt-4o-mini'):
        data_path = os.path.join(parent_path, file_name)
        # print("\nSTART TO CLEAN----------------------START TO CLEAN\n")
        # # First clean the data
        # code_out = self.data_cleaner(data_path)
        # clean_code = code_out.code
        # clean_test = code_out.test
        # clean_running_code = f"{clean_code}\n{clean_test}"  # This is to run it
        # print("\nHere is the code for cleaning\n")
        # print(clean_running_code)

        cleaning_summary = self.brutal_data_cleaner(data_path)

        # self.code_writer.execute_code(clean_running_code)  # New Path of Cleaned Code
        print(cleaning_summary)
        print("\nSTART TO ANALYZE----------------------START TO ANALYZE\n")
        # Second analyze the data
        cleaned_file_path = data_path.replace(".csv", "_cleaned.csv")
        # analysis = self.advanced_data_analyser(new_path)
        # code = analysis.code
        # test = analysis.test
        # running_code = f"{code}\n{test}"  # This is to run it 
        # print(running_code)
        # data_report = self.code_writer.execute_code(running_code)
        # print("\n---------------DATA REPORT-------------------\n")
        # print(data_report)
        data_report = self.brutal_data_analyser(cleaned_file_path)
        print(data_report)

        data_info = f"We have first cleaned it: {cleaning_summary}. Very important information of the data (please follow strictly): {data_report}"
        # Third, transform the demand
        print("\nSTART TO TRANSFORM DEMAND----------------------START TO TRANSFORM DEMAND\n")
        model_specifications = self.demand_analyser(demand, data_report)
        print(model_specifications)
        return model_specifications, cleaned_file_path, data_report


if __name__ == "__main__":
    # Initialize DataAgents class with gpt-4o model
    data_agents = DataAgents(model="gpt-4o-mini")

    # Path to your test data
    test_data_path = "/Users/wangxiang/agent_for_prediction/datasets/social-media.csv"

    # Test data_cleaner method
    # print("Testing Data Cleaner...")
    # cleaned_data = data_agents.data_cleaner(test_data_path)
    # print("Cleaned Data Output:\n", cleaned_data)

    # # Test data_analyser method
    # print("Testing Data Analyser...")
    # analysis_report = data_agents.data_analyser(test_data_path)
    # print("Data Analysis Code:\n", analysis_report)

    # # Test brutal_data_analyser method
    # print("Testing Brutal Data Analyser...")
    # brutal_analysis = data_agents.brutal_data_analyser(test_data_path)
    # print("Brutal Data Analysis:\n", brutal_analysis)

    # # Test demand_analyser method
    # test_demand = "Build a predictive model for success using relevant features."
    # print("Testing Demand Analyser...")
    # demand_analysis = data_agents.demand_analyser(test_demand, brutal_analysis)
    # print("Demand Analysis:\n", demand_analysis)

    # # Test advanced_data_analyser method
    # print("Testing Advanced Data Analyser...")
    # advanced_analysis = data_agents.advanced_data_analyser(test_data_path)
    # print("Advanced Data Analysis:\n", advanced_analysis)

    # Test data_block method
    print("Testing Data Block...")
    model_specs, new_cleaned_path, data_report = data_agents.data_block(
        parent_path="/Users/wangxiang/agent_for_prediction/datasets", 
        file_name="social-media.csv", 
        demand="Predict success metrics"
    )
    print("Model Specifications:\n", model_specs)
    print("Cleaned Data Path:", new_cleaned_path)
    print("Data Report:\n", data_report)