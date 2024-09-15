import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import io
from Agents.code_writer import CodeWriter  # Corrected import path
from utils.response_generator import ResponseGenerator  # Add this import


class DataAgents:
    def __init__(self, model: str = "gpt-4o"):
        # Initialize CodeWriter to reuse code generation and debugging functionality
        self.code_writer = CodeWriter(model=model)
        self.ResponseGenerator = ResponseGenerator(model=model)

    def data_cleaner(self, data_path: str) -> dict:
        demand = "Write a code cleaner that inputs a pandas dataframe and outputs the cleaned dataframe."
        knowledge_base = f"Basic Python syntax, unit testing in Python, use the data in {data_path} for unit test."
        
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

    def data_analyser(self, data_path: str) -> dict:
        demand = (
            "Write a function that performs exploratory data analysis (EDA) on the provided dataset. "
            "Utilize Pandas functions like describe(), info(), shape, etc., to generate a comprehensive analytical report. "
            "Include column names and provide a script to execute the function and return a structured quality report."
        )
        knowledge_base = f"Basic Python syntax and unit testing using the dataset from {data_path}."

        # Use the CodeWriter's advanced_writing_v2 function to ensure correctness
        return self.code_writer.advanced_writing_v2(demand, knowledge_base)

    def brutal_data_analyser(self, data_path: str) -> str:
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
            
            report = (
                f"Info:\n{info}\nShape: {shape}\nDescribe:\n{describe}\n"
                f"Columns: {columns}\nFirst Data Point: {first_data_point}\n"
                f"Missing Values: {missing_values}\nCorrelations:\n{correlations}\n"
            )
            return report
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def demand_analyser(self, demand: str, data_report: str) -> str:
        demand_prompt = (
            "You are a component of a framework that designs machine learning models. "
            "Based on the provided user demand and the data report, describe the label to predict and the features to use. "
            "Include numerical descriptions where applicable. Also, suggest model architectures and parameter settings."
        )
        return self.ResponseGenerator.get_response(demand_prompt, demand + data_report)

    def advanced_data_analyser(self, data_path: str) -> dict:
        demand = (
            "Write a function that performs detailed exploratory data analysis (EDA) on the provided dataset. "
            "Utilize Pandas functions like describe(), info(), shape, and correlations to generate a comprehensive report. "
            "Include creative insights that could help inform model parameter settings and data splitting strategies."
        )
        knowledge_base = f"Basic Python syntax and unit testing using the dataset from {data_path}."

        return self.code_writer.advanced_writing_v2(demand, knowledge_base)
    

if __name__ == "__main__":
    # Initialize DataAgents class with gpt-4o model
    data_agents = DataAgents(model="gpt-4o-mini")

    # Path to your test data
    test_data_path = "/Users/wangxiang/agent_for_prediction/datasets/social-media.csv"

    # Test data_cleaner method
    print("Testing Data Cleaner...")
    cleaned_data = data_agents.data_cleaner(test_data_path)
    print("Cleaned Data Output:\n", cleaned_data)

    # Test data_analyser method
    print("Testing Data Analyser...")
    analysis_report = data_agents.data_analyser(test_data_path)
    print("Data Analysis Report:\n", analysis_report)

    # Test brutal_data_analyser method
    print("Testing Brutal Data Analyser...")
    brutal_analysis = data_agents.brutal_data_analyser(test_data_path)
    print("Brutal Data Analysis:\n", brutal_analysis)

    # Test demand_analyser method
    test_demand = "Build a predictive model for success using relevant features."
    print("Testing Demand Analyser...")
    demand_analysis = data_agents.demand_analyser(test_demand, brutal_analysis)
    print("Demand Analysis:\n", demand_analysis)

    # Test advanced_data_analyser method
    print("Testing Advanced Data Analyser...")
    advanced_analysis = data_agents.advanced_data_analyser(test_data_path)
    print("Advanced Data Analysis:\n", advanced_analysis)

    import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import io
from pprint import pprint
from Agents.code_writer import CodeWriter  # Corrected import path
from utils.response_generator import ResponseGenerator  # Add this import

class DataAgents:
    def __init__(self, model: str = "gpt-4o"):
        # Initialize CodeWriter to reuse code generation and debugging functionality
        self.code_writer = CodeWriter(model=model)
        self.ResponseGenerator = ResponseGenerator(model=model)

    def data_cleaner(self, data_path: str) -> dict:
        demand = "Write a code cleaner that inputs a pandas dataframe and outputs the cleaned dataframe."
        knowledge_base = f"Basic Python syntax, unit testing in Python, use the data in {data_path} for unit test."
        
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

        return self.code_writer.advanced_writing_v2(demand, knowledge_base + sample_code)

    def data_analyser(self, data_path: str) -> dict:
        demand = (
            "Write a function that performs exploratory data analysis (EDA) on the provided dataset. "
            "Utilize Pandas functions like describe(), info(), shape, etc., to generate a comprehensive analytical report. "
            "Include column names and provide a script to execute the function and return a structured quality report."
        )
        knowledge_base = f"Basic Python syntax and unit testing using the dataset from {data_path}."
        return self.code_writer.advanced_writing_v2(demand, knowledge_base)

    def brutal_data_analyser(self, data_path: str) -> str:
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
            
            report = (
                f"Info:\n{info}\nShape: {shape}\nDescribe:\n{describe}\n"
                f"Columns: {columns}\nFirst Data Point: {first_data_point}\n"
                f"Missing Values: {missing_values}\nCorrelations:\n{correlations}\n"
            )
            return report
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def demand_analyser(self, demand: str, data_report: str) -> str:
        demand_prompt = (
            "You are a component of a framework that designs machine learning models. "
            "Based on the provided user demand and the data report, describe the label to predict and the features to use. "
            "Include numerical descriptions where applicable. Also, suggest model architectures and parameter settings."
        )
        return self.ResponseGenerator.get_response(demand_prompt, demand + data_report)

    def advanced_data_analyser(self, data_path: str) -> dict:
        demand = (
            "Write a function that performs detailed exploratory data analysis (EDA) on the provided dataset. "
            "Utilize Pandas functions like describe(), info(), shape, and correlations to generate a comprehensive report. "
            "Include creative insights that could help inform model parameter settings and data splitting strategies."
        )
        knowledge_base = f"Basic Python syntax and unit testing using the dataset from {data_path}."
        return self.code_writer.advanced_writing_v2(demand, knowledge_base)

    def data_block(self, parent_path, file_name, demand, model='gpt-4o-mini'):
        data_path = os.path.join(parent_path, file_name)
        print("\nSTART TO CLEAN----------------------START TO CLEAN\n")
        # First clean the data
        cleaned = self.data_cleaner(data_path)
        clean_code = cleaned[0][1]
        clean_test = cleaned[1][1]
        clean_running_code = f"{clean_code}\n{clean_test}"  # This is to run it
        print("\nHere is the code for cleaning\n")
        print(clean_running_code)

        self.code_writer.execute(clean_running_code)  # New Path of Cleaned Code

        print("\nSTART TO ANALYZE----------------------START TO ANALYZE\n")
        # Second analyze the data
        new_path = os.path.join(parent_path, "cleaned_" + file_name)
        res = self.advanced_data_analyser(new_path, model)
        code = res[0][1]
        test = res[1][1]
        running_code = f"{code}\n{test}"  # This is to run it

        print(running_code)
        data_report = self.code_writer.execute(running_code)
        print("\n---------------DATA REPORT-------------------\n")
        print(data_report)

        # Third, transform the demand
        print("\nSTART TO TRANSFORM DEMAND----------------------START TO TRANSFORM DEMAND\n")
        model_specifications = self.demand_analyser(demand, data_report)
        pprint(model_specifications)
        return model_specifications, new_path, data_report


if __name__ == "__main__":
    # Initialize DataAgents class with gpt-4o model
    data_agents = DataAgents(model="gpt-4o-mini")

    # Path to your test data
    test_data_path = "/Users/wangxiang/agent_for_prediction/datasets/social-media.csv"

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