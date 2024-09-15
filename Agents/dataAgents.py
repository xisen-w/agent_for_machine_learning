from Agents.code_writer import CodeWriter  # Corrected import path
from utils.response_generator import ResponseGenerator  # Add this import
import pandas as pd
import io

class DataAgents:
    def __init__(self, model: str = "gpt-4o"):
        # Initialize CodeWriter to reuse code generation and debugging functionality
        self.code_writer = CodeWriter(model=model)

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

        return self.code_writer.write_code(demand, knowledge_base + sample_code)

    def data_analyser(self, data_path: str) -> dict:
        demand = (
            "Write a function that performs exploratory data analysis (EDA) on the provided dataset. "
            "Utilize Pandas functions like describe(), info(), shape, etc., to generate a comprehensive analytical report. "
            "Include column names and provide a script to execute the function and return a structured quality report."
        )
        knowledge_base = f"Basic Python syntax and unit testing using the dataset from {data_path}."

        return self.code_writer.write_code(demand, knowledge_base)

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
        return self.code_writer.get_response(demand_prompt, demand + data_report)

    def advanced_data_analyser(self, data_path: str) -> dict:
        demand = (
            "Write a function that performs detailed exploratory data analysis (EDA) on the provided dataset. "
            "Utilize Pandas functions like describe(), info(), shape, and correlations to generate a comprehensive report. "
            "Include creative insights that could help inform model parameter settings and data splitting strategies."
        )
        knowledge_base = f"Basic Python syntax and unit testing using the dataset from {data_path}."

        return self.code_writer.write_code(demand, knowledge_base) 

