import pandas as pd
from datasets import load_dataset
import ast
import contextlib
import io
import traceback
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Agents.code_writer import CodeWriter

# Function to execute code
def execute_code(code: str) -> str:
    """Executes the given Python code and returns any output or error."""
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code)
        return output.getvalue().strip()
    except Exception as e:
        return traceback.format_exc()

# Function to evaluate the test cases for the generated code
def evaluate_test_list(code: str, test_list: list, test_setup_code: str = '') -> str:
    """Evaluates the generated code by running the provided test cases."""
    try:
        if test_setup_code:
            exec(test_setup_code)

        exec(code)  # Execute the generated code

        for test in test_list:  # Run each test
            exec(test)

        return "Correct"  # If all tests pass
    except Exception as e:
        return f"Error: {str(e)}"

# Load dataset and sample tasks
def load_sampled_tasks(sample_size=50): 
    """Loads the MBPP dataset and samples tasks."""
    ds = load_dataset("google-research-datasets/mbpp", "full")
    df = pd.DataFrame(ds['train'])
    sampled_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(sampled_df.head())  # Display sample for verification
    return sampled_df

# Function to evaluate tasks using different methods
def evaluate_tasks(sampled_df):
    """Evaluates the tasks using the write_code, advanced_writing, and advanced_writing_v2 methods."""
    # Initialize the CodeWriter class (assume it's defined elsewhere)
    code_writer = CodeWriter()

    results = []
    for index, row in sampled_df.iterrows():
        task_id = row['task_id']
        prompt = row['text']
        ground_truth_code = row['code']
        # Check if test_list is a string that needs evaluation
        if isinstance(row['test_list'], str):
            try:
                test_list = ast.literal_eval(row['test_list']) if row['test_list'] else []
            except Exception as e:
                print(f"Error evaluating test_list for task_id {task_id}: {e}")
                test_list = []  # Fallback to an empty list on error
        else:
            test_list = row['test_list']  # Assume it's already a list

        test_setup_code = row['test_setup_code']

        print(f"Evaluating Task {index + 1}/{len(sampled_df)}: {prompt}")

        # Evaluate with write_code
        write_code_response = code_writer.write_code(prompt, ground_truth_code)
        write_code_correctness = evaluate_test_list(write_code_response.code, test_list, test_setup_code)

        # Evaluate with advanced_writing
        advanced_writing_response = code_writer.advanced_writing(prompt, ground_truth_code)
        advanced_writing_correctness = evaluate_test_list(advanced_writing_response.code, test_list, test_setup_code)

        # Evaluate with advanced_writing_v2
        advanced_writing_v2_response = code_writer.advanced_writing_v2(prompt, ground_truth_code)
        advanced_writing_v2_correctness = evaluate_test_list(advanced_writing_v2_response.code, test_list, test_setup_code)

        # Collect results
        results.append({
            "task_id": task_id,
            "prompt": prompt,
            "ground_truth_code": ground_truth_code,
            "test_list": test_list,
            "write_code_output": write_code_response.code,
            "write_code_correctness": write_code_correctness,
            "write_code_comment": write_code_response.comment,
            "advanced_writing_output": advanced_writing_response.code,
            "advanced_writing_correctness": advanced_writing_correctness,
            "advanced_writing_comment": advanced_writing_response.comment,
            "advanced_writing_v2_output": advanced_writing_v2_response.code,
            "advanced_writing_v2_correctness": advanced_writing_v2_correctness,
            "advanced_writing_v2_comment": advanced_writing_v2_response.comment
        })

        print(f"Task {task_id} evaluation complete.\n")

    return results

# Save the results to a CSV file
def save_results_to_csv(results, filename="mbpp_evaluation_results.csv"):
    df_results = pd.DataFrame(results)
    df_results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    # Load sampled tasks
    sampled_df = load_sampled_tasks()

    # Evaluate the tasks
    evaluation_results = evaluate_tasks(sampled_df)

    # Save the evaluation results
    save_results_to_csv(evaluation_results)

    # 