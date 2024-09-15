import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import io
import contextlib
import traceback
from pydantic import BaseModel, Field
from utils.response_generator import ResponseGenerator  # Ensure this import is correct
import subprocess
import tempfile

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class CodeWriter:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.response_generator = ResponseGenerator(model=self.model)  # Initialize ResponseGenerator

    class CodeOut(BaseModel):
        code: str = Field(description="The generated code")
        test: str = Field(description="Test cases for the generated code")
        comment: str = Field(description="Explanation of the generated code")

    def write_code(self, demand: str, knowledge_base: str) -> CodeOut:
        sys_prompt = '''
        You are an advanced software engineer that writes useful and correct code.
        Given a specific demand and a knowledge base, generate code and a script to run it.
        Provide the code, test cases, and explanation for the code in JSON format.
        '''
        usr_prompt = f"Demand: {demand}\nKnowledge Base: {knowledge_base}\n"
        code_out = self.response_generator.get_structured_response(sys_prompt, usr_prompt, self.CodeOut)  # Use structured response
        return code_out  # Return the CodeOut instance

    def debug(self, code: str, error: str) -> CodeOut:
        usr_prompt = f"Code:\n{code}\n\nError Description:\n{error}\n"
        sys_prompt = '''
        You are a coding assistant with expertise in software engineering.
        Debug the code and provide the corrected code, test cases, and an explanation.
        '''
        code_out = self.response_generator.get_structured_response(sys_prompt, usr_prompt, self.CodeOut)  # Use structured response
        return code_out  # Return the CodeOut instance

    @staticmethod
    def test_coder(code: str, test: str) -> str:
        combined_code = f"{code}\n{test}"
        result = CodeWriter.execute_code(combined_code)
        if result.startswith("Errors:"):
            return result
        return "True"

    @staticmethod
    def execute_code(code: str) -> str:
        """
        Executes the provided code by writing it to a temporary file and running it.

        Parameters:
            code (str): The code to execute.

        Returns:
            str: The output of the executed code or an error message.
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as temp_file:
            temp_file.write(code.encode('utf-8'))  # Write the code to the file
            temp_file_path = temp_file.name  # Get the file path

        try:
            # Execute the file
            result = subprocess.run(['python', temp_file_path], capture_output=True, text=True)
            output = result.stdout
            errors = result.stderr

            if errors:
                return f"Errors:\n{errors}"
            return output
        except Exception as e:
            return f"An error occurred while executing the file: {str(e)}"
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def advanced_writing(self, demand: str, knowledge_base: str) -> CodeOut:
        code_out = self.write_code(demand, knowledge_base)
        state = self.test_coder(code_out.code, code_out.test)
        if state == "True":
            return code_out
        return self.debug(code_out.code, state)

    def advanced_writing_v2(self, demand: str, knowledge_base: str, max_iter: int = 6) -> CodeOut:
        context = ""
        code_out = self.write_code(demand, knowledge_base)
        code, test, comment = code_out.code, code_out.test, code_out.comment
        i = 0
        state = self.test_coder(code, test)

        print(f"Initial Code:\n{code}\n")
        print(f"Initial Test Cases:\n{test}\n")
        final_output = "INITIATION"

        while state != "True" and i < max_iter:
            print(f"Iteration {i + 1}:")
            print(f"Error Notice: {state}")
            print(f"Previous Code Explanation: {comment}\n")
            
            context += f"Iteration {i}: Error Notice: {state}. Previous code explanation: {comment}\n"
            debug_output = self.debug(f"{code}\n{test}", context)
            code, test, comment = debug_output.code, debug_output.test, debug_output.comment
            i += 1
            state = self.test_coder(code, test)
            final_output = self.execute_code(f"{code}\n{test}")

            print(f"Updated Code:\n{code}\n")
            print(f"Updated Test Cases:\n{test}\n")

        print(f"Final Code after {i} iterations:\n{code}\n")
        print(f"Final Test Cases:\n{test}\n")
        print(f"Final Output:\n{final_output}")
        print(f"Final Comment:\n{comment}\n")

        return self.CodeOut(code=code, test=test, comment=comment)

if __name__ == "__main__":
    # Test the CodeWriter class
    cw = CodeWriter()

    # Test write_code method
    demand = "Write a function to calculate the factorial of a number."
    knowledge_base = "The function should handle both positive and negative integers."
    print("Testing write_code:")
    response = cw.write_code(demand, knowledge_base)
    print(response)

    # Test debug method
    code_with_error = "def factorial(n): return n * factorial(n - 1)"  # Missing base case
    error_description = "RecursionError: maximum recursion depth exceeded"
    print("\nTesting debug:")
    debug_response = cw.debug(code_with_error, error_description)
    print(debug_response)

    # Test advanced_writing method
    print("\nTesting advanced_writing:")
    advanced_demand = "Create a Python function to reverse a string."
    advanced_knowledge_base = "The function should handle empty strings and single-character strings."
    advanced_response = cw.advanced_writing(advanced_demand, advanced_knowledge_base)
    print(advanced_response)

    # Test advanced_writing_v2 method
    print("\nTesting advanced_writing_v2:")
    advanced_v2_demand = "Write a function to sort a list of integers."
    advanced_v2_knowledge_base = "The function should sort the list in ascending order."
    advanced_v2_response = cw.advanced_writing_v2(advanced_v2_demand, advanced_v2_knowledge_base)
    print(advanced_v2_response)

    # You can add more tests for other methods as needed