from code_writer import CodeWriter  # Import the CodeWriter class
import os
import requests
import base64
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import utils

class Assessor:
    def __init__(self, model: str = "gpt-4o-mini"):
        # Initialize CodeWriter to reuse code generation and debugging functionality
        self.code_writer = CodeWriter(model=model)

    def assessor(self, pred_model_path: str, data_path: str, model_code: str) -> dict:
        """
        Generates code to assess the performance of a trained machine learning model on a provided dataset.

        Parameters:
            pred_model_path (str): The file path where the trained prediction model is stored.
            data_path (str): The path to the dataset used for evaluating the model.
            model_code (str): The code used to train the model.

        Returns:
            dict: A dictionary containing Python code to evaluate the model's accuracy, precision, recall, and F1-score.
        """
        demand = (
            "Write Python code that loads a trained machine learning model from a specified path, applies it to a dataset, "
            "and computes performance metrics such as accuracy, precision, recall, and F1-score. Include comments within "
            "the code that explain each step. Save the evaluation results to a text file in the same directory as the model."
        )
        knowledge_base = (
            f"Basic Python syntax, data analysis libraries (e.g., Pandas, scikit-learn), and file handling. Use the model "
            f"stored at {pred_model_path} and data at {data_path} to demonstrate functionality in the generated code. "
            f"For context, this is how the model was trained: {model_code}"
        )

        # Use the CodeWriter to generate the code
        evaluation_code = self.code_writer.write_code(demand, knowledge_base)
        return evaluation_code

    def plotter(self, model_code: str, pred_model_path: str, data_path: str, data_report: str) -> dict:
        """
        Generates code to plot various graphs related to model evaluation.

        Parameters:
            model_code (str): The code that initializes and runs the model.
            pred_model_path (str): Path where the trained model is saved.
            data_path (str): Path to the dataset used for plotting.
            data_report (str): Data report that provides context for the model evaluation.

        Returns:
            dict: A dictionary containing Python code to generate plots and save them as PNGs.
        """
        demand = (
            "Generate Python code to plot the training history, classification results, model architecture, "
            "and factor importance of a trained model. Include steps to load the model, apply it to the dataset, "
            "and visualize the results using appropriate libraries like matplotlib or seaborn. "
            f"Save every plot in the same directory as the model and dataset."
        )
        knowledge_base = (
            f"Advanced Python plotting libraries, data manipulation with Pandas, model handling. "
            f"Use paths {pred_model_path} and {data_path}. Here is how we trained the model: {model_code}. "
            f"Also, take into account the data report: {data_report}."
        )

        # Use the CodeWriter to generate the plotting code
        plotting_code = self.code_writer.advanced_writing_v2(demand, knowledge_base)
        return plotting_code

    def inference_writer(self, model_code: str, pred_model_path: str, data_path: str, data_report: str) -> dict:
        """
        Generates code for model inference, particularly for handling single data instances.

        Parameters:
            model_code (str): The code for the model setup and prediction.
            pred_model_path (str): Path where the trained model is saved.
            data_path (str): Path to the dataset used for testing inference.
            data_report (str): Report that summarizes the dataset used.

        Returns:
            dict: A dictionary containing Python code for making inferences with the trained model.
        """
        demand = (
            "Write Python code to adjust data forms to make sure the model can handle inference for a single data instance. "
            "Encapsulate everything as a function `make_prediction()` that accepts a single data instance and returns the prediction."
        )
        knowledge_base = (
            f"Basic Python syntax, data preprocessing with Pandas or NumPy, model loading and inference techniques. "
            f"Use the model stored at {pred_model_path} and data from {data_path}. Here's how we trained the model: {model_code}. "
            f"Here's a data report of the dataset: {data_report}."
        )

        # Use the CodeWriter to generate the inference code
        inference_code = self.code_writer.write_code(demand, knowledge_base)
        return inference_code

    def html_model_reporter(self, accuracy_report: str, plots_paths: str, inference_code: str, plot_description: str, data_report: str, specifications: str) -> str:
        """
        Generates HTML code to document how the model was trained and how it can be used.

        Parameters:
            accuracy_report (str): Text or path to the accuracy and other performance metrics report.
            plots_paths (str): Paths to the generated plots.
            plot_description (str): Descriptions of the plots.
            inference_code (str): Code snippet for model inference.
            specifications (str): Specifications on how the model should be.

        Returns:
            str: Generated HTML code summarizing the model training and usage.
        """
        demand = (
            "Create HTML code that summarizes the training and application of a machine learning model, starting from what the data looks like and what the aim is."
            "Include sections for the accuracy report, visualizations, and how to use the model with the provided inference code. "
            "Ensure the report is detailed and easy to follow for readers unfamiliar with AI."
        )
        knowledge_base = (
            f"HTML and CSS for styling, integration of various media types into HTML documents. Accuracy Report: {accuracy_report}, "
            f"Code for Inference: {inference_code}, What the plots say: {plot_description}, Data used: {data_report}, Model Specifications: {specifications}, Plot Paths: {plots_paths}."
        )

        # Use the CodeWriter to generate the HTML code
        html_code = self.code_writer.get_response(demand, knowledge_base)
        return html_code

    def graph_reader(self, plot_path):
        invoke_url = "https://ai.api.nvidia.com/v1/vlm/google/deplot"
        stream = True

        # Read and encode the image
        with open(plot_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        assert len(image_b64) < 180_000, "To upload larger images, use the assets API (see docs)"

        headers = {
            "Authorization": "Bearer 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",
            "Accept": "text/event-stream" if stream else "application/json"
        }
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'Generate underlying data table of the figure below: <img src="data:image/png;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.20,
            "top_p": 0.20,
            "stream": stream
        }

        response = requests.post(invoke_url, headers=headers, json=payload)
        compiled_content = ""

        if stream:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    print("Received line:", decoded_line)  # Helps to debug what you're actually receiving
                    if decoded_line.startswith('data:'):
                        try:
                            json_data = json.loads(decoded_line[len("data: "):])
                            content = json_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            compiled_content += content.replace("\\n", "\n")
                        except json.JSONDecodeError:
                            print("Failed to decode JSON from:", decoded_line)  # Print error line for debugging
        else:
            response_data = response.json()
            print(response_data)
            # Handle non-streamed response here if necessary

        return compiled_content.strip()

    def pdf_model_report(self, accuracy_report: str, plots_paths: str, inference_code: str, plot_description: str, data_report: str, specifications: str, output_path: str):
        """
        Generates a PDF report summarizing the model training and usage.

        Parameters:
            accuracy_report (str): Text or path to the accuracy and other performance metrics report.
            plots_paths (str): Paths to the generated plots.
            plot_description (str): Descriptions of the plots.
            inference_code (str): Code snippet for model inference.
            specifications (str): Specifications on how the model should be.
            output_path (str): Path to save the generated PDF report.
        """
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 50, "Model Report")

        # Add accuracy report
        c.setFont("Helvetica", 12)
        c.drawString(100, height - 100, "Accuracy Report:")
        c.drawString(100, height - 120, accuracy_report)

        # Add plot description
        c.drawString(100, height - 160, "Plot Description:")
        c.drawString(100, height - 180, plot_description)

        # Add inference code
        c.drawString(100, height - 220, "Inference Code:")
        c.drawString(100, height - 240, inference_code)

        # Add specifications
        c.drawString(100, height - 280, "Model Specifications:")
        c.drawString(100, height - 300, specifications)

        # Add plots
        c.drawString(100, height - 340, "Plots:")
        y_position = height - 360
        for plot_path in plots_paths.split(','):
            c.drawImage(plot_path.strip(), 100, y_position, width=200, height=100)
            y_position -= 120  # Adjust for next image

        c.save()

# Example usage
if __name__ == "__main__":
    # Sample inputs for testing
    pred_model_path = "path/to/trained/model"
    data_path = "path/to/dataset"
    model_code = "model training code here"
    data_report = "data report here"
    accuracy_report = "accuracy report here"
    plots_paths = "path/to/plots1.png, path/to/plots2.png"  # Comma-separated paths
    plot_description = "description of plots"
    specifications = "model specifications here"
    output_pdf_path = "model_report.pdf"

    # Initialize the Assessor
    assessor = Assessor()

    # Test the pdf_model_report method
    assessor.pdf_model_report(accuracy_report, plots_paths, model_code, plot_description, data_report, specifications, output_pdf_path)
    print(f"PDF report generated at: {output_pdf_path}")

    # Test other methods as needed...
    # For example:
    plot_description_sample = assessor.graph_reader()
    print("Compiled Output:\n", plot_description_sample)