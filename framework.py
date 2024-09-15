import os
from Agents.code_writer import CodeWriter  # Corrected import path
from Agents.dataAgents import DataAgents  # Corrected import path
from Agents.assessor import Assessor  # Corrected import path

class A4MLFramework:
    def __init__(self, model='gpt-4o-mini'):
        self.code_writer = CodeWriter(model=model)
        self.data_agent = DataAgents(model=model)
        self.assessor = Assessor(model=model)
        self.model = model

    def a4ml(self, parent_path, file_name, user_demand):
        data_path = os.path.join(parent_path, file_name)
        
        # 1. Process the data and understand it
        model_specifications, new_path, data_report = self.data_agent.data_block(parent_path, file_name, user_demand, model=self.model)

        # 2. Design the Model
        model_design = self.code_writer.write_code("Design a model", model_specifications)
        
        # 3. Load and Assess
        model_name = 'pred_model.pkl'
        pred_model_path = os.path.join(parent_path, model_name)
        mode_code = f"Model Code: {model_design}"

        assessment = self.assessor.assessor(pred_model_path, new_path, mode_code)

        # 4. Generate Plots
        plot_code = self.assessor.plotter(mode_code, pred_model_path, data_path, data_report)
        exec(plot_code)

        # 5. Inference Code
        inference_code = self.assessor.inference_plotter(mode_code, pred_model_path, data_path, data_report)

        # 6. Process plots in directory and generate final report
        directory_path = parent_path
        overall_plot_desc = ""
        overall_plot_paths = ""

        for filename in os.listdir(directory_path):
            if filename.endswith(".png"):
                full_path = os.path.join(directory_path, filename)
                plot_description = self.assessor.graph_reader(full_path)
                overall_plot_desc += "\n" + plot_description + "\n"
                overall_plot_paths += "\n" + full_path + "\n"

        # 7. Generate the model report
        model_report = self.assessor.model_reporter(assessment, overall_plot_paths, inference_code, overall_plot_desc, data_report, model_design)

        return model_report