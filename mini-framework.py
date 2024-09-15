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

        # ... existing code removed ...
        
        return model_design  # Return the model design instead of the full report
    
    def main():
        # Sample usage
        parent_path = 'agent_for_prediction/datasets'
        file_name = 'social-media.csv'
        user_demand = 'Analyze social media trends and generate insights.'

        framework = A4MLFramework()
        model_design = framework.a4ml(parent_path, file_name, user_demand)

        print("Model Design:")
        print(model_design)

    if __name__ == "__main__":
        main()