import ipywidgets as widgets
from IPython.display import display, HTML
from framework import A4MLFramework

# Instantiate the framework class
framework = A4MLFramework(model='gpt-4o-mini')

# Create widgets for user input
parent_path_widget = widgets.Text(value='/path/to/data', description="Parent Path:", placeholder='Type here')
file_name_widget = widgets.Text(value='data.csv', description="File Name:", placeholder='Type here')
user_demand_widget = widgets.Text(value='Predict success from the rest of features.', description="User Demand:", placeholder='Type here')
model_widget = widgets.Dropdown(options=['gpt-4', 'gpt-4o', 'bert', 'transformer'], value='gpt-4o', description='Model:')

# Create a button to trigger the processing
button = widgets.Button(description="Run Model")
output = widgets.Output()

# Display all widgets
display(parent_path_widget, file_name_widget, user_demand_widget, model_widget, button, output)

# Define the action for the button click event
def on_button_clicked(b):
    with output:
        output.clear_output()  # Clear previous output
        # Call the framework's a4ml function with the user inputs
        result = framework.a4ml(parent_path_widget.value, file_name_widget.value, user_demand_widget.value, model_widget.value)
        # Display the result as HTML
        display(HTML(result[0]))

# Bind the click event to the function
button.on_click(on_button_clicked)