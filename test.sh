#!/bin/bash

# Create project root directory
mkdir -p project-root

# Create directories for app, data, scripts, templates, and tests
mkdir -p project-root/assets
mkdir -p project-root/data
mkdir -p project-root/scripts
mkdir -p project-root/templates
mkdir -p project-root/tests

# Create main app.py file
cat <<EOL > project-root/app.py
import shiny
from shiny import ui, render, input, output
import plotly.express as px
import pandas as pd
from scripts.data_processing import load_data
from scripts.plot_helpers import create_histogram, create_scatterplot

# Load data
data = load_data("data/dataset.csv")

# Define UI
app_ui = ui.page_fluid(
    ui.input_dropdown("variable", "Select Variable", choices=data.columns),
    ui.output_plot("histogram_plot"),
    ui.input_dropdown("x_var", "Select X-axis", choices=data.columns),
    ui.input_dropdown("y_var", "Select Y-axis", choices=data.columns),
    ui.output_plot("scatter_plot")
)

# Define server logic
def server(input, output, session):
    
    # Update histogram plot based on dropdown selection
    @output()
    @render.plot
    def histogram_plot():
        return create_histogram(data, input.variable)

    # Update scatter plot based on x and y axis selections
    @output()
    @render.plot
    def scatter_plot():
        return create_scatterplot(data, input.x_var, input.y_var)

# Run the app
shiny.App(app_ui, server)
EOL

# Create requirements.txt
cat <<EOL > project-root/requirements.txt
shiny
pandas
plotly
EOL

# Create empty style.css (for custom styles)
touch project-root/assets/style.css

# Create sample dataset file (for placeholder)
touch project-root/data/dataset.csv

# Create data_processing.py file for data-related functions
cat <<EOL > project-root/scripts/data_processing.py
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(data):
    # Example function to clean the data
    data = data.dropna()  # Remove rows with missing values
    return data
EOL

# Create plot_helpers.py file for plot-related functions
cat <<EOL > project-root/scripts/plot_helpers.py
import plotly.express as px

def create_histogram(data, variable):
    fig = px.histogram(data, x=variable, nbins=50, barmode='overlay')
    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
    return fig

def create_scatterplot(data, x_var, y_var):
    fig = px.scatter(data, x=x_var, y=y_var)
    return fig
EOL

# Create layout.html (template placeholder)
touch project-root/templates/layout.html

# Create test files for data processing and plotting functions
cat <<EOL > project-root/tests/test_data_processing.py
# Test cases for data processing functions

def test_load_data():
    assert True  # Add your test here

def test_clean_data():
    assert True  # Add your test here
EOL

cat <<EOL > project-root/tests/test_plots.py
# Test cases for plot functions

def test_create_histogram():
    assert True  # Add your test here

def test_create_scatterplot():
    assert True  # Add your test here
EOL

# Print success message
echo "Project structure created successfully!"
