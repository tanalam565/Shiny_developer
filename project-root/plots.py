import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shiny import App, ui, render

# Load the dataset
df = pd.read_csv("data/tiny_earth_2024.csv")

# Drop the 'ID' column 
if 'ID' in df.columns:
    df = df.drop(columns=["ID"])

# List of numerical columns for dropdowns
numerical_columns = [
    "Zip Code", "Temperature", "CFUs/g", "Antibiotic Producers( ESKAPE)",
    "Isolates (lactose Fermenter)", "Isolates (Gram Negative)", 
    "Isolates (Gram Positive)", "Isolates (Reduce Sulfur)", 
    "Motile Organisms", "Indole Positive Organisms", 
    "Isolates (Critrate Positive)", "Isolates (Gelatinase Positive)", 
    "Isolates( Hydrolyze Esculin in Bile)", "Isolates (MR Positive)", 
    "Isolates (VP positive)"
]

# Define the UI
app_ui = ui.page_fluid(
    ui.h2("Tiny Earth Explorer"),
    ui.layout_sidebar(
        ui.sidebar(
            # Histogram controls
            ui.h4("Histogram"),
            ui.input_select(
                id="hist_var",
                label="Select a variable for histogram:",
                choices=numerical_columns,
            ),

            # Scatterplot controls
            ui.h4("Scatterplot"),
            ui.input_select(
                id="scatter_x",
                label="Select X-axis for scatterplot:",
                choices=numerical_columns,
            ),
            ui.input_select(
                id="scatter_y",
                label="Select Y-axis for scatterplot:",
                choices=numerical_columns,
            ),
        ),
        ui.output_plot("histogram"),
        ui.output_plot("scatterplot"),
    ),
)

# Define the server logic
def server(input, output, session):
    @output
    @render.plot
    def histogram():
        var = input.hist_var()  # Get selected variable for histogram
        plt.figure(figsize=(12, 6))
        sns.histplot(df[var], bins=20, kde=False, color="skyblue", edgecolor="black")
        plt.title(f"Histogram of {var}")
        plt.xlabel(var)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        return plt.gcf()

    @output
    @render.plot
    def scatterplot():
        x_var = input.scatter_x()  # Get selected x-axis variable
        y_var = input.scatter_y()  # Get selected y-axis variable
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x=x_var, y=y_var, color="blue", alpha=0.7)
        sns.regplot(data=df, x=x_var, y=y_var, scatter=False, color="red", line_kws={"linestyle": "--"})
        plt.title(f"Scatterplot: {x_var} vs {y_var}")
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        return plt.gcf()

# Combine UI and server into a Shiny app
app = App(app_ui, server)