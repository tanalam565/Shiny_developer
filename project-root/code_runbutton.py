import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shiny import App, ui, render, reactive

# Load the dataset
df = pd.read_csv("data/tiny_earth_2024.csv")

# Drop the 'ID' column if it exists
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
    ui.h2("Soil Data Explorer"),
    ui.layout_sidebar(
        ui.sidebar(
            # Histogram controls
            ui.h4("Histogram Settings"),
            ui.input_select(
                id="hist_var",
                label="Select a variable for histogram:",
                choices=numerical_columns,
            ),
            ui.input_action_button("run_hist", "Run Histogram"),  # Run button for histogram

            # Scatterplot controls
            ui.h4("Scatterplot Settings"),
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
            ui.input_action_button("run_scatter", "Run Scatterplot"),  # Run button for scatterplot
        ),
        ui.output_plot("histogram"),
        ui.output_plot("scatterplot"),
    ),
)

# Define the server logic
def server(input, output, session):
    # Reactive value for histogram data
    hist_var = reactive.Value(numerical_columns[0])  # Initialize with the first variable

    # Reactive value for scatterplot data
    scatter_x = reactive.Value(numerical_columns[0])  # Initialize with the first variable
    scatter_y = reactive.Value(numerical_columns[1])  # Initialize with the second variable

    # Update histogram data only when Run Histogram button is clicked
    @reactive.Effect
    def _():
        input.run_hist()  # Trigger when Run Histogram button is clicked
        hist_var.set(input.hist_var())  # Update the reactive value

    # Update scatterplot data only when Run Scatterplot button is clicked
    @reactive.Effect
    def _():
        input.run_scatter()  # Trigger when Run Scatterplot button is clicked
        scatter_x.set(input.scatter_x())  # Update the reactive value
        scatter_y.set(input.scatter_y())  # Update the reactive value

    @output
    @render.plot
    def histogram():
        var = hist_var()  # Use the reactive value
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
        x_var = scatter_x()  # Use the reactive value
        y_var = scatter_y()  # Use the reactive value
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