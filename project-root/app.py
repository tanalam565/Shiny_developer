import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shiny import App, ui, render
import plotly.express as px

# Load the dataset
df = pd.read_csv("data/tiny_earth_2024.csv")
df.columns = df.columns.str.strip()

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

# Define columns of interest for the pie chart and stacked bar plot
columns_of_interest = [
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
        ui.output_ui("interactive_scatterplot"),  # Placeholder for Plotly scatter plot
        ui.output_plot("pie_chart"),  # Placeholder for the pie chart
        ui.output_plot("stacked_bar_plot"),  # Placeholder for the stacked bar plot
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

    @output
    @render.ui
    def interactive_scatterplot():
        x_var = input.scatter_x()  # Get selected x-axis variable
        y_var = input.scatter_y()  # Get selected y-axis variable
        # Create the Plotly scatter plot
        fig = px.scatter(
            df,
            x=x_var,
            y=y_var,
            color="Zip Code",  # Color by Zip Code (or any other column)
            hover_data=numerical_columns,  # Show all numerical columns on hover
            title=f"Interactive Scatter Plot: {x_var} vs {y_var}"
        )
        return fig.to_html(full_html=False)  # Render Plotly figure as HTML

    @output
    @render.plot
    def pie_chart():
        # Sum the total counts of isolates across the categories
        total_isolates = df[columns_of_interest].sum()

        # Plotting the pie chart
        plt.figure(figsize=(8, 8))
        total_isolates.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Set3')
        plt.title('Proportions of Different Isolates')
        plt.ylabel('')  # Hides the ylabel to make it cleaner
        plt.tight_layout()
        return plt.gcf()

    # @output
    # @render.plot
    # def stacked_bar_plot():
        # Group by Zip Code and sum each isolate category
        zip_code_isolates = df.groupby('Zip Code')[columns_of_interest].sum()

        # Sort by total isolates and select the top 5 zip codes
        zip_code_isolates['Total Isolates'] = zip_code_isolates.sum(axis=1)
        top_5_zip_codes = zip_code_isolates.nlargest(5, 'Total Isolates').drop(columns=['Total Isolates'])

        # Plot stacked bar plot
        plt.figure(figsize=(12, 6))
        top_5_zip_codes.plot(kind='bar', stacked=True, cmap='tab20')
        plt.title('Top 5 Zip Codes by Total Isolates')
        plt.xlabel('Zip Code')
        plt.ylabel('Total Isolates')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()

# Combine UI and server into a Shiny app
app = App(app_ui, server)