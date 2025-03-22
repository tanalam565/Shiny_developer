import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shiny import App, ui, render
import plotly.express as px
import numpy as np

# Load the dataset
df = pd.read_csv("data/tiny_earth_2024.csv")
df.columns = df.columns.str.strip()

# Drop the 'ID' column if it exists
if 'ID' in df.columns:
    df = df.drop(columns=["ID"])

# Separate numerical and categorical columns for better selection options
true_numerical_columns = [
    "Temperature", "CFUs/g", "Antibiotic Producers( ESKAPE)",
    "Isolates (lactose Fermenter)", "Isolates (Gram Negative)", 
    "Isolates (Gram Positive)", "Isolates (Reduce Sulfur)", 
    "Motile Organisms", "Indole Positive Organisms", 
    "Isolates (Critrate Positive)", "Isolates (Gelatinase Positive)", 
    "Isolates( Hydrolyze Esculin in Bile)", "Isolates (MR Positive)", 
    "Isolates (VP positive)"
]

categorical_columns = ["Zip Code"]

all_columns = true_numerical_columns + categorical_columns

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
                choices=all_columns,
                selected=true_numerical_columns[0]  # Default to a numerical column
            ),

            # Scatterplot controls
            ui.h4("Scatterplot"),
            ui.input_select(
                id="scatter_x",
                label="Select X-axis for scatterplot:",
                choices=all_columns,
                selected=true_numerical_columns[0]  # Default to a numerical column
            ),
            ui.input_select(
                id="scatter_y",
                label="Select Y-axis for scatterplot:",
                choices=all_columns,
                selected=true_numerical_columns[1]  # Default to a different numerical column
            ),
        ),
        # Main content area
        ui.output_plot("histogram", height="400px"),
        ui.output_plot("scatterplot", height="400px"),
        ui.output_ui("interactive_scatterplot"),
        ui.output_plot("pie_chart", height="400px"),
        ui.output_plot("stacked_bar_plot", height="400px")
    ),
)

# Define the server logic
def server(input, output, session):
    @output
    @render.plot
    def histogram():
        var = input.hist_var()  # Get selected variable for histogram
        plt.figure(figsize=(8, 6))
        
        try:
            # Check if the column contains numeric data
            if var in categorical_columns:
                # For categorical data like Zip Code, use countplot instead
                sns.countplot(y=df[var], color="skyblue")
                plt.title(f"Count of {var}")
                plt.xlabel("Count")
                plt.ylabel(var)
            else:
                # For numerical data, use histplot
                sns.histplot(df[var], bins=20, kde=False, color="skyblue", edgecolor="black")
                plt.title(f"Histogram of {var}")
                plt.xlabel(var)
                plt.ylabel("Frequency")
            
            plt.tight_layout()
            return plt.gcf()
        except Exception as e:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error in histogram: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            return plt.gcf()

    @output
    @render.plot
    def scatterplot():
        try:
            x_var = input.scatter_x()  # Get selected x-axis variable
            y_var = input.scatter_y()  # Get selected y-axis variable
            
            plt.figure(figsize=(8, 6))
            
            # Create basic scatter plot
            sns.scatterplot(data=df, x=x_var, y=y_var, color="blue", alpha=0.7)
            
            # Only add regression line if both variables are numerical
            if x_var in true_numerical_columns and y_var in true_numerical_columns:
                try:
                    sns.regplot(data=df, x=x_var, y=y_var, scatter=False, 
                              color="red", line_kws={"linestyle": "--"})
                except Exception as e:
                    # If regression fails, just log it and continue without the line
                    print(f"Regression failed: {str(e)}")
                    
            plt.title(f"Scatterplot: {x_var} vs {y_var}")
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.tight_layout()
            return plt.gcf()
        except Exception as e:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error in scatterplot: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            return plt.gcf()

    @output
    @render.ui
    def interactive_scatterplot():
        try:
            x_var = input.scatter_x()  # Get selected x-axis variable
            y_var = input.scatter_y()  # Get selected y-axis variable
            
            # Create an HTML container with a specific height
            return ui.div(
                {"style": "height:500px;"},
                # Create the Plotly scatter plot
                ui.HTML(
                    px.scatter(
                        df,
                        x=x_var,
                        y=y_var,
                        color="Zip Code",  # Color by Zip Code
                        hover_data=["Temperature", "CFUs/g"],  # Limit hover data to reduce payload
                        title=f"Interactive Scatter Plot: {x_var} vs {y_var}"
                    ).to_html(full_html=False, include_plotlyjs='cdn')  # Use CDN for plotly.js
                )
            )
        except Exception as e:
            return ui.div(
                {"style": "height:500px; display:flex; align-items:center; justify-content:center;"},
                f"Error in interactive plot: {str(e)}"
            )

    @output
    @render.plot
    def pie_chart():
        try:
            # Sum the total counts of isolates across the categories
            total_isolates = df[columns_of_interest].sum()
            
            # Limit to top 5 categories for readability
            top_5_isolates = total_isolates.nlargest(5)
            other_isolates = pd.Series({'Other Isolates': total_isolates.sum() - top_5_isolates.sum()})
            pie_data = pd.concat([top_5_isolates, other_isolates])

            # Plotting the pie chart
            plt.figure(figsize=(8, 6))
            pie_data.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Set3')
            plt.title('Proportions of Different Isolates')
            plt.ylabel('')  # Hides the ylabel to make it cleaner
            plt.tight_layout()
            return plt.gcf()
        except Exception as e:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error in pie chart: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            return plt.gcf()

    @output
    @render.plot
    def stacked_bar_plot():
        try:
            # Convert Zip Code to string if it's not already
            df['Zip Code'] = df['Zip Code'].astype(str)
            
            # Group by Zip Code and sum each isolate category
            zip_code_isolates = df.groupby('Zip Code')[columns_of_interest].sum()

            # Sort by total isolates and select the top 5 zip codes
            zip_code_isolates['Total Isolates'] = zip_code_isolates.sum(axis=1)
            top_5_zip_codes = zip_code_isolates.nlargest(5, 'Total Isolates').drop(columns=['Total Isolates'])

            # Plot stacked bar plot
            plt.figure(figsize=(8, 6))
            top_5_zip_codes.plot(kind='bar', stacked=True, cmap='tab20')
            plt.title('Top 5 Zip Codes by Total Isolates')
            plt.xlabel('Zip Code')
            plt.ylabel('Total Isolates')
            plt.xticks(rotation=45)
            plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), fontsize='small')
            plt.tight_layout()
            return plt.gcf()
        except Exception as e:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error in stacked bar plot: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            return plt.gcf()

# Combine UI and server into a Shiny app
app = App(app_ui, server)