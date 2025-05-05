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

# Define environmental factor columns and isolate characteristic columns
environmental_factors = ["Zip Code", "Temperature", "Weather", "Description of vegetation", "Plants observed"]

# Columns to exclude from hover data
exclude_columns = [
    "Name", 
    "Month of Soil Collection", 
    "Observations", 
    "Sampling Site with Vegetation",
    "Media Features",
    "Other Biochemical Tests",
    "Other Observations",
    "How fun is Tiny Earth?"
]

# Check if the required environmental columns exist in the dataframe
available_env_factors = [col for col in environmental_factors if col in df.columns]

# All other columns that are not environmental factors or excluded will be isolate characteristics
isolate_characteristics = [col for col in df.columns if col not in available_env_factors and col not in exclude_columns]

# Create custom hover data template
hover_template = "<b>Environmental Factors:</b><br>"
for col in available_env_factors:
    hover_template += f"{col}: %{{customdata[{available_env_factors.index(col)}]}}<br>"

hover_template += "<br><b>Isolate Characteristics:</b><br>"
for col in isolate_characteristics:
    hover_template += f"{col}: %{{customdata[{len(available_env_factors) + isolate_characteristics.index(col)}]}}<br>"

# Separate numerical and categorical columns for better selection options
true_numerical_columns = [
    "Temperature", "CFUs/g", "Antibiotic Producers",
    "Lactose Fermenter", "Gram Negative", 
    "Gram Positive", "Reduce Sulfur", 
    "Motile Organisms", "Indole Positive Organisms", 
    "Critrate Positive", "Gelatinase Positive", 
    "Hydrolyze Esculin in Bile", "MR Positive", 
    "VP positive"
]

categorical_columns = ["Zip Code"]

all_columns = true_numerical_columns + categorical_columns

# Define columns of interest for the pie chart and stacked bar plot
columns_of_interest = [
    "Lactose Fermenter", "Gram Negative", 
    "Gram Positive", "Reduce Sulfur", 
    "Motile Organisms", "Indole Positive Organisms", 
    "Critrate Positive", "Gelatinase Positive", 
    "Hydrolyze Esculin in Bile", "MR Positive", 
    "VP positive"
]

# Get unique ZIP codes for dropdown
unique_zip_codes = sorted(df["Zip Code"].unique().astype(str))

# Define the UI
app_ui = ui.page_fluid(
    ui.h2("Tiny Earth Explorer"),
    ui.layout_sidebar(
        ui.sidebar(
            # ZIP Code Profile controls
            ui.h4("ZIP Code Profile"),
            ui.input_select(
                id="selected_zip",
                label="Select ZIP Code:",
                choices=unique_zip_codes,
                selected=unique_zip_codes[0] if unique_zip_codes else None
            ),
            
            # Histogram controls
            ui.h4("Histogram"),
            ui.input_select(
                id="hist_var",
                label="Select a variable for histogram:",
                choices=all_columns,
                selected=true_numerical_columns[0]  # Default to a numerical column
            ),

            # Static Scatterplot controls
            ui.h4("Static Scatterplot"),
            ui.input_select(
                id="static_x",
                label="Select X-axis for static scatterplot:",
                choices=all_columns,
                selected=true_numerical_columns[0]  # Default to a numerical column
            ),
            ui.input_select(
                id="static_y",
                label="Select Y-axis for static scatterplot:",
                choices=all_columns,
                selected=true_numerical_columns[1]  # Default to a different numerical column
            ),
            
            # Interactive Scatterplot controls
            ui.h4("Interactive Scatterplot"),
            ui.input_select(
                id="interactive_x",
                label="Select X-axis for interactive scatterplot:",
                choices=all_columns,
                selected=true_numerical_columns[2]  # Default to a different numerical column
            ),
            ui.input_select(
                id="interactive_y",
                label="Select Y-axis for interactive scatterplot:",
                choices=all_columns,
                selected=true_numerical_columns[3]  # Default to a different numerical column
            ),
        ),
        # Main content area - ZIP Code profile first
        ui.h3("ZIP Code Profile"),
        ui.output_plot("zip_profile", height="400px"),
        
        # Other plots follow
        ui.output_plot("histogram", height="400px"),
        
        # Static scatterplot
        ui.h3("Static Scatterplot"),
        ui.output_plot("static_scatterplot", height="400px"),
        
        # Interactive scatterplot
        ui.h3("Interactive Scatterplot"),
        ui.output_ui("interactive_scatterplot"),
        
        ui.h3("Pie Chart for Isolates"),
        ui.output_plot("pie_chart", height="400px"),

        ui.h3("Bar Plot for Isolates"),
        ui.output_plot("new_bar_plot", height="400px")
    ),
)

# Define the server logic
def server(input, output, session):
    @output
    @render.plot
    def zip_profile():
        try:
            selected_zip = input.selected_zip()
            
            # Filter data for the selected ZIP code
            zip_data = df[df["Zip Code"].astype(str) == selected_zip]
            
            if len(zip_data) == 0:
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f"No data available for ZIP code {selected_zip}", 
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.axis('off')
                return plt.gcf()
            
            # Get means of numerical columns for the selected ZIP
            zip_means = zip_data[true_numerical_columns].mean()
            
            # Get Temperature and CFUs/g values separately
            temp_value = zip_means["Temperature"]
            cfu_value = zip_means["CFUs/g"]
            
            # Filter out Temperature and CFUs/g for the bar chart
            filtered_columns = [col for col in true_numerical_columns if col not in ["Temperature", "CFUs/g"]]
            filtered_means = zip_means[filtered_columns]
            
            # Select top 8 variables for better visualization
            top_vars = filtered_means.nlargest(8).index.tolist()
            
            # Create bar chart with larger figure size
            fig, ax = plt.subplots(figsize=(20, 8))
            bars = ax.bar(top_vars, filtered_means[top_vars], color=plt.cm.viridis(np.linspace(0, 0.8, len(top_vars))))
            
            # Improve readability with larger fonts
            plt.xticks(rotation=20, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.ylabel('Average Frequency of Isolates', fontsize=12)
            plt.title(f'Average Frequency of Isolates for ZIP Code {selected_zip}', 
                     size=14, pad=20)
            
            # Add value labels on top of bars
            y_max = ax.get_ylim()[1]
            for i, rect in enumerate(bars):
                height = rect.get_height()
                if height > 0.85 * y_max:
                    ax.text(rect.get_x() + rect.get_width()/2., height * 0.9,
                            f'{filtered_means[top_vars[i]]:.2f}',
                            ha='center', va='top', color='white', fontsize=10)
                else:
                    ax.text(rect.get_x() + rect.get_width()/2., height + 0.1,
                            f'{filtered_means[top_vars[i]]:.2f}',
                            ha='center', va='bottom', fontsize=10)
            
            # Add Temperature and CFUs/g as text annotations
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            text_str = f"Average Temperature: {temp_value:.2f}\nAverage CFUs/g: {cfu_value:.2f}"
            ax.text(0.95, 0.95, text_str,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top', 
                horizontalalignment='right',
                bbox=props)
            
            plt.tight_layout(pad=5.0)
            return fig
            
        except Exception as e:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error in ZIP profile plot: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            return plt.gcf()

    @output
    @render.plot
    def histogram():
        var = input.hist_var()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            if var in categorical_columns:
                sns.countplot(y=df[var], color="skyblue", ax=ax)
                plt.title(f"Count of {var}", pad=15)
                plt.xlabel("Count", labelpad=10)
                plt.ylabel(var, labelpad=10)
            else:
                sns.histplot(df[var], bins=20, kde=False, color="skyblue", edgecolor="black", ax=ax)
                plt.title(f"Histogram of {var}", pad=15)
                plt.xlabel(var, labelpad=10)
                plt.ylabel("Frequency", labelpad=10)
            
            plt.tight_layout(pad=2.0)
            return fig
        except Exception as e:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error in histogram: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            return plt.gcf()

    @output
    @render.plot
    def static_scatterplot():
        try:
            x_var = input.static_x()
            y_var = input.static_y()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.scatterplot(data=df, x=x_var, y=y_var, color="blue", alpha=0.7, ax=ax)
            
            if x_var in true_numerical_columns and y_var in true_numerical_columns:
                try:
                    sns.regplot(data=df, x=x_var, y=y_var, scatter=False, 
                            color="red", line_kws={"linestyle": "--"}, ax=ax)
                except Exception as e:
                    print(f"Regression failed: {str(e)}")
                    
            plt.title(f"Static Scatterplot: {x_var} vs {y_var}", pad=15)
            plt.xlabel(x_var, labelpad=10)
            plt.ylabel(y_var, labelpad=10)
            
            plt.tight_layout(pad=2.0)
            return fig
        except Exception as e:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error in static scatterplot: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            return plt.gcf()

    @output
    @render.ui
    def interactive_scatterplot():
        try:
            x_var = input.interactive_x()
            y_var = input.interactive_y()
            
            # Prepare custom data for hover
            custom_data = df[available_env_factors + isolate_characteristics].values
            
            # Create base scatter plot
            fig = px.scatter(
                df,
                x=x_var,
                y=y_var,
                color="Zip Code",
                title=f"Interactive Scatter Plot: {x_var} vs {y_var}"
            )
            
            # Update hover template with customdata
            fig.update_traces(
                customdata=custom_data,
                hovertemplate=hover_template
            )
            
            return ui.div(
                {"style": "height:500px;"},
                ui.HTML(
                    fig.to_html(full_html=False, include_plotlyjs='cdn')
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
            total_isolates = df[columns_of_interest].sum()
            top_8_isolates = total_isolates.nlargest(8)
            other_isolates = pd.Series({'Other Isolates': total_isolates.sum() - top_8_isolates.sum()})
            pie_data = pd.concat([top_8_isolates, other_isolates])

            fig, ax = plt.subplots(figsize=(14, 8))
            wedges, texts, autotexts = ax.pie(pie_data, autopct='%1.1f%%', startangle=90, 
                                             colors=plt.cm.Set3.colors[:len(pie_data)],
                                             textprops={'fontsize': 12})
            ax.set_title('Proportions of Different Isolates', pad=20)
            
            legend_labels = [f"{label} ({pie_data.iloc[i]:.0f} samples)" for i, label in enumerate(pie_data.index)]
            ax.legend(wedges, legend_labels, title="Categories", loc="center left", 
                     bbox_to_anchor=(1, 0.5), fontsize=12)
            
            plt.tight_layout(pad=4.0, rect=[0, 0, 0.85, 1])
            return fig
        except Exception as e:
            plt.figure(figsize=(20, 8))
            plt.text(0.5, 0.5, f"Error in pie chart: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            return plt.gcf()

    @output
    @render.plot
    def new_bar_plot():
        fig, ax = plt.subplots(figsize=(20, 10))
        column_counts = df[columns_of_interest].apply(lambda x: (x > 0).sum())
        ax.bar(column_counts.index, column_counts.values, color=plt.cm.Set3(np.linspace(0, 1, len(column_counts))))
        ax.set_title('Isolates Count by Category', pad=15)
        ax.set_xlabel('Isolate Categories', labelpad=20)
        ax.set_ylabel('Number of Samples', labelpad=10)
        plt.xticks(rotation=20, ha='right', fontsize=10)
        plt.tight_layout(pad=5.0)
        return fig

 
# Combine UI and server into a Shiny app
app = App(app_ui, server)