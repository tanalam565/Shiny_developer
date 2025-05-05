import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shiny import App, ui, render
import plotly.express as px
import numpy as np
from shiny.ui import tags

# Set custom color palette for consistency
MAIN_COLOR = "#3498db"  # Primary blue
ACCENT_COLOR = "#e74c3c"  # Red accent
COMPLEMENTARY_COLOR = "#2ecc71"  # Green complementary
BACKGROUND_COLOR = "#f5f8fa"  # Light background
TEXT_COLOR = "#2c3e50"  # Dark blue text

# Custom CSS for better styling
custom_css = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f8fa;
    color: #2c3e50;
}
h2 {
    color: #3498db;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
    margin-bottom: 20px;
}
h3 {
    color: #2c3e50;
    border-left: 4px solid #3498db;
    padding-left: 10px;
    margin-top: 30px;
    background-color: #ecf0f1;
    padding: 8px 15px;
    border-radius: 4px;
}
h4 {
    color: #3498db;
    margin-top: 25px;
}
.sidebar {
    background-color: #ecf0f1;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.control-label {
    font-weight: 600;
    color: #34495e;
}
.form-control {
    border-radius: 4px;
    border: 1px solid #bdc3c7;
}
.form-control:focus {
    border-color: #3498db;
    box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
}
.plot-container {
    background-color: white;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 25px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.app-header {
    background-color: #3498db;
    color: white;
    padding: 15px 20px;
    margin-bottom: 20px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.app-header h2 {
    margin: 0;
    border: none;
    color: white;
}
.app-header img {
    height: 40px;
    margin-right: 15px;
}
.footer {
    text-align: center;
    margin-top: 30px;
    padding: 15px;
    background-color: #ecf0f1;
    border-radius: 8px;
    font-size: 0.9em;
    color: #7f8c8d;
}
"""

# Load the dataset
df = pd.read_csv("data/tiny_earth_2024.csv")
df.columns = df.columns.str.strip()

# Drop the 'ID' column if it exists
if 'ID' in df.columns:
    df = df.drop(columns=["ID"])

# Define environmental factor columns and isolate characteristic columns
environmental_factors = ["Zip Code", "Temperature", "Description of vegetation", "Plants observed"]

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

# Create custom hover data template with styling
hover_template = "<span style='font-weight:bold; color:#3498db; font-size:14px'>Environmental Factors:</span><br>"
for col in available_env_factors:
    hover_template += f"<span style='font-weight:bold'>{col}:</span> %{{customdata[{available_env_factors.index(col)}]}}<br>"

hover_template += "<br><span style='font-weight:bold; color:#e74c3c; font-size:14px'>Isolate Characteristics:</span><br>"
for col in isolate_characteristics:
    hover_template += f"<span style='font-weight:bold'>{col}:</span> %{{customdata[{len(available_env_factors) + isolate_characteristics.index(col)}]}}<br>"

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

# Set custom matplotlib style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')

# Define the UI with enhanced styling
app_ui = ui.page_fluid(
    # Custom CSS
    tags.style(custom_css),
    
    # App Header with Logo
    tags.div(
        {"class": "app-header"},
        tags.img(src="https://placekitten.com/100/100", alt="Tiny Earth Logo"),
        ui.h2("Tiny Earth Explorer Dashboard"),
    ),
    
    # Brief introduction/description
    tags.div(
        {"style": "margin-bottom: 20px; padding: 15px; background-color: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);"},
        tags.p("Welcome to the Tiny Earth Explorer! This interactive dashboard allows you to explore soil microbiome data collected by Tiny Earth participants. Analyze environmental factors and isolate characteristics across different locations."),
    ),
    
    ui.layout_sidebar(
        ui.sidebar(
            {"class": "sidebar"},
            # ZIP Code Profile controls
            ui.h4("ZIP Code Profile"),
            ui.input_select(
                id="selected_zip",
                label="Select ZIP Code:",
                choices=unique_zip_codes,
                selected=unique_zip_codes[0] if unique_zip_codes else None
            ),
            
            # Histogram controls
            ui.h4("Histogram Analysis"),
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
                label="Select X-axis:",
                choices=all_columns,
                selected=true_numerical_columns[0]  # Default to a numerical column
            ),
            ui.input_select(
                id="static_y",
                label="Select Y-axis:",
                choices=all_columns,
                selected=true_numerical_columns[1]  # Default to a different numerical column
            ),
            
            # Interactive Scatterplot controls
            ui.h4("Interactive Scatterplot"),
            ui.input_select(
                id="interactive_x",
                label="Select X-axis:",
                choices=all_columns,
                selected=true_numerical_columns[2]  # Default to a different numerical column
            ),
            ui.input_select(
                id="interactive_y",
                label="Select Y-axis:",
                choices=all_columns,
                selected=true_numerical_columns[3]  # Default to a different numerical column
            ),
        ),
        
        # Main content area
        # ZIP Code profile
        ui.h3("ZIP Code Microbiome Profile"),
        tags.div(
            {"class": "plot-container"},
            ui.output_plot("zip_profile", height="400px"),
        ),
        
        # Histogram
        ui.h3("Distribution Analysis"),
        tags.div(
            {"class": "plot-container"},
            ui.output_plot("histogram", height="400px"),
        ),
        
        # Static scatterplot
        ui.h3("Correlation Analysis (Static)"),
        tags.div(
            {"class": "plot-container"},
            ui.output_plot("static_scatterplot", height="400px"),
        ),
        
        # Interactive scatterplot
        ui.h3("Interactive Data Explorer"),
        tags.div(
            {"class": "plot-container"},
            ui.output_ui("interactive_scatterplot"),
        ),
        
        # Pie Chart
        ui.h3("Isolate Distribution"),
        tags.div(
            {"class": "plot-container"},
            ui.output_plot("pie_chart", height="400px"),
        ),

        # Bar Plot
        ui.h3("Isolate Frequency Analysis"),
        tags.div(
            {"class": "plot-container"},
            ui.output_plot("new_bar_plot", height="400px"),
        ),
        
        # Footer
        tags.div(
            {"class": "footer"},
            "Tiny Earth Explorer © 2025 | Data visualization platform for soil microbiome analysis",
            tags.br(),
            "For more information, visit the Tiny Earth Initiative website",
        ),
    ),
)

# Define the server logic with enhanced visualizations
def server(input, output, session):
    @output
    @render.plot
    def zip_profile():
        try:
            selected_zip = input.selected_zip()
            
            # Filter data for the selected ZIP code
            zip_data = df[df["Zip Code"].astype(str) == selected_zip]
            
            if len(zip_data) == 0:
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
                plt.text(0.5, 0.5, f"No data available for ZIP code {selected_zip}", 
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.axis('off')
                return fig
            
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
            
            # Create bar chart with larger figure size and better colors
            fig, ax = plt.subplots(figsize=(20, 8), facecolor='white')
            
            # Create a custom colormap from blue to red
            colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(top_vars)))
            
            bars = ax.bar(top_vars, filtered_means[top_vars], color=colors)
            
            # Add a subtle grid for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Improve readability with larger fonts
            plt.xticks(rotation=20, ha='right', fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylabel('Average Frequency of Isolates', fontsize=14, fontweight='bold')
            plt.title(f'Microbiome Profile for ZIP Code {selected_zip}', 
                     size=16, fontweight='bold', pad=20)
            
            # Add value labels on top of bars
            y_max = ax.get_ylim()[1]
            for i, rect in enumerate(bars):
                height = rect.get_height()
                if height > 0.85 * y_max:
                    ax.text(rect.get_x() + rect.get_width()/2., height * 0.9,
                            f'{filtered_means[top_vars[i]]:.2f}',
                            ha='center', va='top', color='white', fontsize=12, fontweight='bold')
                else:
                    ax.text(rect.get_x() + rect.get_width()/2., height + 0.1,
                            f'{filtered_means[top_vars[i]]:.2f}',
                            ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # Add Temperature and CFUs/g as text annotations with better styling
            props = dict(boxstyle='round,pad=0.5', facecolor=MAIN_COLOR, alpha=0.8)
            text_str = f"Average Temperature: {temp_value:.2f}°C\nAverage CFUs/g: {cfu_value:.2f}"
            ax.text(0.95, 0.95, text_str,
                transform=ax.transAxes,
                fontsize=13,
                color='white',
                fontweight='bold',
                verticalalignment='top', 
                horizontalalignment='right',
                bbox=props)
            
            # Set frame color
            for spine in ax.spines.values():
                spine.set_color('#cccccc')
            
            plt.tight_layout(pad=5.0)
            return fig
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            plt.text(0.5, 0.5, f"Error in ZIP profile plot: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            return fig

    @output
    @render.plot
    def histogram():
        var = input.hist_var()
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        
        try:
            if var in categorical_columns:
                # For categorical variables, use a horizontal bar chart with custom color
                sns.countplot(y=df[var], color=MAIN_COLOR, ax=ax)
                plt.title(f"Distribution of {var}", pad=15, fontsize=16, fontweight='bold')
                plt.xlabel("Count", labelpad=10, fontsize=14, fontweight='bold')
                plt.ylabel(var, labelpad=10, fontsize=14, fontweight='bold')
                
                # Add count labels to bars
                for i, p in enumerate(ax.patches):
                    width = p.get_width()
                    ax.text(width + 1, p.get_y() + p.get_height()/2, 
                            f'{int(width)}', 
                            ha='left', va='center', fontsize=12)
            else:
                # For numerical variables, use a histogram with KDE
                sns.histplot(df[var], bins=20, kde=True, color=MAIN_COLOR, 
                            edgecolor="white", linewidth=1, alpha=0.7, ax=ax)
                plt.title(f"Distribution of {var}", pad=15, fontsize=16, fontweight='bold')
                plt.xlabel(var, labelpad=10, fontsize=14, fontweight='bold')
                plt.ylabel("Frequency", labelpad=10, fontsize=14, fontweight='bold')
                
                # Add mean and median markers
                mean_val = df[var].mean()
                median_val = df[var].median()
                plt.axvline(mean_val, color=ACCENT_COLOR, linestyle='dashed', linewidth=2, 
                           label=f'Mean: {mean_val:.2f}')
                plt.axvline(median_val, color=COMPLEMENTARY_COLOR, linestyle='dotted', linewidth=2, 
                           label=f'Median: {median_val:.2f}')
                plt.legend(fontsize=12)
            
            # Add grid for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Set frame color
            for spine in ax.spines.values():
                spine.set_color('#cccccc')
                
            plt.tight_layout(pad=2.0)
            return fig
        except Exception as e:
            plt.figure(figsize=(8, 6), facecolor='white')
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
            
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            
            # Create scatterplot with better styling
            scatter = sns.scatterplot(
                data=df, 
                x=x_var, 
                y=y_var, 
                hue="Zip Code" if "Zip Code" in df.columns else None,
                palette="viridis",
                alpha=0.7, 
                s=80,  # Larger point size
                edgecolor='w',  # White edge for better visibility
                linewidth=0.5,
                ax=ax
            )
            
            # Add regression line if both variables are numerical
            if x_var in true_numerical_columns and y_var in true_numerical_columns:
                try:
                    sns.regplot(
                        data=df, 
                        x=x_var, 
                        y=y_var, 
                        scatter=False,
                        color=ACCENT_COLOR, 
                        line_kws={"linestyle": "--", "linewidth": 2},
                        ax=ax
                    )
                    
                    # Calculate and display correlation
                    corr = df[x_var].corr(df[y_var])
                    
                    # FIXED: Move correlation text to bottom right inside the plot
                    # This ensures it's always visible within the chart boundaries
                    ax.text(0.98, 0.45, f'Correlation: {corr:.2f}', 
                           transform=ax.transAxes, 
                           fontsize=12,
                           ha='right',  # Right-aligned
                           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                except Exception as e:
                    print(f"Regression failed: {str(e)}")
            
            # Improve plot styling
            plt.title(f"Relationship: {x_var} vs {y_var}", pad=15, fontsize=16, fontweight='bold')
            plt.xlabel(x_var, labelpad=10, fontsize=14, fontweight='bold')
            plt.ylabel(y_var, labelpad=10, fontsize=14, fontweight='bold')
            
            # Add grid for better readability
            ax.grid(linestyle='--', alpha=0.3)
            
            # Set frame color
            for spine in ax.spines.values():
                spine.set_color('#cccccc')
                
            # Improve legend if present
            if scatter.get_legend() is not None:
                scatter.get_legend().set_title("ZIP Code")
                
            plt.tight_layout(pad=2.0)
            return fig
        except Exception as e:
            plt.figure(figsize=(8, 6), facecolor='white')
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
            
            # Create base scatter plot with enhanced styling
            fig = px.scatter(
                df,
                x=x_var,
                y=y_var,
                color="Zip Code",
                title=f"Interactive Data Explorer: {x_var} vs {y_var}",
                color_discrete_sequence=px.colors.qualitative.Bold,  # Use bold color scheme
                size_max=15,  # Maximum marker size
                opacity=0.75,  # Slight transparency
            )
            
            # Update hover template with customdata and styling
            fig.update_traces(
                customdata=custom_data,
                hovertemplate=hover_template,
                marker=dict(
                    line=dict(width=1, color='DarkSlateGrey')  # Add marker outline
                )
            )
            
            # Enhance layout styling
            fig.update_layout(
                plot_bgcolor='white',  # White background
                paper_bgcolor='white',  # White paper
                font=dict(family="Segoe UI, Arial, sans-serif", size=12),  # Better font
                title=dict(
                    font=dict(size=18, color=TEXT_COLOR),
                    x=0.5,  # Center title
                ),
                xaxis=dict(
                    title=dict(font=dict(size=14, color=TEXT_COLOR)),
                    gridcolor='#f0f0f0',  # Lighter grid
                    zerolinecolor='#e0e0e0',  # Zero line color
                ),
                yaxis=dict(
                    title=dict(font=dict(size=14, color=TEXT_COLOR)),
                    gridcolor='#f0f0f0',  # Lighter grid
                    zerolinecolor='#e0e0e0',  # Zero line color
                ),
                margin=dict(l=60, r=30, t=80, b=60),  # Better margins
                legend=dict(
                    title=dict(font=dict(size=14, color=TEXT_COLOR)),
                    bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent background
                    bordercolor='#e0e0e0',  # Border color
                    borderwidth=1,
                )
            )
            
            # Add trendline if both variables are numerical
            if x_var in true_numerical_columns and y_var in true_numerical_columns:
                try:
                    fig.update_layout(
                        shapes=[
                            dict(
                                type='line',
                                xref='x', yref='y',
                                x0=df[x_var].min(), y0=df[y_var].min(),
                                x1=df[x_var].max(), y1=df[y_var].max(),
                                line=dict(color=ACCENT_COLOR, width=2, dash='dash')
                            )
                        ]
                    )
                except Exception as e:
                    print(f"Trendline failed: {str(e)}")
            
            return ui.div(
                {"style": "height:500px;"},
                ui.HTML(
                    fig.to_html(full_html=False, include_plotlyjs='cdn')
                )
            )
        except Exception as e:
            return ui.div(
                {"style": "height:500px; display:flex; align-items:center; justify-content:center; background-color:white; border-radius:8px;"},
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

            # Create pie chart with enhanced styling
            fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
            
            # Custom color palette
            custom_colors = plt.cm.tab10(np.linspace(0, 1, len(pie_data)))
            
            # Create pie chart with shadow and better styling
            wedges, texts, autotexts = ax.pie(
                pie_data, 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=custom_colors,
                shadow=True,  # Add shadow for 3D effect
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'},  # White border around wedges
                textprops={'fontsize': 12, 'fontweight': 'bold'}  # Bold text
            )
            
            # Improve title styling
            ax.set_title('Distribution of Microbial Isolates', pad=20, fontsize=16, fontweight='bold')
            
            # Improve legend styling
            legend_labels = [f"{label} ({pie_data.iloc[i]:.0f} samples)" for i, label in enumerate(pie_data.index)]
            ax.legend(
                wedges, 
                legend_labels, 
                title="Isolate Categories", 
                loc="center left", 
                bbox_to_anchor=(1, 0.5), 
                fontsize=12,
                title_fontsize=14,
                frameon=True,  # Add frame around legend
                facecolor='white',  # White background
                edgecolor='#cccccc'  # Light gray border
            )
            
            # Add a circular equal aspect ratio
            ax.set_aspect('equal')
            
            plt.tight_layout(pad=4.0, rect=[0, 0, 0.85, 1])
            return fig
        except Exception as e:
            plt.figure(figsize=(14, 8), facecolor='white')
            plt.text(0.5, 0.5, f"Error in pie chart: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            return plt.gcf()

    @output
    @render.plot
    def new_bar_plot():
        try:
            fig, ax = plt.subplots(figsize=(20, 10), facecolor='white')
            
            # Get counts of each isolate type
            column_counts = df[columns_of_interest].apply(lambda x: (x > 0).sum())
            
            # Sort values for better visualization
            column_counts = column_counts.sort_values(ascending=False)
            
            # Create a colormap
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(column_counts)))
            
            # Create the bar chart with enhanced styling
            bars = ax.bar(
                column_counts.index, 
                column_counts.values, 
                color=colors,
                edgecolor='white',  # White edge
                linewidth=1.5,  # Edge width
                alpha=0.85  # Slight transparency
            )
            
            # FIXED: Adjust y-axis limits to ensure there's room for labels
            # Get the maximum count and add some margin (15%) for labels
            max_count = column_counts.max()
            ax.set_ylim(0, max_count * 1.15)  # Add 15% to ensure labels fit within the plot
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + (max_count * 0.02),  # Position the label slightly above the bar
                    f'{int(height)}',
                    ha='center', 
                    va='bottom', 
                    fontsize=12, 
                    fontweight='bold'
                )
            
            # Add a subtle grid for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Improve styling
            ax.set_title('Frequency of Isolate Types Across All Samples', pad=15, fontsize=16, fontweight='bold')
            ax.set_xlabel('Isolate Categories', labelpad=20, fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Samples', labelpad=10, fontsize=14, fontweight='bold')
            plt.xticks(rotation=20, ha='right', fontsize=12)
            
            # Set frame color
            for spine in ax.spines.values():
                spine.set_color('#cccccc')
                
            plt.tight_layout(pad=5.0)
            return fig
        except Exception as e:
            plt.figure(figsize=(20, 10), facecolor='white')
            plt.text(0.5, 0.5, f"Error in bar plot: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            return plt.gcf()

# Combine UI and server into a Shiny app
app = App(app_ui, server)