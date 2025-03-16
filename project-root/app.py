from shiny import App, ui, render
import pandas as pd
import plotly.express as px

# Load the dataset
data = pd.read_csv("data/tiny_earth_2024.csv")

# Define the UI
app_ui = ui.page_fluid(
    ui.h1("Interactive Data Visualizations"),
    
    # Correctly use ui.page_sidebar() for layout
    ui.page_sidebar(
        sidebar=ui.sidebar(
            ui.input_select("x_var", "Select X-axis variable:", choices=list(data.select_dtypes(include='number').columns)),
            ui.input_select("y_var", "Select Y-axis variable:", choices=list(data.select_dtypes(include='number').columns))
        ),
        main=ui.layout_main(
            ui.output_plot("histogram_plot"),
            ui.output_plot("scatter_plot")
        )
    )
)

# Define the server logic
def server(input, output, session):
    
    @output
    @render.plot
    def histogram_plot():
        fig = px.histogram(data, x=input.x_var(), nbins=30, opacity=0.75)
        fig.update_layout(bargap=0.1)  # Ensure bins are mostly touching
        return fig
    
    @output
    @render.plot
    def scatter_plot():
        fig = px.scatter(data, x=input.x_var(), y=input.y_var(), trendline="ols")
        fig.update_layout(title="Scatter Plot", xaxis_title=input.x_var(), yaxis_title=input.y_var())
        return fig

# Create and run the app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run
