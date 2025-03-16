import plotly.express as px

def create_histogram(data, variable):
    fig = px.histogram(data, x=variable, nbins=50, barmode='overlay')
    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
    return fig

def create_scatterplot(data, x_var, y_var):
    fig = px.scatter(data, x=x_var, y=y_var)
    return fig
