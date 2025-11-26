from dash import Dash, dcc, html
import pandas as pd
import plotly.express as px

# Sample data
df = pd.DataFrame({
    "Country": ["USA", "Canada", "Mexico"],
    "Cases": [1000, 500, 700]
})

app = Dash(__name__)

fig = px.bar(df, x="Country", y="Cases", title="Sample Dashboard")

app.layout = html.Div(children=[
    html.H1("COVID-19 Dashboard"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    # 这里一定要用 app.run，而不是 app.run_server
    app.run(debug=True)
