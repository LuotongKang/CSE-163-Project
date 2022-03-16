"""
CSE 163 Group Project
Luotong Kang, Yiyang Chen

A function file that does visualization for question 2.
Uses sklearn.svm to regress, and plotly to plot.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR


def melt_and_merge(GDP, age_diff, start, end):
    """
    Preprocesses GDP and age_diff dataframes. Takes start/end date as string.
    Returns graphing dataframe with rows containing age 18 boys' mean height
    for each country and each country's GDP per capita in corresponding years.
    """
    melted = GDP.melt(id_vars=["Country Name_clean"],
                      value_vars=[str(i) for i in range(start, end)])
    age_diff["ind"] = age_diff['Country_clean'] + age_diff['Year'].astype(str)
    melted["ind"] = melted["Country Name_clean"] + \
        melted["variable"].astype(str)
    boys_18_flr = (age_diff["Age group"] == 18) & (age_diff["Sex"] == "Boys")
    df = age_diff[boys_18_flr].merge(melted, left_on="ind", right_on="ind")
    df = df.dropna()
    return df


def regression_and_plot(df):
    """
    3D regression: independent Year and GDP per capita; dependent Mean height
    Visualize the regression and save the interactive graph as html
    """
    mesh_size = 1  # smaller will be accurate, but will cause RAM overload
    margin = 0
    X = df[['value', 'Year']]
    y = df['Mean height']

    # Condition the model on Year and GDP per capita, predict Mean height
    model = SVR(C=1.)
    model.fit(X, y)

    # Create a mesh grid on which we will run our model
    x_min, x_max = X.value.min() - margin, X.value.max() + margin
    y_min, y_max = X.Year.min() - margin, X.Year.max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)

    # Run model (Be careful of crashing)
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)

    # Generate the plot and save
    fig = px.scatter_3d(df, x='value', y='Year', z='Mean height')
    fig.update_traces(marker=dict(size=1))
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred,
                   name='pred_surface'))
    fig.write_html("3d_regression.html")
