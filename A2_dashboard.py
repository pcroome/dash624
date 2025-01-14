#!/usr/bin/env python3

# pip3 install scipy
# sudo apt-get install python3-scipy

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import numpy as np

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


def load_data(url):
    """
    Load data from a shared google drive csv
    :param url: the shared url string
    :returns: a pandas dataframe
    """
    file_id = url.split("/")[-2]
    dwn_url = "https://drive.google.com/uc?id=" + file_id
    df = pd.read_csv(dwn_url)
    return df

# Loading data
url = ""
sdf = load_data(url)

# Including only variables of interest
varsinterest = ['patient', 'Age', 'Gender', 'ICULOS', 'Unit1', 'Unit2', 'SepsisLabel']

sdf1 = sdf[varsinterest]


# Selecting only the last row for all individual patients, based
# on their total length of stay
sdf_con1 = sdf1.sort_values('ICULOS', ascending=False).drop_duplicates(['patient'])


# Fill NaN values with 0 so that we can more 
# easily assign general ICU label

sdf_con1 = sdf_con1.fillna(0)


# Identifying patients not in either MICU or SICU!

sdf_con1['GenICU'] = np.where(np.logical_and((sdf_con1['Unit1'] == 0.0), (sdf_con1['Unit2'] == 0.0)), 1, 0)


# Create new unit-label column

conditions = [
    sdf_con1['Unit1'] == 1, 
    sdf_con1['Unit2'] == 1, 
    sdf_con1['GenICU'] == 1
]

icus = ['MICU', 'SICU', 'GenICU']

sdf_con1['UnitLabel'] = np.select(conditions, icus, default = 0)



# Assigning gender labels
conditions_gender =[
    sdf_con1['Gender'] == 0,
    sdf_con1['Gender'] == 1,
]

gender = ['Female', 'Male']
sdf_con1['Sex'] = np.select(conditions_gender, gender, default = 0)



# Drop unnecessary columns

sdf_icu2 = sdf_con1[['patient','Age', 'ICULOS','SepsisLabel','UnitLabel', 'Sex']]


### Figure 1 - Paul

# Density plots -- new version

fig1a = go.Figure()

# Add histogram for sepsis negative patients
fig1a.add_trace(go.Histogram(
        x=sdf_icu2.ICULOS[sdf_icu2['SepsisLabel'] == 0], 
        histnorm='probability density',
        name='Sep(-)',
        xbins=dict(
            start=0.0,
            end=350.0,
            size=5
        )
    )
    
)

# Add histogram for sepsis positive patients
fig1a.add_trace(go.Histogram(
        x=sdf_icu2.ICULOS[sdf_icu2['SepsisLabel'] == 1], 
        histnorm='probability density',
        name='Sep(+)',
        xbins=dict(
            start=0.0,
            end=350.0,
            size=5
        )
    )
)

# Overlay histograms and update title/axes
fig1a.update_layout(
    barmode='overlay',
    title_text="Density Plot of Length of Stay Based on Sepsis Status",
    xaxis_title_text='ICU Length of Stay (hours)',
    yaxis_title_text='Probability Density',
)

fig1a.update_traces(opacity=0.75)



### Figure 2 - Jason

vio = px.violin(
    sdf_icu2,
    x = 'UnitLabel',
    y = 'ICULOS',
    color = 'Sex',
    labels = {
        "ICULOS": "ICU Length of Stay (hours)",
        "Sex": "Gender"
    },
    color_discrete_sequence=["orange", "blue"], # can use this to change the colour
    box=True,
    facet_col="Sex",
)
vio.update_layout(title="ICU Length of Stay by ICU Type and Gender")





# Josh's variable preparation

data = [['MICU', 'Under 20',6.25], 
    ['MICU', '20-30',9.59],
    ['MICU', '31-40',8.49],
    ['MICU', '41-50',9.53],
    ['MICU', '51-60',10.5],
    ['MICU', '61-70',13.3],
    ['MICU', '71-80',11.4],
    ['MICU', 'Over 80',9.83],
        
        ['SICU', 'Under 20',0], 
    ['SICU', '20-30',0],
    ['SICU', '31-40',4.61],
    ['SICU', '41-50',4.27],
    ['SICU', '51-60',3.55],
    ['SICU', '61-70',4.0],
    ['SICU', '71-80',4.17],
    ['SICU', 'Over 80',4.65],
        
          ['General', 'Under 20',6.0],
    ['General', '20-30',9.83],
    ['General', '31-40',8.80],
    ['General', '41-50',10.2],
    ['General', '51-60',10.2],
    ['General', '61-70',10.3],
    ['General', '71-80',12.2],
    ['General', 'Over 80',9.52]]
  
df = pd.DataFrame(data, columns=['ICU Admittance', 'Age', "Sepsis Positive (%)"])
df.style.format({"Sepsis Positive (%)": "{:30,.2f}%"})



### Figure 3 - Josh

fig3 = px.bar(df, x="Age", color="ICU Admittance",
             y='Sepsis Positive (%)',
             title="<b>Sepsis Positivity Rate by Age & ICU Admittance<b>", category_orders={"ICU Admittance": ["SICU", "General", "MICU"]},
             barmode='group', text_auto = True, 
             height=700,
             facet_col="ICU Admittance",
             color_discrete_map={
                'MICU': 'cornflowerblue',
                'SICU': 'coral',
                 'General':'black'
                }
            )
fig3.update_layout(
    font_family="Times New Roman",
    font_color="Black",
    title_font_family="Times New Roman",
    title_font_color="Black",
    legend_title_font_color="Black"
    
)  
fig3.update_layout(
    title={
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig3.update_traces(width=1)
fig3.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)




## ACTUAL PAGE CONTENT

# Textual and graphic items
app.layout = html.Div(
    [
        html.H1("Assignment 2 -- Exploring Sepsis Data", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        html.P("Welcome to Paul, Josh, and Jason's exploration of the sepsis data! Figure one was completed by Paul; figure 2 by Jason; and figure 3 by Josh.", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        html.H2("Figure 1", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        dcc.Graph(
            figure=fig1a,
            style={
                "width": "80%",
                "height": "80vh",
            },
            id="OurFirstFigure"
        ),
        html.H2("Figure 2", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        dcc.Graph(
            figure=vio,
            style={
                "width": "80%",
                "height": "80vh",
            },
        ),
        html.H2("Figure 3", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        dcc.Graph(
            figure=fig3,
            style={
                "width": "80%",
                "height": "80vh",
            },
        ),
        html.P("We do not appreciate python anywhere, any way, or at any time.",
              style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center', 
                   'color': 'grey'})
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
    
    
    
