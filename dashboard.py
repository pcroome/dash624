#!/usr/bin/env python3

# pip3 install scipy
# sudo apt-get install python3-scipy

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc


# Import necessary libraries

import pandas as pd
import numpy as np
import plotly.express as px

# Import TSNE module
from sklearn.manifold import TSNE

# Import KMEANS module
from sklearn.cluster import KMeans

# import DBSCAN module
from sklearn.cluster import DBSCAN

# Silhouette module import
from sklearn.metrics import silhouette_samples, silhouette_score

# Import Agglomerative Clustering module
from sklearn.cluster import AgglomerativeClustering

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])



# Load dataset
df = pd.read_csv("2023-02-08-DATA624-Assignment4-Data.csv")



# Figure 1 - Scatterplot Matrix

fig1 = px.scatter_matrix(df,
                       title="Figure 1: Scatter Matrix of the Provided Dataset",
                       width=600, 
                       height=800)
fig1.update_traces(diagonal_visible=False)



# Figure 2 -- t-SNE Tuning

# Optimal TSNE: 

ts = TSNE(
    perplexity=75,
    learning_rate="auto",
    n_iter=2000,
    init='pca',
).fit_transform(df)

fig2 = px.scatter(
    ts,
    x=0,
    y=1, 
    title="Figure 2: Hyperparameter-Tuned t-SNE For Two Dimensions"
)



# Figure 3 -- Optimal K-Means

# Re-running the original (best) K-Means solution

kmeans = KMeans(
    n_clusters = 4, 
    init = "k-means++", 
    max_iter= 100, 
    tol=0.0001,
).fit(
    df
)

# Silhouette score for K-Means
sscore_kmeans = silhouette_score(df, kmeans.labels_)

# Assigning labels based on model with n_clusters=4

# Initialize dataframe for storing labels
ldf = pd.DataFrame(kmeans.labels_)

# Rename column
ldf.rename(columns={0:'KM_labels'}, inplace=True)

# To (properly) plot using TSNE, I will 
# create one dataframe (tmp) which includes both
# the 2-D TSNE results and the KM labels

tmp = pd.concat([pd.DataFrame(ts),ldf['KM_labels']])

tmp['labels'] = ldf['KM_labels']


# TSNE plot to visualize the clusters identified 
# through KMEANS
fig3 = px.scatter(
    tmp,
    x=0,
    y=1,
    color='labels',
    title="Figure 3: Four Clusters Identified via K-Means Clustering"
)



# Figure 4 -- Optimal DBSCAN

df_db = df.copy()

# Tuning hyperparameters

best_eps = 0
best_mP = 0
best_sscore = 0

sscore_DBSCAN=[]


for e in np.arange(1, 5.25, 0.25):
    for m in np.arange(5, 50, 5):
        dbscan = DBSCAN(eps=e, min_samples=m).fit(df)
        
        silhouette_avg = silhouette_score(df, dbscan.labels_)
        
        if silhouette_avg > best_sscore:
            best_eps = e
            best_mP = m
            best_sscore = silhouette_avg
            df_db['Cluster'] = dbscan.labels_
        
        
        sscore_DBSCAN.append(silhouette_avg)
    

# Adding best DBSCAN labels to ldf

ldf['DBSCAN_labels'] = df_db['Cluster']

# To (properly) plot using TSNE, I will 
# create one dataframe (tmp) which includes both
# the 2-D TSNE results and the DBSCAN labels

tmp2 = pd.concat([pd.DataFrame(ts),ldf['DBSCAN_labels']])

tmp2['labels'] = ldf['DBSCAN_labels']

fig4 = px.scatter(
    tmp2,
    x=0,
    y=1,
    color='labels',
    title="Figure 4: Four Clusters Identified via DBSCAN Clustering"
)



# Figure 5 -- Optimal Agglomerative Clustering

# Creating df for Agg Clustering
df_ac = df.copy()

best_nc = 0
best_l = 0
best_sscore = 0

sscore_AC=[]


for nc in np.arange(2, 10, 1):
    for l in ['ward', 'single', 'complete']:
        aggclust = AgglomerativeClustering(
            n_clusters = nc, # Number of clusters to find
            linkage=l
        ).fit(df)
        
        silhouette_avg = silhouette_score(df, aggclust.labels_)
        
        if silhouette_avg > best_sscore:
            best_nc = nc
            best_l = l
            best_sscore = silhouette_avg
            df_ac['Cluster'] = aggclust.labels_
        
        
        sscore_AC.append(silhouette_avg)
        

# Adding best AggClus labels to ldf

ldf['AggClus_labels'] = df_ac['Cluster']


# To (properly) plot using TSNE, I will 
# create one dataframe (tmp) which includes both
# the 2-D TSNE results and the AggClus labels

tmp3 = pd.concat([pd.DataFrame(ts),ldf['AggClus_labels']])

tmp3['labels'] = ldf['AggClus_labels']

fig5 = px.scatter(
    tmp2,
    x=0,
    y=1,
    color='labels',
    title="Figure 5: Four Clusters Identified via Agglomerative Clustering"
)


# Figure 6 -- FINAL

fig6 = px.scatter(
    tmp,
    x=0,
    y=1,
    color='labels',
    title="Figure 6: Final Four-Cluster Solution Via K-Means Clustering"
)





## ACTUAL PAGE CONTENT

# Textual and graphic items
app.layout = html.Div(
    [
        html.H1("Assignment 4 --  Exploratory Data Visualization & Clustering", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        html.P("Welcome to Paul Croome's results from Assignment 4. On this page, you will find a polished version of the results detailed in my code submitted via D2L. I have streamlined the original material in order to save both your time and the real estate on this webpage.", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        html.H2("Initial Exploration", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
         html.P("""Exploring the Profiling output, I made the following observations:
- The dataset contains 15 variables with 6170 observations and zero missing values; 
- Each variable contains values which are real numbers (floats); 
- Variables with similar distributions include: 
    - a) variables 0 and 1 (each has a roughly symmetric distribution about 0.0, with a sharp spike in values just above and below 0.0, and tails stretching to roughly +/- 5.0); 
    - b) variables 2, 3, and 4 (each has a roughly symmetric distribution about 0.0, with a sharp spike in values at or near 0.0 and tails stretching to roughly +/- 4.0); 
    - c) variable 5 (having a small and more distributed peak near 0.0, and sharp spikes in values near +/- 5.5
    - d) variables 6, 8, and 9 (each having a roughly symmetric distribution about 0.0, with a sharp peak at 0.0 and tails stretching only to about +/-1.0 or +/-2.0. 
    - e) variable 7 (having a small and more distributed peak near -2.5 and a single sharp spike in values at about 5.5. 
    - f) variables 0.1, 1.1, and 2.1 (each has a roughly normal distribution about 0.0, with tails stretching to roughly +/-3.0). 
    - g) variables 3.1 and 4.1 (where 100% of the values for 3.1 are 0, and 90% of the values for 4.1 are 0, the remaining 10% being 1). 


Assuming that highly correlated variables will likely represent distinct clusters, I anticipate that I will find roughly 7 different clusters by the conclusion of the assignment. However, more investigation is certainly required. 
""", 
                style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        dcc.Graph(
            figure=fig1,
            style={
                "width": "80%",
                "height": "80vh",
            },
            id="FirstFigure"
        ),
        html.P("""From the scatterpolot matrix above (Figure 1), we can see that some general groups of variables (where the variables display similar relationships with other variables, within similar ranges) emerge: 
- Variables 1-4, 6, 0.1, 1.1, and 2.1.
- Variables 8 and 9. 
- Variables 3.1 and 4.1. 
- Variables 7 and 5 appear to occupy their own groupings, in terms of their manner of association with other variables and their ranges. 

Examined in this way, it appears that there is likely to be either 4 or 5 emergent clusters. This is in contrast to the previous profiling results and histogram investigations.""", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        
        html.H2("t-SNE Tuning", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        html.P("""To visualize the results of my clustering attempts with the data, I used t-SNE to reduce the dimensionality of the dataset to 2. In order to ensure that clusters were optimally represented, I tuned the t-SNE hyperparameters (using trial-and-error) and determined that the optimal hyperparameter values were perplexity=75, n_iter=2000, and init='pca'. See Figure 2 for the clustering results with the optimal t-SNE parameters, and note how there appear to be, reasonably, about about 4-7 identifiable clusters.""", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        dcc.Graph(
            figure=fig2,
            style={
                "width": "80%",
                "height": "80vh",
            },
        ),
        
        html.H2("K-Means Clustering", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        html.P("""After using both the elbow method (with inertia) and the silhouette score method to tune the n_clusters hyperparamter (using different values for the algorithm's maximum iterations (max_iter) and tolerance (tol)), I found that the K-Means algorithm consistently identified 4 clusters as producing the best clustering results. This agrees with the groupings identified by examining the scatterplot matrix in Figure 1, and also is a reasonable solution, when examining the optimal t-SNE clustering. The silhouette score for this solution was about 0.6096""", 
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
        
        html.H2("DBSCAN Clustering", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        html.P("""Attempting DBSCAN clustering, I used the silhouette score to tune the epsilon (eps) and minimum points (min_samples) hyperparameters. This method of clustering, like K-Means ended up producing a similar 4-cluster solution, now with 'outliers' identified as well. The optimal epsilon and minimum points in this algorithm were 3.5 and 40, respectively, and the silhouette score for the optimal DBSCAN result was about 0.5702. 
        """, 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        dcc.Graph(
            figure=fig4,
            style={
                "width": "80%",
                "height": "80vh",
            },
        ),
        
        html.H2("Agglomerative Clustering", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        html.P("""Finally, conducting agglomerative clustering, I used the silhouette method to tune the hyperparameters of the number of clusters (n_clusters) and linkage type (link). Once again, I found that the optimal number of clusters was 4. In addition, the optimal linkage type was 'ward' and the silhouette score for the optimal agglomerative clustering solution was 0.6084. 
        """, 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        dcc.Graph(
            figure=fig5,
            style={
                "width": "80%",
                "height": "80vh",
            },
        ),
        
        html.H2("Final Determination", 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        html.P("""Considering together all of the clustering results, I submit the 4-cluster solution as the best solution in Assignment 4. In particular, I submit the 4-cluster solution discovered using the K-Means algorithm, as it has the highest silhouette score (at 0.6096) and is the most visually well-clustered solution. 
        """, 
               style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center'}),
        dcc.Graph(
            figure=fig6,
            style={
                "width": "80%",
                "height": "80vh",
            },
        ),
        
        
        
        
        html.P("What do you all machine learning completed in Canada? Eh-I!",
              style={
                   'margin': '10px', 
                   'width': '80%', 
                   'text-align': 'center', 
                   'color': 'grey'})
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
    
    
    
