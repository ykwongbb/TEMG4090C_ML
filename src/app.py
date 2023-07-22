#Importing Relevant Libraries
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
#Loading the Dataset
data = pd.read_csv("TEMG4940C - Assignment Dataset.csv", sep="\t")
print("Number of datapoints:", len(data)) # Show number of Data Rows in Dataset
data.head(50) # Print Top 50 Rows of the dataset for preview
#@title Q0. Generate Y - Predictor Value
## Before Proceeding, You may want to evaluate the Y -Predictor Value first
## By combining Y = AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5 + Response

data['Y'] = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + data['AcceptedCmp4'] + data['AcceptedCmp5'] + data['Response']
#@title Q1b. Handle the missing values and outliers, and explain why was that particular methodology chosen.
#@title Q1a. In whichever preferred means (tables, graphs etcs), showcase the existence of missing value, outliers & imbalances in the dataset.

# Q1a. Showcase existence of missing value, outliers & imbalances within dataset

# Missing Value 
print("Missing Value in Dataset:", data.isnull().sum().sum()) # Show number of Missing Value in Dataset

# Outliers
numerical_cols = data.select_dtypes(include = [np.number])
fig = go.Figure()
for col in numerical_cols:
    q25, q75 = np.percentile(data[col], [25, 75])
    iqr = q75 - q25
    min = q25 - 1.5 * iqr
    max = q75 + 1.5 * iqr
    outliers = data[(data[col] < min) | (data[col] > max)][col]
    if (len(outliers) > 0):
        fig.add_trace(go.Box(x = data[col], name = col, boxpoints = 'outliers', line_width = 0.7))
fig.update_layout(title='Distribution of numerical columns with outliers', width = 1000, height = 600)
fig.show()


# Imbalances    
data['Y'].value_counts() # Show number of Imbalances in Y - Predictor Value
data['NumWebPurchases'].value_counts() 
# Q1b. Codes to handle (i) Missing Values, (ii) Outliers
# Tips: You may refer to Slide 19 in Tutorial PPT
data_copy = data.copy()
# (ii) Outliers
for col in numerical_cols:
    q25, q75 = np.percentile(data_copy[col], [25, 75])
    iqr = q75 - q25
    min = q25 - 1.5 * iqr
    max = q75 + 1.5 * iqr
    data_copy.loc[data_copy[col] < min, col] = np.nan
    data_copy.loc[data_copy[col] > max, col] = np.nan

# (i) Missing Values
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors = 5) # k = 5
numerical_cols = data_copy.select_dtypes(include = [np.number]) # Select Numerical Columns
data_copy[numerical_cols.columns] = imputer.fit_transform(data_copy[numerical_cols.columns])  # Impute Numerical Columns
#@title Q1d. Plot 3 or more types of charts over all columns of data
# Q1d. Exploratory Data Analysis Graph Plotting

# import xxx as xxx (Import Graphing Libraries)
import dash
import dash_core_components as dcc
import dash_html_components as html
# Plotting Histogram, Boxplot & Violin Plot for all columns
figs = []
col = data_copy.columns
for j in range(3):
    fig = sp.make_subplots(rows=5, cols=6, subplot_titles = data_copy.columns)
    for i, col in enumerate(data_copy.columns):
        if (j == 0):
            fig.add_trace(go.Histogram(x = data_copy[col], name = col), row = i // 6 + 1, col =i % 6 +1) # histogram 
        elif (j == 1):         
            fig.add_trace(go.Box(x = data_copy[col], name = " "), row = i // 6 + 1, col =i % 6 +1) # boxplot
        else:
            fig.add_trace(go.Violin(x = data_copy[col], name = " "), row = i // 6 + 1, col =i % 6 +1) # violin plot
    fig.update_layout(title='Distribution of all columns', width = 2000, height = 1300, showlegend = False)
    figs.append(fig)
    fig.show()



# Define the Plotly Dash app
app = dash.Dash(__name__)
server = app.server
# Define the layout of the dashboard
app.layout = html.Div([
    html.H1('Machine Learning Project Dashboard'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Data Exploration', value='tab-1'),
        dcc.Tab(label='Model Evaluation', value='tab-2'),
        dcc.Tab(label='Model Insight Generation', value='tab-3')
    ]),
    html.Div(id='tab-content')
])

# Define the content of the data exploration tab
exploration_tab = html.Div([
    html.H2('Data Exploration'),
    dcc.Dropdown(
        id='plot-type',
        options=[
            {'label': 'Histogram', 'value': 0},
            {'label': 'Boxplot', 'value': 1},
            {'label': 'Violin plot', 'value': 2}
        ],
        value=0
    ),
    html.Div(id='plot-div')
])

# Define the content of the model evaluation tab
evaluation_tab = html.Div([
    html.H2('Model Evaluation'),
    # Add components to display model evaluation
])

# Define the content of the model insight generation tab
insight_tab = html.Div([
    html.H2('Model Insight Generation'),
    # Add components to display model insight generation
])

# Define the callbacks to display the content of each tab
@app.callback(
    dash.dependencies.Output('tab-content', 'children'),
    [dash.dependencies.Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-1':
        return exploration_tab
    elif tab == 'tab-2':
        return evaluation_tab
    elif tab == 'tab-3':
        return insight_tab
    else:
        return None

@app.callback(
    dash.dependencies.Output('plot-div', 'children'),
    [dash.dependencies.Input('plot-type', 'value')]
)
def update_plot(plot_type):
    if plot_type == 0:
        fig = figs[0]
    elif plot_type == 1:
        fig = figs[1]
    elif plot_type == 2:
        fig = figs[2]
    else:
        fig = None

    if fig is not None:
        graph = dcc.Graph(figure=fig)
        return graph
    else:
        return None
def update_plot(plot_type):
    # get the corresponding plot figure
    fig = figs[plot_type]
    # convert the figure to a Plotly graph object
    graph = dcc.Graph(figure=fig)
    # return the graph object to be displayed in the dashboard
    return graph

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8000)

#@title Q2b. Perform the aforementioned Data Preprocessings
# Q2b. Preform Data Preprocessing

# import xxx as xxx (Import panda Libraries)
# from sklearn.preprocessing import xxx (Import sklearn libraries)
from datetime import date
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import ADASYN

df = pd.read_csv("TEMG4940C - Assignment Dataset.csv", sep="\t")
df['Y'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']
df['Y'] = df['Y'].astype(int)
df.drop(['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response'], axis = 1, inplace = True)

# Feature Selection
df.drop(['ID','Z_CostContact', 'Z_Revenue'], axis = 1, inplace = True)

# Feature Creation, duarion
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format = '%d-%m-%Y')
df['Dt_Customer'] = (pd.Timestamp(date.today()) - df['Dt_Customer']).dt.days

# One-hot encoding
ohe = OneHotEncoder()
Marital_Status = ohe.fit_transform(df[['Marital_Status']])
df['Marital_Status'] = Marital_Status.toarray()
# df['Divorced'] =

# Rank Replacement
df['Education'] = df['Education'].replace({'Basic': 1, 'Graduation': 2, '2n Cycle': 3, 'Master': 4, 'PhD': 5})

# Missing value imputation
imputer = KNNImputer(n_neighbors = 3) # k = 3
df['Income'] = imputer.fit_transform(df[['Income']])

# Outliers
outliers = ['Year_Birth', 'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
for col in outliers:
    q25, q75 = np.percentile(df[col], [25, 75])
    iqr = q75 - q25
    min = q25 - 1.5 * iqr
    max = q75 + 1.5 * iqr
    df.loc[df[col] < min, col] = min
    df.loc[df[col] > max, col] = max
    
# Normalization
# Transpose the dataframe
from sklearn.preprocessing import MinMaxScaler
df_T = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].T
scaler = MinMaxScaler(feature_range=(0, 1)).fit(df_T)
normalized_X = scaler.transform(df_T)
df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']] = normalized_X.T

    
# Imbalanced data handling
adasyn = ADASYN(random_state = 42)
Y = df['Y']
x = df.drop(['Y'], axis = 1)
x, Y = adasyn.fit_resample(x, Y)

#@title Q3a. Generate Test/Training Data Split
# Q3a. Generate Test / Training Data Split

from sklearn.model_selection import train_test_split # Import Train Test Split From Libraries
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size= 0.2, random_state= 42)  

#@title Q3b. Choose and deploy dataset to 3+ ML Model
# Q3b. Choose 3 models to deploy dataset to
# Tips: Do not afraid to test out various models, to find out which model fits the datasets the best

# Step 1. Import Model

# Step 2. Fit Dataframe into Model

# Step 3. Generate Model Prediction (Y)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, Y_train)
gnb_pred = gnb.predict(x_test)
gnb_probi = gnb.predict_proba(x_test)
# gnb_score = gnb.predict_proba(x_test)[:, 1]
# gnb_probi_norm = gnb_probi / gnb_probi.sum(axis=1, keepdims=True)
        
# Multi-layer Perceptron
from sklearn.neural_network import MLPClassifier
dt = MLPClassifier()
dt.fit(x_train, Y_train)
dt_pred = dt.predict(x_test)
dt_probi = dt.predict_proba(x_test)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, Y_train)
rf_pred = rf.predict(x_test)
rf_probi = rf.predict_proba(x_test)

#@title Q4a. Evaluate the Model's Accuracy
from dash import dash_table
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, auc, roc_curve, roc_auc_score, confusion_matrix
import plotly.graph_objs as go
# (1) 
# Confusion matrices
cm_list = [metrics.confusion_matrix(Y_test, gnb_pred),
           metrics.confusion_matrix(Y_test, dt_pred),
           metrics.confusion_matrix(Y_test, rf_pred)]
name_list = ['Naive Bayes', 'Multi-layer Perceptron', 'Random Forest']
cm_display_list = [metrics.ConfusionMatrixDisplay(confusion_matrix=cm) for cm in cm_list]
fig = sp.make_subplots(rows = 1, cols = 3, subplot_titles = name_list, horizontal_spacing = 0.15)
cbarlocs = [.265, .63, 1]

for i in range(3):
    fig_list = [go.Figure(go.Heatmap(z=cm_display.confusion_matrix, colorscale='Viridis', hoverinfo='z', colorbar=dict(len=0.9, x=cbarlocs[i])), ) for cm_display in cm_display_list]
    fig.append_trace(fig_list [i].data[0], row=1, col=i+1)
    fig.update_xaxes(title_text='Predicted Class', row=1, col=i+1)
    fig.update_yaxes(title_text='True Class', row=1, col=i+1)
fig.update_layout(height=400, width=1200, title_text="Confusion Matrices", showlegend=False)
fig.show()


# (2)
# Accuracy Score
print('Accuracy score of Naive Bayes:', accuracy_score(Y_test, gnb_pred))
print('Accuracy score of Multi-layer Perceptron:', accuracy_score(Y_test, dt_pred))
print('Accuracy score of Random forest', accuracy_score(Y_test, rf_pred))

# Recall Score
print('Recall score of Naive Bayes:', recall_score (Y_test,gnb_pred, average='weighted'))
print('Recall score of Multi-layer Perceptron:', recall_score (Y_test,gnb_pred, average='weighted'))
print('Recall score of Random forest:', recall_score (Y_test, gnb_pred, average='weighted'))

#  AUC Score
for i in range(len(np.unique(Y))):
    rocauc_gnb = roc_auc_score(Y_test == i, gnb_probi[:, i])
    print('AUC score of Naive Bayes for output', i, ':', rocauc_gnb)
    rocauc_dt = roc_auc_score(Y_test == i, dt_probi[:, i])
    print('AUC score of Multi-layer Perceptron for output', i, ':',rocauc_dt)
    rocauc_rf = roc_auc_score(Y_test == i, rf_probi[:, i])
    print('AUC score of Random forest for output', i, ':', rocauc_rf)

# F1 Score
print('F1 score of Naive Bayes:', f1_score (Y_test, gnb_pred, average="weighted") )
print('F1 score of Multi-layer Perceptron:', f1_score (Y_test, dt_pred, average="weighted") )
print('F1 score of Random forest:', f1_score (Y_test, rf_pred, average="weighted") )

# (3)
# ROC Curve
gnb_fprs, gnb_tprs, dt_fprs, dt_tprs, rf_fprs, rf_tprs = [], [], [], [], [], []

for i in range(len(np.unique(Y))):
    fpr_gnb, tpr_gnb, thresholds_gnb = roc_curve(Y_test == i, gnb_probi[:, i])
    fpr_dt, tpr_dt, thresholds_dt = roc_curve(Y_test == i, dt_probi[:, i])
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(Y_test == i, rf_probi[:, i])
    gnb_fprs.append(fpr_gnb)
    gnb_tprs.append(tpr_gnb)
    dt_fprs.append(fpr_dt)
    dt_tprs.append(tpr_dt)
    rf_fprs.append(fpr_rf)
    rf_tprs.append(tpr_rf)

fig1 = go.Figure()
for i in range(len(np.unique(Y))):
    fig1.add_trace(go.Scatter(x=gnb_fprs[i], y=gnb_tprs[i], name='Output ' + str(i), mode='lines'))
fig1.update_layout(title='ROC Curve for Naive Bayes', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
fig1.show()

fig2 = go.Figure()
for i in range(len(np.unique(Y))):
    fig2.add_trace(go.Scatter(x=dt_fprs[i], y=dt_tprs[i], name='Output ' + str(i), mode='lines'))
fig2.update_layout(title='ROC Curve for Multi-layer Perceptron', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
fig2.show()

fig3 = go.Figure()
for i in range(len(np.unique(Y))):
    fig3.add_trace(go.Scatter(x=rf_fprs[i], y=rf_tprs[i], name='Output ' + str(i), mode='lines'))
fig3.update_layout(title='ROC Curve for Random Forest', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
fig3.show()


gnb_acc = accuracy_score(Y_test, gnb_pred)
gnb_prec = precision_score(Y_test, gnb_pred, average='weighted')
gnb_rec = recall_score(Y_test, gnb_pred, average='weighted')
gnb_f1 = f1_score(Y_test, gnb_pred, average='weighted')

dt_acc = accuracy_score(Y_test, dt_pred)
dt_prec = precision_score(Y_test, dt_pred, average='weighted')
dt_rec = recall_score(Y_test, dt_pred, average='weighted')
dt_f1 = f1_score(Y_test, dt_pred, average='weighted')

rf_acc = accuracy_score(Y_test, rf_pred)
rf_prec = precision_score(Y_test, rf_pred, average='weighted')
rf_rec = recall_score(Y_test, rf_pred, average='weighted')
rf_f1 = f1_score(Y_test, rf_pred, average='weighted')


table_data = [
    {
        'Model': 'Naive Bayes',
        'Accuracy': gnb_acc,
        'Precision': gnb_prec,
        'Recall': gnb_rec,
        'F1 Score': gnb_f1
    },
    {
        'Model': 'Multi-layer Perceptron',
        'Accuracy': dt_acc,
        'Precision': dt_prec,
        'Recall': dt_rec,
        'F1 Score': dt_f1
    },
    {
        'Model': 'Random Forest',
        'Accuracy': rf_acc,
        'Precision': rf_prec,
        'Recall': rf_rec,
        'F1 Score': rf_f1
    }
]
table_columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
table = dash_table.DataTable(
    data=table_data,
    columns=[{'name': col, 'id': col} for col in table_columns],
    
)

evaluation_tab = html.Div([
    html.H2('Model Evaluation'),
    dcc.Graph(figure=fig),
    dcc.Graph(figure=fig1),
    dcc.Graph(figure=fig2),
    dcc.Graph(figure=fig3),
    table
    ])

#@title Q6. Plot 3+ Graph that effectively explain insights that you uncovered about the trained model, comment to describe the insight that was uncovered.

# # Step 1. Import Graph Plotting Libraries (if any)
import base64
import plotly.graph_objs as go
import plotly.tools as tls
import matplotlib.pyplot as plt
# # (a) Feature Importance Plot
from yellowbrick.model_selection import FeatureImportances
from sklearn.ensemble import RandomForestClassifier
# Creating the feature importances plot
visualizer = FeatureImportances(RandomForestClassifier (max_depth=3), relative=True)
visualizer.fit (x_test, Y_test)
visualizer.show("Feature Importances Plot")
plt.savefig("feature_importance.png")
plt.show()
# pio.write_html(fig, 'feature_importances_plot.html')
# Limitation: It does not provide the direction of the relationship between the feature and the target variable
# Implications: The 3 most important features are Income, NumCatalogPurchases and NumStorePurchases
# Recommendations: Check whether income is the most decisive factor among all variables


# #(b) SHAP value Plot
import shap
# Fit the random forest classifier model
rf_b = RandomForestClassifier(max_depth=3)
rf_b.fit(x_train, Y_train)
# Create a SHAP explainer object using the fitted model and training data
explainer = shap.Explainer(rf_b.predict, x_test)
# Calculate SHAP values for a set of test data
shap_values = explainer(x_test)
# Visualize the SHAP values for a single observation using a summary plot
fig_1, ax = plt.subplots()
shap.summary_plot(shap_values, x_test, feature_names = x_test.columns,  plot_type='bar', show=False)
# Set the title of the Matplotlib axes object
plt.title("SHAP Values for Random Forest Classifier")
plt.savefig("SHAP_Values_for_Random_Forest_Classifier.png")
plt.show()

# Limitation: The computation is time-consuming
# Implications: The 3 most important features are Income, NumCatalogPurchases and Teenhome
# Recommendations: Check the importance of Kidhome to the model as the results shown in a and b are different

# (c) LIME Plot
import lime 
import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, feature_names = x_test.columns, class_names = ['0', '1', '2', '3', '4', '5'], random_state = 42)
exp = explainer.explain_instance(x_test.values[0], rf.predict_proba, num_features=10)
# Visualize the LIME explanation using Matplotlib
fig_lime = tls.mpl_to_plotly(exp.as_pyplot_figure())
fig_2 = exp.as_pyplot_figure()
ax = fig_2.gca()
ax.set_title("LIME Explanation Plot")
plt.savefig("LIME_Explanation_Plot.png")
plt.show()
# Limitation: A minor change in data can lead to a different explanation
# Implications: The 3 most important features are Income, NumCatalogPurchases and NumStorePurchases
# Recommendations: Check whether income is the most decisive factor among all variables
import plotly.express as px

insight_tab = html.Div([
    html.H2('Model Insight Generation'),
    
    html.H3("Feature Importances Plot"),
    html.Img(src="data:image/png;base64,{}".format(base64.b64encode(open("feature_importance.png", "rb").read()).decode())),
     html.H3("SHAP Values for Random Forest Classifier"),
    html.Img(src="data:image/png;base64,{}".format(base64.b64encode(open("SHAP_Values_for_Random_Forest_Classifier.png", "rb").read()).decode())),
    html.H3("LIME Explanation Plot"),
    html.Img(src="data:image/png;base64,{}".format(base64.b64encode(open("LIME_Explanation_Plot.png", "rb").read()).decode())),  
])
    