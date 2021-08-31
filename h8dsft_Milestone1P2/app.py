from os import link
import flask
from flask.globals import request
from flask import Flask, render_template
# library used for prediction
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
# library used for insights
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__, template_folder = 'templates')

link_active = None
# render home template
@app.route('/')
def main():
    return(render_template('home.html', title = 'Home'))

# load nn model    
model = load_model('model/final_model.h5')
# load pickle file
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route('/form')
def form():
    show_prediction = False
    link_active = 'Form'
    return(render_template('form.html', title = 'Form', show_prediction = show_prediction, link_active = link_active))

@app.route('/insights')
def insights():
    link_active = 'Insights'

    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = df.replace(' ', np.NaN)
    df['TotalCharges'] = df['TotalCharges'].astype('float64')
    df.dropna(inplace = True)

    color_map = {'Yes': '#FFD700', 'No': '#6699CC'}

    fig1 = px.box(
        df, x = 'Churn', y='tenure', title='Customer churn by Tenure',
        color='Churn', color_discrete_map=color_map,
        labels = {
            "tenure": "Tenure"
        }
    )
    fig1.update_layout(legend_traceorder='reversed', showlegend=False)
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    fig2 = px.box(
        df, x = 'Churn', y='MonthlyCharges', title='Customer churn by Monthly Charges',
        color='Churn', color_discrete_map=color_map,
        labels = {
            "MonthlyCharges": "Monthly Charges"
        }
    )
    fig2.update_layout(legend_traceorder='reversed')
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    dist_contract = df.groupby(['Contract', "Churn"]).count()[["customerID"]]
    cat_group = df.groupby(['Contract']).count()[["customerID"]]
    dist_contract["percentage"] = dist_contract.div(cat_group, level = 'Contract') * 100
    dist_contract.reset_index(inplace = True)
    dist_contract.columns = ['Contract', "Churn", "count", "percentage"]
    dist_contract = dist_contract.sort_values(['Contract', 'Churn'], ascending=True)

    fig3 = px.bar(
        dist_contract, x = 'Contract', y='percentage', title='Customer churn by Type of Contract',
        color='Churn', color_discrete_map=color_map, barmode="group", range_y = [0, 100]
    )
    fig3.update_layout(legend_traceorder='reversed')
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    dist_payment_billing = df.groupby(['PaperlessBilling', 'PaymentMethod', 'Churn']).count()[['customerID']]
    cat_group2 = df.groupby(['PaperlessBilling']).count()[['customerID']]
    dist_payment_billing["percentage"] = dist_payment_billing.div(cat_group2, level = 'PaperlessBilling') * 100
    dist_payment_billing.reset_index(inplace = True)
    dist_payment_billing.columns = ['PaperlessBilling', 'PaymentMethod', 'Churn', 'count', 'percentage']

    fig4 = px.bar(
        dist_payment_billing, x="PaymentMethod", y="percentage", color="Churn", range_y = [0, 100],
        barmode="group", facet_col="PaperlessBilling", color_discrete_map=color_map,
        title = 'Customer Churn by Billing and Payment Method',
        labels = {
            "PaymentMethod": "Payment Method",
            "PaperlessBilling": "Paperless Billing"
        }
    )
    fig4.update_layout(legend_traceorder='reversed')
    graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

    df['UseInternetService'] = np.where(df['InternetService'] == 'No', 'No', 'Yes')

    dist_internet_service = df.groupby(['UseInternetService', 'InternetService', 'Churn']).count()[['customerID']]
    cat_group2 = df.groupby(['UseInternetService']).count()[['customerID']]
    dist_internet_service["percentage"] = dist_internet_service.div(cat_group2, level = 'UseInternetService') * 100
    dist_internet_service.reset_index(inplace = True)
    dist_internet_service.columns = ['UseInternetService', 'InternetService', 'Churn', 'count', 'percentage']
    
    fig5 = px.bar(
    dist_internet_service, x="InternetService", y="percentage", color="Churn", range_y = [0, 100],
        barmode="group", color_discrete_map=color_map,
        title = 'Customer Churn by Internet Service',
        labels = {
            "InternetService": "Internet Service"
        }
    )
    fig5.update_layout(legend_traceorder='reversed')
    graph5JSON = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

    fig6 = px.sunburst(
        dist_internet_service, path=['UseInternetService', 'InternetService', 'Churn'],
        values='percentage', color='Churn', color_discrete_map={'(?)':'#c0c0c0', 'Yes': '#FFD700', 'No': '#6699CC'},
        title = 'Customer Churn by Internet Service'
    )
    fig6.update_layout(legend_traceorder='reversed')
    graph6JSON = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

    return(render_template('insights.html', title = 'Insights', link_active = link_active, graph1JSON = graph1JSON, graph2JSON = graph2JSON, graph3JSON = graph3JSON, graph4JSON = graph4JSON, graph5JSON = graph5JSON, graph6JSON = graph6JSON))

@app.route('/evaluation')
def evaluation():
    link_active = 'Evaluation'

    model_performance = pd.read_csv('final_model_history_df.csv')
    fig7=px.line(model_performance[['loss']])
    fig8=px.line(model_performance[['val_loss']])
    fig9=px.line(model_performance[['auc']])
    fig10=px.line(model_performance[['val_auc']])
    fig11=px.line(model_performance[['recall']])
    fig12=px.line(model_performance[['val_recall']])
    fig13=px.line(model_performance[['accuracy']])
    fig14=px.line(model_performance[['val_accuracy']])

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Loss", "Area Under Curve", "Recall", "Accuracy"))

    for d in fig7.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'], line = dict(color='#808080', width=3), name = d['name'])), row=1, col=1)
    for d in fig8.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'], line = dict(color='black', width=3), name = d['name'])), row=1, col=1)
            
    for d in fig9.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'],  line = dict(color='#d2691e', width=3), name = d['name'])), row=1, col=2)
    for d in fig10.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'],  line = dict(color='#FFBF00', width=3), name = d['name'])), row=1, col=2)

    for d in fig11.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'], line = dict(color='#3EB489', width=3), name = d['name'])), row=2, col=1)
    for d in fig12.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'], line = dict(color='#006D5B', width=3), name = d['name'])), row=2, col=1)
            
    for d in fig13.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'],  line = dict(color='#4169e1', width=3), name = d['name'])), row=2, col=2)
    for d in fig14.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'],  line = dict(color='#191970', width=3), name = d['name'])), row=2, col=2)
    fig.update_layout(height=800, title_text="Sequential NN Model Performance")    

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return(render_template('evaluation.html', title = 'Model Evaluation', link_active = link_active, graphJSON = graphJSON))

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering prediction result.
    '''
    link_active = 'Result'
    show_prediction = True

    # retrieve data
    gender_male = request.form.get('gender_male')
    SeniorCitizen = request.form.get('SeniorCitizen')
    Partner_yes = request.form.get('Partner_yes')
    Dependents_yes = request.form.get('Dependents_yes')
    tenure = int(request.form.get('tenure'))
    Contract = request.form.get('Contract')
    PaperlessBilling_yes = request.form.get('PaperlessBilling_yes')
    MonthlyCharges = float(request.form.get('MonthlyCharges'))
    TotalCharges = float(request.form.get('TotalCharges'))
    PaymentMethod = request.form.get('PaymentMethod')
    PhoneService_yes = request.form.get('PhoneService_yes')
    MultipleLines = request.form.get('MultipleLines')
    InternetService_yes = request.form.get('InternetService_yes')
    InternetServiceType = request.form.get('InternetServiceType')
    OnlineSecurity = request.form.get('OnlineSecurity')
    OnlineBackup = request.form.get('OnlineBackup')
    DeviceProtection = request.form.get('DeviceProtection')
    TechSupport = request.form.get('TechSupport')
    StreamingTV = request.form.get('StreamingTV')
    StreamingMovies = request.form.get('StreamingMovies')



    # set previously known values for one-hot encoding
    known_Contract = ['month_to_month', 'one_year', 'two_year']
    known_PaymentMethod = ['bank_transfer_automatic', 'credit_card_automatic', 'electronic_check', 'mailed_check']
    known_MultipleLines = ['no', 'no_phone_service', 'yes']
    known_InternetServiceType = ['dsl', 'fiber_optic', 'no_internet_service']
    known_OnlineSecurity = ['no', 'no_internet_service', 'yes']
    known_OnlineBackup = ['no', 'no_internet_service', 'yes']
    known_DeviceProtection = ['no', 'no_internet_service', 'yes']
    known_TechSupport = ['no', 'no_internet_service', 'yes']
    known_StreamingTV = ['no', 'no_internet_service', 'yes']
    known_StreamingMovies = ['no', 'no_internet_service', 'yes']



    # encode the categorical value
    Contract_type = pd.Series([Contract])
    Contract_type = pd.Categorical(Contract_type, categories = known_Contract)
    Contract_input = pd.get_dummies(Contract_type, prefix = 'Contract', drop_first=True)

    PaymentMethod_type = pd.Series([PaymentMethod])
    PaymentMethod_type = pd.Categorical(PaymentMethod_type, categories = known_PaymentMethod)
    PaymentMethod_input = pd.get_dummies(PaymentMethod_type, prefix = 'PaymentMethod', drop_first=True)

    MultipleLines_type = pd.Series([MultipleLines])
    MultipleLines_type = pd.Categorical(MultipleLines_type, categories = known_MultipleLines)
    MultipleLines_input = pd.get_dummies(MultipleLines_type, prefix = 'MultipleLines', drop_first=True)

    InternetServiceType_type = pd.Series([InternetServiceType])
    InternetServiceType_type = pd.Categorical(InternetServiceType_type, categories = known_InternetServiceType)
    InternetServiceType_input = pd.get_dummies(InternetServiceType_type, prefix = 'InternetServiceType', drop_first=True)

    OnlineSecurity_type = pd.Series([OnlineSecurity])
    OnlineSecurity_type = pd.Categorical(OnlineSecurity_type, categories = known_OnlineSecurity)
    OnlineSecurity_input = pd.get_dummies(OnlineSecurity_type, prefix = 'OnlineSecurity', drop_first=True)

    OnlineBackup_type = pd.Series([OnlineBackup])
    OnlineBackup_type = pd.Categorical(OnlineBackup_type, categories = known_OnlineBackup)
    OnlineBackup_input = pd.get_dummies(OnlineBackup_type, prefix = 'OnlineBackup', drop_first=True)

    DeviceProtection_type = pd.Series([DeviceProtection])
    DeviceProtection_type = pd.Categorical(DeviceProtection_type, categories = known_DeviceProtection)
    DeviceProtection_input = pd.get_dummies(DeviceProtection_type, prefix = 'DeviceProtection', drop_first=True)

    TechSupport_type = pd.Series([TechSupport])
    TechSupport_type = pd.Categorical(TechSupport_type, categories = known_TechSupport)
    TechSupport_input = pd.get_dummies(TechSupport_type, prefix = 'TechSupport', drop_first=True)

    StreamingTV_type = pd.Series([StreamingTV])
    StreamingTV_type = pd.Categorical(StreamingTV_type, categories = known_StreamingTV)
    StreamingTV_input = pd.get_dummies(StreamingTV_type, prefix = 'StreamingTV', drop_first=True)

    StreamingMovies_type = pd.Series([StreamingMovies])
    StreamingMovies_type = pd.Categorical(StreamingMovies_type, categories = known_StreamingMovies)
    StreamingMovies_input = pd.get_dummies(StreamingMovies_type, prefix = 'StreamingMovies', drop_first=True)



    # concat new data
    onehot_result1 = list(pd.concat([MultipleLines_input], axis = 1).iloc[0])
    onehot_result2 = list(pd.concat([InternetServiceType_input], axis = 1).iloc[0])
    
    add_ons = pd.concat([OnlineSecurity_input, OnlineBackup_input, DeviceProtection_input, TechSupport_input, StreamingTV_input, StreamingMovies_input, Contract_input], axis = 1)
    no_internet_service = add_ons.columns[add_ons.columns.str.contains('no_internet_service')].tolist()
    add_ons.drop(columns=no_internet_service, inplace = True)
    onehot_result3 = list(add_ons.iloc[0])

    onehot_result4 = list(pd.concat([PaymentMethod_input], axis = 1).iloc[0])

    new_data = [[tenure, MonthlyCharges, TotalCharges, SeniorCitizen, gender_male, Partner_yes, Dependents_yes, PhoneService_yes] + onehot_result1 + [InternetService_yes] + onehot_result2 + onehot_result3 + [PaperlessBilling_yes] + onehot_result4]

    scaled_input = scaler.transform(new_data)
    prediction = np.where(model.predict(scaled_input)>0.5, 1, 0)

    
    if prediction == 1:
        prediction_churn = True
    else:
        prediction_churn = False

    output = {0: 'less likely to churn', 1: 'more likely to churn'}

    return render_template('form.html', title = 'Prediction', show_prediction = show_prediction, prediction_text = 'The Customer will {}.'.format(output[prediction[0][0]]), link_active = link_active, prediction_churn = prediction_churn)

if __name__ == '__main__':
    app.run(debug = True)
