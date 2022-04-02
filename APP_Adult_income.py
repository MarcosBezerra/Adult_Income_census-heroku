#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports
# -------
# Dash-Plotly
import dash
import dash_core_components as dcc
#import dash_html_components as html
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from category_encoders.woe import WOEEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

# System
from joblib import load

# Modeling
from pandas import DataFrame
# from category_encoders.woe import WOEEncoder

# Components
# ----------

# Set stylesheetgs
yeti = dbc.themes.YETI

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[yeti])
server = app.server
# Inputs

slider_neuronio = dbc.FormGroup([
	dbc.Col(
		[
			dbc.Label("Neurônios", html_for="slider_neuronio", style={"fontWeight": "bold"}),
			dcc.Slider(
				id="slider_neuronio",
				min=108, max=500, step=1, value=108,
				marks={
					17: {"label": "108"},
					90: {"label": "500"}
				},
				tooltip={"always_visible": False, "placement": "bottom"}
			)
		],
		width=12
	)
])

dropdown_activation = dbc.FormGroup([
	dbc.Col(
		[
			dbc.Label("Função de ativação", html_for="dropdown_activation", style={"fontWeight": "bold"}),
			dbc.Select(
			id="dropdown_activation",
			options=[
				{"label": 'relu', "value": 'relu'},
				{"label": 'elu', "value": 'elu'},
				{"label": 'sigmoid', "value": 'sigmoid'},
			]
		)
	],
	width=8
	)
])

radio_inicializador = dbc.FormGroup([
	dbc.Col(
		[
			dbc.Label("inicializador", html_for="radio_inicializador", style={"fontWeight": "bold"}),
			dbc.RadioItems(
				id="radio_inicializador",
				className="form-check",
				labelClassName="form-check-label",
				inputClassName="form-check-input",
				options=[
					{"label": "he_normal", "value": "he_normal"},
					{"label": "uniform", "value": "uniform"},
				]
			)
		]
	)],
	className="form-group"
)

dropdown_otimizador = dbc.FormGroup([
	dbc.Col(
		[
			dbc.Label("Otimizador", html_for="dropdown_otimizador", style={"fontWeight": "bold"}),
			dbc.Select(
				id="dropdown_otimizador",
				options=[
					{"label": 'adam', "value": 'adam'},
					{"label": 'SGD', "value": 'SGD'},
				]
			)
		],
		width=8
	)
])

input_paciencia = dbc.FormGroup([
	dbc.Col(
		[
			dbc.Label("Paciência", className="control-label", html_for="input_paciencia",
					  style={"fontWeight": "bold"}),
			dbc.InputGroup([
				dbc.Input(id="input_paciencia", className="form-control", type="number", min=15, max=25, step=1,value=20),
				dbc.InputGroupAddon("avg. hrs/wk", className="input-group-append", addon_type="append")
			]),
			dbc.FormText("Must be a whole number from 15-25")
		],
		width=8
	)
])

button_run = dbc.Col(
	dbc.Button("Run", id="button_run", color="primary", style={"margin-bottom": "10px"})
)

output_card = dbc.Card(
	[
		dbc.CardHeader("Resultado", style={"fontWeight": "bold"}),
		dbc.CardBody([
			html.H1("0", id="output_probability", className="card-title"),
			html.P(
				[
					"Temos que falar algo"
					" Aqui"
					" Neste momento ",
					html.A(
						"GitHub repo.",
						href="https://github.com/jmischung/classificationAndCharting_adultIncome",
						target="_blank"
					)
				],
				className="card-text"
			)
		])
	],
	color="light",
	outline=True
)

alert = dbc.Alert(
			[
				html.H5("Oops!", className="alert-heading"),
				html.P("One or more of the inputs hasn't been completed or is invalid...", className="mb-0")
			],
			className="alert alert-dismissable alert-danger",
			color="danger",
			dismissable=True,
			is_open=True,
			duration=6000)




# Layout
form = dbc.Form([slider_neuronio, dropdown_activation, radio_inicializador,
				 dropdown_otimizador, input_paciencia, button_run])
app.layout = dbc.Container(
	[
		html.H1("Adult Income | Probability of Earning More Than $50K Per Year"),
		html.Hr(),
		dbc.Row(
			[
				dbc.Col(form, md=4),
				dbc.Col(output_card, id="output_card", md=4)
			],
		)
	],
	fluid=True
)

# Interactivity

@app.callback(
    Output("output_probability", "children"),
    Input("button_run", "n_clicks"),
    State("slider_neuronio", "value"),
    State("dropdown_activation", "value"),
    State("radio_inicializador", "value"),
    State("dropdown_otimizador", "value"),
    State("input_paciencia", "value"))
     
def run_model(n_clicks,neuronios,ativacao,inicializador,optimizador,paciencia):

    filename = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    headers = ["age","workclass","fnlwgt","education","education-num", "marital-status","occupation",
         "relationship","race","sex", "capital-gain","capital-loss","hours-per-week","native-country","indice"]
    adult = pd.read_csv(filename, names = headers)

     # filling missing values
    col_names = adult.columns

    for c in col_names:
        adult[c] = adult[c].replace("?", np.NaN)

    adult = adult.apply(lambda x:x.fillna(x.value_counts().index[0]))    

    adult_data = adult.drop(columns = ['indice'])
    adult_label = adult.indice


    adult_cat_1hot = pd.get_dummies(adult_data)
    indice_map={' <=50K':1,' >50K':0}
    adult_label=adult_label.map(indice_map).astype(int)
    adult_label


    train_data, test_data, train_label, test_label = train_test_split(adult_cat_1hot, adult_label, test_size=0.2, random_state=4)

    scaler = StandardScaler()  

    # Fitting only on training data
    scaler.fit(train_data)  
    train_data = scaler.transform(train_data)  

    # Applying same transformation to test data
    test_data = scaler.transform(test_data) 

    X_train = train_data 
    X_test = test_data
    y_train = train_label
    y_test = test_label

    early_stopping = EarlyStopping(min_delta=0.001, # minimium amount of change to count as an improvement
                                   patience=paciencia, # how many epochs to wait before stopping
                                   restore_best_weights=True,
                                  )
    # create model
    model = Sequential()
    model.add(Dense(neuronios, input_dim=108, activation=ativacao, kernel_initializer=inicializador))
    model.add(Dropout(0.2))
    model.add(Dense(neuronios, activation=ativacao, kernel_constraint=maxnorm(3), kernel_initializer=inicializador))
    model.add(Dropout(0.2))
    model.add(Dense(neuronios, activation=ativacao, kernel_initializer=inicializador))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid', kernel_initializer="he_normal"))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer= optimizador, metrics=['accuracy'])

    # Fit the model
    history = model.fit(X_train, y_train,
                        validation_split=0.33,
                        batch_size=32,
                        epochs=5,
                        callbacks=[early_stopping],
                        verbose=0,  # turn off training log
                       )
    
    scores = model.evaluate(X_test, y_test)
    prob = "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
    return prob


# Main block
if __name__ == '__main__':
	app.run_server(debug=False)


# In[ ]:




