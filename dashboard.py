import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dash_table
import os
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
from prophet import Prophet
from prophet.plot import plot_plotly
warnings.filterwarnings("ignore")
from sqlalchemy import create_engine
import os

usuario = os.environ.get("DB_USER")
contraseña = os.environ.get("DB_PASS")
host = os.environ.get("DB_HOST")
puerto = os.environ.get("DB_PORT")
base_datos = os.environ.get("DB_NAME")

engine = create_engine(f"postgresql+psycopg2://{usuario}:{contraseña}@{host}:{puerto}/{base_datos}")

# Función para cargar datos desde PostgreSQL
def cargar_datos():
    query = "SELECT * FROM ipc_excel"
    df = pd.read_sql(query, engine)
    return df

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard del Proyecto Final "
server = app.server

# Cargar datos de inflación
df = cargar_datos()

# Componentes para el análisis de inflación
def crear_componentes_inflacion():
    if df.empty:
        return html.Div([
            html.H4("Error: No se pudieron cargar los datos de inflación"),
            html.P("Verifique que el archivo 'inflacion_procesada.xlsx' existe y es accesible.")
        ])
    
    return html.Div([
        html.H4("Análisis de IPC e Inflación en Colombia"),
        html.Div([
            dbc.Alert(
                "Este análisis permite visualizar la evolución del IPC y la inflación en Colombia por ciudades. "
                "Seleccione los parámetros deseados para personalizar la visualización.",
                color="info",
                className="mb-3"
            ),
        ]),
        
        html.Div([
            html.Div([
                html.Label("Seleccione una métrica:"),
                dcc.Dropdown(
                    id='metrica-dropdown',
                    options=[
                        {'label': 'IPC', 'value': 'IPC'},
                        {'label': 'Inflación Anual', 'value': 'Inflacion_Anual'}
                    ],
                    value='IPC'
                )
            ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '20px'}),
            
            html.Div([
                html.Label("Seleccione ciudades:"),
                dcc.Dropdown(
                    id='ciudad-dropdown',
                    options=[{'label': ciudad, 'value': ciudad} for ciudad in sorted(df['Ciudad'].unique())] if not df.empty else [],
                    value=sorted(df['Ciudad'].unique()) if not df.empty else [],
                    multi=True
                )
            ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '20px'}),
            
            html.Div([
                html.Label("Rango de fechas:"),
                dcc.RangeSlider(
                    id='fecha-slider',
                    min=df['Fecha'].dt.year.min() if not df.empty else 2000,
                    max=df['Fecha'].dt.year.max() if not df.empty else 2023,
                    step=1,
                    marks={year: str(year) for year in range(
                        df['Fecha'].dt.year.min() if not df.empty else 2000, 
                        (df['Fecha'].dt.year.max() if not df.empty else 2023)+1, 
                        5
                    )},
                    value=[
                        (df['Fecha'].dt.year.max()-10) if not df.empty else 2013, 
                        df['Fecha'].dt.year.max() if not df.empty else 2023
                    ]
                )
            ], style={'width': '60%', 'padding': '20px 0px'})
        ]),
        
        dcc.Graph(id='main-graph'),
        
        html.Div([
            html.H3("Estadísticas descriptivas"),
            html.Div(id='stats-table')
        ])
    ])

# Subtabs para la metodología
subtabs_metodologia = dcc.Tabs([
    dcc.Tab(label='a. Definición del Problema', children=[
        html.H4('a. Definición del Problema a Resolver'),
        html.Ul([
            html.Li('Tipo de problema: clasificación / regresión / agrupamiento / series de tiempo'),
            html.Li('Variable objetivo o de interés: Nombre de la variable')
        ])
    ]),
    dcc.Tab(label='b. Preparación de Datos', children=[
        html.H4('b. Preparación de los Datos'),
        html.Ul([
            html.Li('Limpieza y transformación de datos'),
            html.Li('División del dataset en entrenamiento y prueba o validación cruzada')
        ])
    ]),
    dcc.Tab(label='c. Selección del Modelo', children=[
        html.H4('c. Selección del Modelo o Algoritmo'),
        html.Ul([
            html.Li('Modelo(s) seleccionados'),
            html.Li('Justificación de la elección'),
            html.Li('Ecuación o representación matemática si aplica')
        ])
    ]),
    dcc.Tab(label='d. Evaluación del Modelo', children=[
        html.H4('d. Entrenamiento y Evaluación del Modelo'),
        html.Ul([
            html.Li('Proceso de entrenamiento'),
            html.Li('Métricas de evaluación: RMSE, MAE, Accuracy, etc.'),
            html.Li('Validación utilizada')
        ])
    ])
])

# Subtabs para resultados (CÓDIGO CORREGIDO)
subtabs_resultados = dcc.Tabs([
    dcc.Tab(label='a. EDA', children=[
        html.H4('a. Análisis Exploratorio de Datos (EDA)'),
        crear_componentes_inflacion()
    ]),
    dcc.Tab(label='b. Análisis Complementario', children=[
        html.H4('b. Análisis Complementario de Inflación'),
        html.Div([
            html.Div([
                html.H5("Distribución de Inflación por Ciudad", className="mb-3"),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': f'Año {year}', 'value': year} 
                            for year in sorted(df['Fecha'].dt.year.unique())] if not df.empty else [],
                    value=df['Fecha'].dt.year.max() if not df.empty else None,
                    placeholder="Seleccione un año"
                ),
                dcc.Graph(id='boxplot-ciudad')
            ], className="mb-4"),
            
            html.Div([
                html.H5("Comparación Estacional", className="mb-3"),
                html.Div([
                    html.Div([
                        html.Label("Seleccione métrica:"),
                        dcc.RadioItems(
                            id='metrica-heatmap',
                            options=[
                                {'label': 'IPC', 'value': 'IPC'},
                                {'label': 'Inflación Anual', 'value': 'Inflacion_Anual'}
                            ],
                            value='Inflacion_Anual',
                            inline=True
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '10px'}),
                    
                    html.Div([
                        html.Label("Seleccione ciudad:"),
                        dcc.Dropdown(
                            id='ciudad-heatmap',
                            options=[{'label': ciudad, 'value': ciudad} 
                                    for ciudad in sorted(df['Ciudad'].unique())] if not df.empty else [],
                            value=df['Ciudad'].unique()[0] if not df.empty and len(df['Ciudad'].unique()) > 0 else None,
                            placeholder="Seleccione una ciudad"
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '10px'})
                ]),
                html.Div([
                    html.Label("Seleccione rango de años:"),
                    dcc.RangeSlider(
                        id='years-heatmap',
                        min=df['Fecha'].dt.year.min() if not df.empty else 2000,
                        max=df['Fecha'].dt.year.max() if not df.empty else 2023,
                        step=1,
                        marks={year: str(year) for year in range(
                            df['Fecha'].dt.year.min() if not df.empty else 2000,
                            (df['Fecha'].dt.year.max() if not df.empty else 2023)+1,
                            5
                        )},
                        value=[
                            (df['Fecha'].dt.year.max()-5) if not df.empty else 2018,
                            df['Fecha'].dt.year.max() if not df.empty else 2023
                        ]
                    )
                ], style={'width': '60%', 'padding': '20px 0px'}),
                dcc.Graph(id='heatmap-estacional')
            ])
        ])
    ]),
    dcc.Tab(label='c. Visualización del Modelo', children=[  # ¡CORRECTO ANIDADO!
        html.Div([
            html.H4('Modelado Predictivo del IPC', className='mb-4'),
            dbc.Row([
                dbc.Col([
                    html.Label("Seleccione Ciudad:", className='mb-2'),
                    dcc.Dropdown(
                        id='modelo-ciudad',
                        options=[{'label': c, 'value': c} for c in sorted(df['Ciudad'].unique())],
                        value='Cali',
                        clearable=False
                    )
                ], md=4, className='mb-4'),
                dbc.Col([
                    html.Label("Seleccione Modelo:", className='mb-2'),
                    dcc.Dropdown(
                        id='tipo-modelo',
                        options=[
                            {'label': 'Prophet (Meta)', 'value': 'prophet'},
                        ],
                        value='prophet',
                        clearable=False
                    )
                ], md=4, className='mb-4'),
                dbc.Col([
                    html.Label("Acción:", className='mb-2'),
                    dbc.Button("Generar Predicciones", 
                              id='boton-prediccion', 
                              color="primary",
                              className='w-100')
                ], md=4, className='mb-4')
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-predicciones",
                        type="circle",
                        children=dcc.Graph(id='grafico-predicciones')
                    )
                ], md=8),
                dbc.Col([
                    html.Div(id='metricas-modelo', className='mb-4'),
                    html.Div(id='tabla-predicciones')
                ], md=4)
            ])
        ], className='p-3')
    ])
])
# Tabs principales
tabs = [
    dcc.Tab(label='1. Introducción', children=[
        html.H2('Introducción'),
        html.P('Aquí se presenta una visión general del contexto de la problemática, el análisis realizado y los hallazgos encontrados.'),
        html.P('De manera resumida, indicar lo que se pretende lograr con el proyecto')
    ]),
    dcc.Tab(label='2. Contexto', children=[
        html.H2('Contexto'),
        html.P('Descripción breve del contexto del proyecto.'),
        html.Ul([
            html.Li('Fuente de los datos: Nombre de la fuente'),
            html.Li('Variables de interés: listar variables-operacionalización')
        ])
    ]),
    dcc.Tab(label='3. Planteamiento del Problema', children=[
        html.H2('Planteamiento del Problema'),
        html.P('Describe en pocas líneas la problemática abordada.'),
        html.P('Pregunta problema: ¿Cuál es la pregunta que intenta responder el análisis?')
    ]),
    dcc.Tab(label='4. Objetivos y Justificación', children=[
        html.H2('Objetivos y Justificación'),
        html.H4('Objetivo General'),
        html.Ul([html.Li('Objetivo general del proyecto')]),
        html.H4('Objetivos Específicos'),
        html.Ul([
            html.Li('Objetivo específico 1'),
            html.Li('Objetivo específico 2'),
            html.Li('Objetivo específico 3')
        ]),
        html.H4('Justificación'),
        html.P('Explicación breve sobre la importancia de abordar el problema planteado y los beneficios esperados.')
    ]),
    dcc.Tab(label='5. Marco Teórico', children=[
        html.H2('Marco Teórico'),
        html.P('Resumen de conceptos teóricos (definiciones formales) claves relacionados con el proyecto. Se pueden incluir referencias o citas.')
    ]),
    dcc.Tab(label='6. Metodología', children=[
        html.H2('Metodología'),
        subtabs_metodologia
    ]),
    dcc.Tab(label='7. Resultados y Análisis Final', children=[
        html.H2('Resultados y Análisis Final'),
        subtabs_resultados
    ]),

    
    dcc.Tab(label='8. Conclusiones', children=[
        html.Div([
            html.H2("Conclusiones", className="mb-4"),
            
            html.Ul([
                html.Li([
                    html.Strong("Limitaciones: "),
                    """Al intentar implementar el modelo arima con la busqueda de los mejores
                    hiperparametros no se dieron los resultados esperados, no habia un buen ajuste a los datos 
                    reales y el modelo arima daba predicciones erroneas y por tanto se decidio implementar este modelo
                    prophet desarrolado por meta y facebook para predecir series temporales.Concluimos que el modelo tiene buenas predicciones."""
                ], className="mb-3"),

                html.Li([
                    html.Strong("Variables externas no consideradas: "),
                    """El modelo no incluye factores macroeconómicos como tasas de interés, 
                    desempleo o políticas gubernamentales que impactan el IPC."""
                ], className="mb-3"),
                
                html.Li([
                    html.Strong("Estacionalidad fija: "),
                    """La componente estacional anual asume patrones constantes, 
                    pero en realidad estos pueden variar por crisis económicas o eventos disruptivos."""
                ], className="mb-3"),
                
                html.Li([
                    html.Strong("Limitaciones computacionales: "),
                    """La optimización manual de parámetros SARIMA puede no encontrar 
                    la mejor combinación posible debido a restricciones de tiempo de cómputo."""
                ], className="mb-3"),
                
                html.Li([
                    html.Strong("Sensibilidad a outliers: "),
                    """Eventos atípicos como la pandemia COVID-19 (2020-2021) 
                    distorsionan las predicciones a mediano plazo."""
                ])
            ], className="list-unstyled"),
            
            html.Div([
                html.H4("Recomendaciones para Mejoras Futuras", className="mt-4"),
                html.Ol([
                    html.Li("Integrar datos de fuentes externas (ej: Banco de la República)"),
                    html.Li("Implementar modelos híbridos con redes neuronales LSTM"),
                    html.Li("Desarrollar un sistema de monitoreo de outliers automático"),
                    html.Li("Aumentar frecuencia de actualización a datos trimestrales")
                ])
            ], className="bg-light p-4 rounded")
        ], className="p-4")
    ]),
]


# Layout de la aplicación
app.layout = dbc.Container([
    html.H1("Dashboard del Proyecto Final ", className="text-center my-4"),
    dcc.Tabs(tabs)
], fluid=True)

# Imports adicionales para visualizaciones complementarias
import plotly.graph_objects as go
import numpy as np

# Callbacks para el análisis de inflación
@app.callback(
    [Output('main-graph', 'figure'),
     Output('stats-table', 'children')],
    [Input('metrica-dropdown', 'value'),
     Input('ciudad-dropdown', 'value'),
     Input('fecha-slider', 'value')]
)
def update_graph(metrica, ciudades, rango_anos):
    # Si no hay datos cargados, devolver un gráfico vacío y un mensaje
    if df.empty:
        fig = px.line(title="No hay datos disponibles")
        return fig, html.Div("No hay datos disponibles para generar estadísticas")
    
    # Filtrar datos
    filtro_fecha = (df['Fecha'].dt.year >= rango_anos[0]) & (df['Fecha'].dt.year <= rango_anos[1])
    filtro_ciudad = df['Ciudad'].isin(ciudades)
    df_filtrado = df[filtro_fecha & filtro_ciudad]
    
    # Título para la gráfica según la métrica
    titulo = f"Evolución de {'la Inflación Anual' if metrica == 'Inflacion_Anual' else 'el IPC'} por Ciudad ({rango_anos[0]}-{rango_anos[1]})"
    y_label = "Tasa de Inflación (%)" if metrica == 'Inflacion_Anual' else "IPC"
    
    # Crear gráfica
    fig = px.line(
        df_filtrado, 
        x='Fecha', 
        y=metrica, 
        color='Ciudad',
        title=titulo,
        labels={'Fecha': 'Año', metrica: y_label}
    )
    
    # Crear estadísticas personalizadas con valores máximos y mínimos
    stats_list = []
    for ciudad in ciudades:
        df_ciudad = df_filtrado[df_filtrado['Ciudad'] == ciudad]
        
        if not df_ciudad.empty:
            # Encontrar el valor máximo y su fecha
            max_idx = df_ciudad[metrica].idxmax()
            max_valor = df_ciudad.loc[max_idx, metrica]
            max_fecha = df_ciudad.loc[max_idx, 'Fecha']
            
            # Encontrar el valor mínimo y su fecha
            min_idx = df_ciudad[metrica].idxmin()
            min_valor = df_ciudad.loc[min_idx, metrica]
            min_fecha = df_ciudad.loc[min_idx, 'Fecha']
            
            stats_list.append({
                'Ciudad': ciudad,
                'Mes Máximo': max_fecha.strftime('%b %Y'),
                'Valor Máximo': max_valor,
                'Mes Mínimo': min_fecha.strftime('%b %Y'),
                'Valor Mínimo': min_valor
            })
    
    # Convertir a DataFrame para formato tabular
    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        
        # Formatear valores numéricos
        formato = '.2f' if metrica == 'Inflacion_Anual' else '.4f'
        stats_df['Valor Máximo'] = stats_df['Valor Máximo'].map(lambda x: f"{x:{formato}}")
        stats_df['Valor Mínimo'] = stats_df['Valor Mínimo'].map(lambda x: f"{x:{formato}}")
        
        # Crear tabla HTML
        stats_table = html.Table([
            html.Thead(html.Tr([html.Th(col) for col in stats_df.columns])),
            html.Tbody([
                html.Tr([
                    html.Td(stats_df.iloc[i][col]) for col in stats_df.columns
                ]) for i in range(len(stats_df))
            ])
        ], className='table table-striped', style={'width': '100%', 'border': '1px solid', 'margin-top': '10px'})
    else:
        stats_table = html.Div("No hay datos disponibles para generar estadísticas")
    
    return fig, stats_table

# Callback para boxplot por ciudad
@app.callback(
    Output('boxplot-ciudad', 'figure'),
    [Input('year-dropdown', 'value')]
)
def update_boxplot(year):
    if df.empty or not year:
        return px.box(title="No hay datos disponibles")
    
    # Filtrar datos por año
    df_year = df[df['Fecha'].dt.year == year]
    
    # Crear boxplot
    fig = px.box(
        df_year, 
        x='Ciudad', 
        y='Inflacion_Anual',
        title=f'Distribución de Inflación por Ciudad (Año {year})',
        labels={'Ciudad': 'Ciudad', 'Inflacion_Anual': 'Inflación Anual (%)'},
        color='Ciudad'
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'median descending'},
        showlegend=False
    )
    
    return fig

# Callback para heatmap estacional
@app.callback(
    Output('heatmap-estacional', 'figure'),
    [Input('metrica-heatmap', 'value'),
     Input('ciudad-heatmap', 'value'),
     Input('years-heatmap', 'value')]
)
def update_heatmap(metrica, ciudad, years_range):
    if df.empty or not ciudad:
        return px.imshow(pd.DataFrame([[0]]), title="No hay datos disponibles")
    
    # Filtrar datos por ciudad y rango de años
    year_min, year_max = years_range
    df_ciudad = df[(df['Ciudad'] == ciudad) & 
                  (df['Fecha'].dt.year >= year_min) & 
                  (df['Fecha'].dt.year <= year_max)]
    
    # Extraer mes y año
    df_ciudad['Mes'] = df_ciudad['Fecha'].dt.month_name()
    df_ciudad['Año'] = df_ciudad['Fecha'].dt.year
    
    # Ordenar meses correctamente
    meses_orden = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    # Crear pivote para heatmap (año vs mes)
    pivot = df_ciudad.pivot_table(
        index='Año', 
        columns='Mes', 
        values=metrica,
        aggfunc='mean'
    )
    
    # Reordenar columnas según el orden natural de los meses
    pivot = pivot.reindex(columns=[mes for mes in meses_orden if mes in pivot.columns])
    
    # Crear heatmap
    titulo = f"{'IPC' if metrica == 'IPC' else 'Inflación Anual'} por Mes y Año - {ciudad} ({year_min}-{year_max})"
    fig = px.imshow(
        pivot,
        text_auto='.2f' if metrica == 'Inflacion_Anual' else '.4f',
        aspect="auto",
        color_continuous_scale='Viridis' if metrica == 'IPC' else 'RdYlGn_r',
        title=titulo
    )
    
    fig.update_layout(
        xaxis_title="Mes",
        yaxis_title="Año",
        height=500
    )
    
    return fig
# Callback para modelos predictivos
@app.callback(
    [Output('grafico-predicciones', 'figure'),
     Output('metricas-modelo', 'children'),
     Output('tabla-predicciones', 'children')],
    [Input('boton-prediccion', 'n_clicks')],
    [State('modelo-ciudad', 'value'),
     State('tipo-modelo', 'value')]
)
def ejecutar_modelo(n_clicks, ciudad, modelo):
    if n_clicks is None:
        return go.Figure(), [], []
    
    try:
        # Preparar datos
        df_ciudad = df[df['Ciudad'] == ciudad][['Fecha', 'IPC']]
        df_ciudad = df_ciudad.rename(columns={'Fecha': 'ds', 'IPC': 'y'}).dropna()
        
        if len(df_ciudad) < 12:
            return (
                go.Figure(), 
                dbc.Alert("⚠️ Datos insuficientes (mínimo 12 meses)", color="warning"), 
                []
            )
        
        # Dividir datos
        train_size = int(len(df_ciudad) * 0.8)
        train, test = df_ciudad.iloc[:train_size], df_ciudad.iloc[train_size:]
        
        # Entrenar modelo
        if modelo == 'prophet':
            model = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.3)
            model.fit(train)
            future = model.make_future_dataframe(periods=len(test)+12, freq='M')
            forecast = model.predict(future)
            pred_test = forecast.iloc[-len(test)-12:-12]['yhat']
            pred_future = forecast[['ds', 'yhat']].tail(12)
            fig = plot_plotly(model, forecast)
        else:
            model = SARIMAX(train['y'], order=(1,1,1), seasonal_order=(1,0,1,12)).fit(disp=False)
            pred_test = model.get_forecast(steps=len(test)).predicted_mean
            pred_future = model.get_forecast(steps=12).predicted_mean
            fig = go.Figure([
                go.Scatter(x=train['ds'], y=train['y'], name='Entrenamiento'),
                go.Scatter(x=test['ds'], y=test['y'], name='Real'),
                go.Scatter(x=test['ds'], y=pred_test, name='Predicción Test'),
                go.Scatter(x=pred_future.index, y=pred_future, name='Futuro')
            ])
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(test['y'], pred_test))
        mape = mean_absolute_percentage_error(test['y'], pred_test) * 100
        
        # Crear tabla
        tabla = dash_table.DataTable(
            columns=[{'name': 'Fecha', 'id': 'ds'}, {'name': 'IPC Predicho', 'id': 'yhat'}],
            data=pred_future.reset_index().rename(columns={'index': 'ds'}).to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'}
        )
        
        # Crear tarjeta de métricas
        metricas = dbc.Card([
            dbc.CardHeader("Rendimiento del Modelo"),
            dbc.CardBody([
                html.P(f"RMSE: {rmse:.2f}", className="card-text"),
                html.P(f"MAPE: {mape:.2f}%", className="card-text")
            ])
        ])
        
        return fig, metricas, tabla
    
    except Exception as e:
        return (
            go.Figure(), 
            dbc.Alert(f"Error: {str(e)}", color="danger"), 
            []
        )
# Ejecutar la aplicación
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host="0.0.0.0", port=port)