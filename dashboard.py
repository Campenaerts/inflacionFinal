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

usuario = "dataviz_db_a2qu_user"
contraseña = "aZ1oCNwaqUh2SruEvnFrNSO1Tp2Hk6ti"  
host = "dpg-d0oe98muk2gs73bp2g3g-a.oregon-postgres.render.com"
puerto = "5432"
base_datos = "dataviz_db_a2qu"

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
        html.Div(className="card border-0 shadow-sm p-4 mb-4", children=[
            html.H5('Tipo de Problema', className="text-primary"),
            html.Div(className="alert alert-info", children=[
                html.Strong('Series de Tiempo - Predicción Multivariada'),
                html.P('Este proyecto aborda un problema de predicción de series temporales multivariadas, donde se busca pronosticar los valores futuros del Índice de Precios al Consumidor (IPC) para múltiples ciudades colombianas simultáneamente.', className="mb-0 mt-2")
            ]),
            
            html.H5('Variable Objetivo', className="text-primary mt-4"),
            html.Div(className="row", children=[
                html.Div(className="col-md-6", children=[
                    html.H6('Variable Principal:', className="fw-bold"),
                    html.P('IPC (Índice de Precios al Consumidor) mensual por ciudad')
                ]),
                html.Div(className="col-md-6", children=[
                    html.H6('Dimensión Temporal:', className="fw-bold"),
                    html.P('Serie histórica desde 1979 hasta la actualidad')
                ])
            ]),
            
            html.H5('Características del Problema', className="text-primary mt-4"),
            html.Ul([
                html.Li([html.Strong('Horizonte de predicción: '), '12 meses hacia el futuro para cada ciudad']),
                html.Li([html.Strong('Granularidad: '), 'Datos mensuales agregados por ciudad']),
                html.Li([html.Strong('Cobertura geográfica: '), '7 ciudades principales de Colombia']),
                html.Li([html.Strong('Naturaleza de los datos: '), 'Series temporales con componentes estacionales, tendencia y ruido']),
                html.Li([html.Strong('Desafíos específicos: '), 'Volatilidad histórica, eventos extremos, heterogeneidad regional'])
            ]),
            
            html.H5('Justificación del Enfoque', className="text-primary mt-4"),
            html.P('La elección de un enfoque de series temporales se justifica por la naturaleza secuencial y temporal de los datos del IPC, donde los valores pasados contienen información valiosa para predecir valores futuros. La variabilidad regional requiere un modelo que pueda adaptarse a patrones locales específicos.')
        ])
    ]),
    dcc.Tab(label='b. Preparación de Datos', children=[
        html.H4('b. Preparación de los Datos'),
        html.Div(className="card border-0 shadow-sm p-4 mb-4", children=[
            html.H5('Fuente de Datos', className="text-primary"),
            html.P('Los datos provienen del Departamento Administrativo Nacional de Estadística (DANE) de Colombia, el archivo que contiene información histórica del IPC desde 1979.'),
            
            html.H5('Proceso de Limpieza y Transformación', className="text-primary mt-4"),
            
            html.H6('1. Carga y Estructuración Inicial', className="fw-bold mt-3"),
            html.Div(className="bg-light p-3 rounded", children=[
                html.Code('df <- read_excel("inflacion.xlsx", sheet = 2, skip = 2)', className="d-block mb-2"),
                html.Small('Se omiten las dos primeras filas (encabezados vacíos) y se carga la hoja 2 del archivo Excel.', className="text-muted")
            ]),
            
            html.H6('2. Formateo de Fechas', className="fw-bold mt-3"),
            html.Div(className="bg-light p-3 rounded", children=[
                html.Code([
                    'colnames(df)[1] <- "Fecha"', html.Br(),
                    'df$Fecha <- as.numeric(df$Fecha)', html.Br(),
                    'df$Fecha <- as.Date(df$Fecha, origin = "1899-12-30")'
                ], className="d-block mb-2"),
                html.Small('Conversión de fechas desde formato Excel serial a formato Date de R, usando el origen correcto para Excel.', className="text-muted")
            ]),
            
            html.H6('3. Limpieza de Nombres de Columnas', className="fw-bold mt-3"),
            html.Div(className="bg-light p-3 rounded", children=[
                html.Code('colnames(df) <- gsub("\\u00A0", "_", colnames(df))', className="d-block mb-2"),
                html.Small('Eliminación de espacios no rompibles (\\u00A0) y reemplazo por guiones bajos para consistencia.', className="text-muted")
            ]),
            
            html.H6('4. Selección de Variables de Interés', className="fw-bold mt-3"),
            html.Div(className="bg-light p-3 rounded", children=[
                html.Code('df <- df %>% select(Fecha, IPC_Barranquilla, IPC_Bogotá, IPC_Bucaramanga, IPC_Cali, IPC_Manizales, IPC_Medellín, IPC_Pasto)', className="d-block mb-2"),
                html.Small('Se seleccionan las columnas de fecha y las principales ciudades para el análisis inicial.', className="text-muted")
            ]),
            
            html.H5('Estructura Final de los Datos', className="text-primary mt-4"),
            html.Div(className="row", children=[
                html.Div(className="col-md-4", children=[
                    html.H6('Dimensiones:', className="fw-bold"),
                    html.P('3787 filas × 4 columnas (Fecha, ciudad, IPC e inflación)')
                ]),
                html.Div(className="col-md-4", children=[
                    html.H6('Período Temporal:', className="fw-bold"),
                    html.P('1980 - 2025 (para incluir la inflación)')
                ]),
                html.Div(className="col-md-4", children=[
                    html.H6('Frecuencia:', className="fw-bold"),
                    html.P('Datos mensuales')
                ])
            ]),
            
            html.H5('Validación de Calidad de Datos', className="text-primary mt-4"),
            html.Ul([
                html.Li([html.Strong('Valores faltantes: '), 'No hubo en las series']),
                html.Li([html.Strong('Consistencia temporal: '), 'Se verificó la continuidad en las fechas mensuales']),
                html.Li([html.Strong('Formato numérico: '), 'Conversión apropiada de todas las variables IPC a formato numérico'])
            ]),
            
            html.Div(className="alert alert-warning mt-4", children=[
                html.Strong('Nota Importante: '),
                'Los datos del IPC requieren tratamiento especial ya que representan índices base 100, donde los cambios porcentuales son más relevantes que los valores absolutos para el análisis inflacionario.'
            ])
        ])
    ]),
    dcc.Tab(label='c. Selección del Modelo', children=[
        html.H4('c. Selección del Modelo o Algoritmo'),
        html.Ul([
            html.Li('Modelo seleccionado: Prophet'),
            html.Li('El modelo Prophet fue seleccionado por su eficacia probada en el pronóstico de series de tiempo, su capacidad para manejar patrones estacionales y tendencias, y su buen desempeño en la predicción del IPC en diversas ciudades.'),
            html.Li('El modelo Prophet descompone una serie de tiempo en cuatro componentes aditivos clave: una tendencia no lineal que captura los cambios a largo plazo (g(t)), efectos estacionales periódicos (s(t)), impactos de eventos o feriados (h(t)), y un término de error aleatorio (ϵ )todo sumado para predecir el valor futuro (y(t)).')
        ])
    ]),
    dcc.Tab(label='d. Evaluación del Modelo', children=[
        html.H4('d. Entrenamiento y Evaluación del Modelo'),
        html.Ul([
            html.Li('Proceso de entrenamiento : Se dividio el dataset en entrenamiento y test 80/20, luego entrenamos nuestro modelos y se utilizo la validación para series temporales.'),
            html.Li('Métricas de evaluación:Los indicadores de rendimiento, como el Error Cuadrático Medio (RMSE) y el Error Porcentual Absoluto Medio (MAPE), son aceptables, lo que valida la capacidad del modelo para generar pronósticos con un nivel de error razonable en el contexto económico.'),
            html.Li('Aunque los detalles específicos varían ligeramente entre ciudades, la consistencia en la calidad de los resultados sugiere que el modelo se adapta bien a las particularidades de los mercados de consumo de cada localidad, proporcionando predicciones fiables a nivel municipal.')
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
        dbc.Row([
            dbc.Col([
                html.H2('Introducción', className="text-center mb-4"),
                
                # Imagen alusiva local
                html.Div(className="text-center mb-4", children=[
                    html.Img(
                        src=app.get_asset_url('inflacion.png'),  # Usar imagen local
                        style={
                            'width': '80%',
                            'max-width': '600px',
                            'border-radius': '10px',
                            'box-shadow': '0 4px 8px rgba(0,0,0,0.1)'
                        }
                    ),
                    html.P("Evolución histórica de la inflación en Colombia", 
                           className="text-muted mt-2")
                ]),
                
                # Texto introductorio
                html.Div(className="card border-0 shadow-sm p-4 mb-4", children=[
                    html.P('Este dashboard analiza la evolución del Índice de Precios al Consumidor (IPC) y la inflación en Colombia desde 1979 hasta la actualidad. El proyecto combina:'),
                    html.Ul([
                        html.Li('Visualización interactiva de datos históricos'),
                        html.Li('Análisis comparativo entre regiones'),
                        html.Li('Modelado predictivo con técnicas avanzadas')
                    ]),
                    
                    html.P('El objetivo principal es proporcionar herramientas para entender la dinámica inflacionaria y apoyar la toma de decisiones económicas.'),
                    
                    html.H5('Principales características:', className="mt-4"),
                    html.Ul([
                        html.Li('Datos desde 1979 hasta proyecciones 2025'),
                        html.Li('Cobertura de ciudades principales de Colombia'),
                        html.Li('Modelo predictivo con evaluación de precisión')
                    ])
                ])
            ], width=10)
        ], justify='center')
    ]),
    

    # 2. Contexto
    dcc.Tab(label='2. Contexto', children=[
        html.H2('Contexto'),
        html.P('Este proyecto se centra en el análisis y predicción del Índice de Precios al Consumidor (IPC) en Colombia, utilizando datos históricos oficiales. El IPC es el indicador estadístico más importante para medir la inflación, representando el cambio promedio en los precios de una canasta de bienes y servicios consumidos por los hogares colombianos.'),
        html.Ul([
            html.Li('El Índice de Precios al Consumidor (IPC) es fundamental para medir la variación promedio de los precios de los bienes y servicios que consume un hogar típico. En Colombia, como en cualquier economía, el IPC es crucial porque refleja la inflación y, por ende, el poder adquisitivo de la moneda. Su seguimiento es vital para la formulación de políticas monetarias, la negociación salarial, el análisis de la capacidad de compra de los ciudadanos y la toma de decisiones tanto a nivel gubernamental como empresarial.'),
            html.Li('Fuente de los datos: Portal del DANE (Departamento Administrativo Nacional de Estadística)'),
            html.Li('Variables de interés: IPC (Indices del precio del consumidor) y se tuvieron en cuenta los datos historicos y las ciudades principales del Colombia')
            
        ])
    ]),

# 3. Planteamiento del Problema
    dcc.Tab(label='3. Planteamiento del Problema', children=[
        html.H2('Planteamiento del Problema'),
        html.Div(className="card border-0 shadow-sm p-4 mb-4", children=[
            html.P('La inflación es uno de los indicadores económicos más críticos, afectando directamente el poder adquisitivo de la población, las decisiones de política monetaria y la planeación estratégica de empresas y gobiernos. En Colombia, históricamente ha presentado alta volatilidad, con períodos de hiperinflación en los años 90 y desafíos recientes post-pandemia.'),
            
            html.H4('Problema Central', className="mt-4 text-primary"),
            html.P('La dificultad para predecir con precisión el comportamiento futuro del IPC limita la capacidad de:'),
            html.Ul([
                html.Li('Los formuladores de política para diseñar medidas efectivas'),
                html.Li('Las empresas para planificar sus estrategias de precios y costos'),
                html.Li('Los ciudadanos para tomar decisiones financieras informadas')
            ]),
            
            html.H4('Pregunta de Investigación', className="mt-4 text-primary"),
            html.Div(className="alert alert-secondary", children=[
                html.H5('¿Cómo podemos desarrollar un modelo predictivo preciso del IPC para las principales ciudades colombianas que capture tanto tendencias históricas como patrones estacionales, utilizando técnicas avanzadas de series de tiempo?')
            ]),
            
            html.H4('Alcance del Proyecto', className="mt-4 text-primary"),
            html.Ul([
                html.Li('Análisis histórico desde 1979 hasta la actualidad'),
                html.Li('Predicciones a 12 meses para 7 ciudades'),
                html.Li('Implementación del modelo Prophet desarrollado por Meta (Facebook)'),
                html.Li('Dashboard interactivo para visualización de resultados')
            ])
        ])
    ]),
    dcc.Tab(label='4. Objetivos y Justificación', children=[
        html.H2('Objetivos'),
        html.Ul([html.Li('Objetivo General:Predecir el IPC en Colombia para fechas futuras y poder tomar mejores desiciones en el ambito económico.')]),
        html.H4('Objetivos Específicos'),
        html.Ul([
            html.Li('Detectar curioridades como por ejemplo un cambio drastico en el valor del IPC.'),
            html.Li('Visualizar de forma detallada el incremento del IPC por ciudades.'),
            html.Li('Objetivo general del proyecto:Implementar un buen modelo para obtener buenos resultados y detectar mejoras para implementar otros modelos a futuro.'),
        ]),
        html.H4('Justificación'),
        html.P('Entender patrones regionales, anticipar escenarios inflacionarios y poder implementar estas herramientas para una buena toma de decisiones económicas.')
    ]),
    dcc.Tab(label='5. Marco Teórico', children=[
        html.H2('Marco Teórico'),
        html.Div(className="card border-0 shadow-sm p-4 mb-4", children=[
            
            html.H4('5.1 Fundamentos del Índice de Precios al Consumidor (IPC)', className="text-primary"),
            html.P('El IPC constituye el indicador estadístico más importante para medir la inflación, representando el cambio promedio en los precios de una canasta de bienes y servicios consumidos por los hogares. En Colombia, el DANE es responsable de su cálculo y publicación, utilizando una metodología que refleja los patrones de consumo de los hogares colombianos.'),
            
            html.H4('5.2 Teorías Económicas de la Inflación', className="text-primary mt-4"),
            html.Div(className="row", children=[
                html.Div(className="col-md-4", children=[
                    html.H6('Teoría Monetarista', className="fw-bold"),
                    html.P('La inflación surge cuando el incremento en la masa monetaria excede la demanda de dinero (Friedman).')
                ]),
                html.Div(className="col-md-4", children=[
                    html.H6('Inflación por Demanda', className="fw-bold"),
                    html.P('Ocurre cuando la demanda agregada supera la oferta agregada de bienes y servicios.')
                ]),
                html.Div(className="col-md-4", children=[
                    html.H6('Inflación por Costos', className="fw-bold"),
                    html.P('Se origina por aumentos en costos de producción (materias primas, salarios, energía).')
                ])
            ]),
            
            html.H4('5.3 Modelos de Predicción de Series Temporales', className="text-primary mt-4"),
            html.P('Los modelos econométricos tradicionales como ARIMA y GARCH han sido ampliamente utilizados para predicción económica. Sin embargo, el avance en inteligencia artificial ha permitido el desarrollo de modelos más sofisticados como Prophet.'),
            
            html.H4('5.4 El Modelo Prophet', className="text-primary mt-4"),
            html.Div(className="alert alert-info", children=[
                html.P('Prophet es un modelo desarrollado por Meta (Facebook) diseñado específicamente para series temporales con fuertes patrones estacionales. Sus principales ventajas incluyen:', className="mb-2"),
                html.Ul([
                    html.Li('Robustez ante datos faltantes'),
                    html.Li('Detección automática de cambios de tendencia'),
                    html.Li('Modelado de múltiples estacionalidades'),
                    html.Li('Incorporación de efectos de días especiales'),
                    html.Li('Intervalos de incertidumbre en las predicciones')
                ])
            ]),
            
            html.H4('5.5 Contexto de la Inflación en Colombia', className="text-primary mt-4"),
            html.Div(className="row", children=[
                html.Div(className="col-md-6", children=[
                    html.H6('Comportamiento Histórico', className="fw-bold"),
                    html.P('Colombia experimentó períodos de hiperinflación en los años 90 (>30% anual). La implementación del esquema de inflación objetivo por el Banco de la República ha contribuido a mayor estabilidad desde los 2000s.')
                ]),
                html.Div(className="col-md-6", children=[
                    html.H6('Desafíos Post-Pandemia', className="fw-bold"),
                    html.P('La pandemia introdujo nuevas dinámicas inflacionarias. La corrección en precios de alimentos, especialmente perecederos en 2023, jugó un papel significativo en el descenso gradual de la inflación.')
                ])
            ]),
            
            html.H4('5.6 Heterogeneidad Regional', className="text-primary mt-4"),
            html.P('Las diferencias regionales en Colombia presentan desafíos únicos. Factores como geografía, estructura productiva local, costos de transporte y dinámicas de mercado específicas podrían generar comportamientos diferenciados en la inflación por ciudades.'),
            
            html.H4('5.7 Aplicaciones en Política Económica', className="text-primary mt-4"),
            html.Div(className="row", children=[
                html.Div(className="col-md-4", children=[
                    html.H6('Política Monetaria', className="fw-bold"),
                    html.P('Las predicciones del IPC son fundamentales para decisiones del Banco de la República.')
                ]),
                html.Div(className="col-md-4", children=[
                    html.H6('Planificación Empresarial', className="fw-bold"),
                    html.P('Empresas utilizan proyecciones para estrategias de precios y contratos.')
                ]),
                html.Div(className="col-md-4", children=[
                    html.H6('Decisiones de Inversión', className="fw-bold"),
                    html.P('Hogares e inversionistas requieren estimaciones para optimizar decisiones financieras.')
                ])
            ]),
            
            html.Hr(),
            html.Small([
                html.Strong('Fuentes principales: '),
                'DANE (2024), Amazon Web Services - Algoritmo Prophet, Banco de la República, estudios previos sobre predicción del IPC en Colombia.'
            ], className="text-muted")
        ])
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
