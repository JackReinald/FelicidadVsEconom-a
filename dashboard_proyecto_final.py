import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# =========================================
# 1. Configuración inicial con tema oscuro completo
# =========================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Proyecto final - Dashboard de Felicidad"
server = app.server

# =================================================================================
# 2. Datos
# =================================================================================

dfFelicidad = pd.read_csv("WHI_Inflation.csv", sep="\t")
print(dfFelicidad.head())

dfCleaned = dfFelicidad.drop_duplicates().dropna()

# Extracción de los continentes
continentsGroup = dfFelicidad.groupby("Continent")
indicePorContinent = continentsGroup["Rank"].mean()
categoriesFelicidad = list(indicePorContinent.index)

# Datos iniciales para las cards
initial_year = 2016  # Año inicial por defecto
happiness_initial = dfFelicidad[dfFelicidad["Year"] == initial_year]["Score"].mean()

summary_data_initial = {
    "Total de paises incluidos": f"{dfFelicidad["Country"].drop_duplicates().dropna().count():,.0f}",
    "Felicidad mundial promedio": f"{happiness_initial:.2f}" if pd.notna(happiness_initial) else "Año no registrado.",
}

# =================================================================================
# 3. Configuración de tema oscuro para los gráficos
# =================================================================================
dark_template = {
    "layout": {
        "paper_bgcolor": "#222",
        "plot_bgcolor": "#222",
        "font": {"color": "#EEE"},
        "xaxis": {"gridcolor": "#444", "linecolor": "#666", "zerolinecolor": "#444"},
        "yaxis": {"gridcolor": "#444", "linecolor": "#666", "zerolinecolor": "#444"},
        "hoverlabel": {"bgcolor": "#333", "font": {"color": "white"}},
        "colorway": ["#00bc8c", "#3498db", "#f39c12", "#e74c3c"],
    }
}

# =================================================================================
# 4. Creación de componentes visuales
# =================================================================================
def create_dark_card(title, value, icon_name, color, id):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.I(className=f"fas fa-{icon_name} fa-2x me-2"),
                        html.H5(title, className="card-title mb-0"),
                    ],
                    className="d-flex align-items-center",
                ),
                html.H2(value, className="mt-2 mb-0", id=id),
            ]
        ),
        className=f"border-0 mb-3 h-100 bg-{color}",  # Usamos las clases de color originales
        style={
            "border-left": f"4px solid {get_border_color(color)}"
        },  # Borde lateral para contraste
    )


# Paleta de colores
def get_border_color(color):
    return {
        "primary": "#3498db",
        "success": "#00bc8c",
        "info": "#17a2b8",
        "warning": "#f39c12",
        "danger": "#e74c3c",
    }.get(color, "#555")


def create_light_dropdown(id, label, options, value, multi=False):
    return html.Div(
        [
            dbc.Label(
                label, className="mb-2 fw-bold text-light"
            ),  # Label claro sobre fondo oscuro
            dcc.Dropdown(
                id=id,
                options=[{"label": opt, "value": opt} for opt in options],
                value=value,
                multi=multi,
                clearable=False,
                className="light-dropdown",
                style={
                    "backgroundColor": "white",  # Fondo blanco
                    "color": "#333",  # Texto oscuro
                    "borderRadius": "6px",
                },
            ),
        ],
        className="mb-4",
    )


# Gráficos de tendencia interactivos
top5PaisesMasFelices = pd.read_csv("top5_paises_felices.csv")
def create_trend_chart():
    fig = px.line(
        top5PaisesMasFelices,
        x="Year",
        y="Score",
        color="Country",
        template=dark_template,
    )
    fig.update_layout(
        title={          
            "text": "Top 5 paises más felices por año",
            "yanchor": "top",
        },
        hovermode="x unified",
        legend=dict(orientation="v", yanchor="bottom", y=1.02, xanchor="right", x=1, ),
    )
    return fig

PIBPaisesMasFelices = pd.read_csv("PIB_paises_mas_felices.csv")
def create_PIB_chart():
    fig = px.line(
        PIBPaisesMasFelices,
        x="Year",
        y="GDP per Capita",
        color="Country",
        template=dark_template,
    )
    fig.update_layout(
        title="Tendencia del PIB por el top 5 paises más felices",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def create_happiness_per_continent_chart(continent="South America"):
    # Eliminación de filas duplicadas
    dfLimpio = dfFelicidad.drop_duplicates()

    # Reemplazo de filas con NaN por ceros
    dfLimpio = dfLimpio.dropna()
    dfLimpio
    # Data enfocada en Colombia
    filtroPorContinente = dfLimpio["Continent"] == continent
    paisesDelContinente = dfLimpio[filtroPorContinente]
    #Crea el grafico
    grafico = px.line(paisesDelContinente, x="Year", y="Score", color="Country")

    grafico.update_layout(title="Felicidad de " + continent,
                          template=dark_template)
    return grafico


def create_multiple_bar_charts_plotly_with_titles_spacing(highlight_continent=None):
    dfLimpio = dfFelicidad.drop_duplicates().dropna()
    agrupacionPorContinente = dfLimpio.groupby("Continent")

    titulos = [
        "Healthy life expectancy at birth",
        "Freedom to make life choices",
        "Perceptions of corruption",
        "Food Consumer Price Inflation",
        "GDP deflator Index growth rate"
    ]

    num_graficos = len(titulos)
    num_filas = (num_graficos + 1) // 2
    num_columnas = 2

    fig = make_subplots(rows=num_filas, cols=num_columnas, subplot_titles=titulos)

    for i, titulo in enumerate(titulos):
        analisis = agrupacionPorContinente[titulo].mean().sort_values(ascending=False).reset_index()
        
        if analisis[titulo].mean() >= 0 and analisis[titulo].mean() <= 1:
            analisis[titulo] = analisis[titulo] * 100

        # Colorear la barra del continente seleccionado
        colors = ['#f39c12' if cont == highlight_continent else '#3498db' for cont in analisis["Continent"]]

        fila = (i // num_columnas) + 1
        columna = (i % num_columnas) + 1

        fig.add_trace(
            go.Bar(
                x=analisis["Continent"],
                y=analisis[titulo],
                name=titulo,
                showlegend=False,
                marker_color=colors,
            ),
            row=fila,
            col=columna
        )

        fig.update_yaxes(title_text="Percentage (%)", row=fila, col=columna)
        fig.update_xaxes(title_text="Continent", row=fila, col=columna)

    fig.update_layout(
        title={
            'text': "Análisis de Indicadores de Felicidad por Continente",
            'x': 0.5,
            'xanchor': 'center'
        },
        margin=dict(t=50),
        height=300 * num_filas,
        template=dark_template
    )

    return fig


# Matriz de correlación:
def create_correlation_matrix():
    # Eliminación de filas duplicadas
    dfLimpio = dfFelicidad.drop_duplicates()

    # Reemplazo de filas con NaN por ceros
    dfLimpio = dfLimpio.dropna()

    # Averiguar la categoría que más afecta a Score mediante matriz de confusión
    correlationMatrix = dfLimpio.select_dtypes(include="number").corr() # Selecciona solo las columnas numéricas y saca correlación

    # Obtener las etiquetas para el eje x e y
    labels = correlationMatrix.columns.tolist()

    # Crear la figura con Plotly
    fig = go.Figure(data=go.Heatmap(
        z=correlationMatrix.values,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0, # Centrar el colorscale en 0 para la correlación
        colorbar=dict(title='Correlación'),
    ))

    fig.update_layout(
        title='Matriz de Correlación',
        template=dark_template,
        xaxis_side='bottom',
        xaxis=dict(
            side='top',
            tickangle=-45, # Ángulo para las etiquetas del eje x (diagonal)
            tickfont=dict(size=13) # Opcional: ajustar el tamaño de la fuente
        ),
        height=800,
        width=1500,
        annotations=[ go.layout.Annotation(
            x=col,
            y=row,
            text=f'{correlationMatrix.iloc[row, col]:.2f}',
            showarrow=False,
            font=dict(size=11.5),
            font_color='black' if -0.3 < correlationMatrix.iloc[row, col] < 0.3 else 'white' # Texto negro para valores cercanos a 0, blanco para el resto
        ) for row in range(correlationMatrix.shape[0]) for col in range(correlationMatrix.shape[1]) ]
    )
    return fig


# =================================================================================
# 5. Layout del dashboard
# =================================================================================
app.layout = dbc.Container(
    fluid=True,
    className="dark-theme",
    children=[
        # Header con logo y título
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.I(className="fas fa-chart-line fa-2x me-2"),
                                html.H1("Dashboard de Felicidad", className="mb-0"),
                            ],
                            className="d-flex align-items-center py-3",
                        )
                    ],
                    width=12,
                )
            ],
            className="border-bottom  mb-4",
        ),
        # Filtros y Controles
        dbc.Row(
            [
                dbc.Col(
                    create_light_dropdown(
                        "year-selector",
                        "Seleccione un año",
                        dfFelicidad["Year"].drop_duplicates(),
                        2016 # Valor inicial
                    ),
                    md=2
                ),
                dbc.Col( # Nuevo Col para el dropdown de continente
                    create_light_dropdown(
                        "continent-selector",
                        "Seleccione un continente",
                        dfCleaned["Continent"].drop_duplicates(),
                        "South America" # Valor inicial
                    ),
                    md=4
                ),
            ]
        ),

        # Tarjetas
        dbc.Row(
            [
                dbc.Col(
                    create_dark_card(
                        "Total de paises incluidos",
                        summary_data_initial["Total de paises incluidos"],
                        "earth-europe",
                        "success",
                        id="total-paises"
                    ),
                    md=3,
                ),
                dbc.Col(
                    create_dark_card(
                        "Felicidad mundial promedio",
                        summary_data_initial["Felicidad mundial promedio"],
                        "chart-line",
                        "info",
                        id="felicidad-promedio", # Añadimos un ID a esta tarjeta
                    ),
                    md=3,
                ),
                dbc.Col(
                    create_dark_card(
                        "Total de registros",
                        len(dfFelicidad),
                        "file   ",
                        "primary",
                        id="total-records"
                    ),
                    md=4
                ),
            ],
            className="mb-4 g-3",
        ),
        # Primera fila de gráficos
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="trend-chart",
                        figure=create_trend_chart(),
                        className="dark-graph",
                    ),
                    lg=6,
                    className="mb-3",
                ),
                dbc.Col(
                    dcc.Graph(
                        id="PIB-chart",
                        figure=create_PIB_chart(),
                        className="dark-graph",
                    ),
                    lg=6,
                    className="mb-3",
                ),
            ],
            className="g-3",
        ),
        # Segunda fila de graficos
        dbc.Row(
            dbc.Col(
                html.Div(
                    dcc.Graph(
                        id="continental-happiness",
                        figure=create_happiness_per_continent_chart(),
                        className="dark-graph",
                    ),
                    className="my-2",
                    style={"width": "64%"} # Ancho del grafico
                ),
                className="mb-3",
                style={
                    "display": "flex",
                    "justify-content": "center",
                }     
            ),
        ),
        # Tercera fila de gráficos
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="correlation-matrix",
                        figure=create_correlation_matrix(),
                        className="dark-graph",
                    ),
                    className="mb-4",
                    style={
                        "display": "flex",
                        "justify-content": "center"
                    }
                ),
            ]
        ),
        # Cuarta fila dedicada a un subtitulo
        dbc.Row(
            dbc.Col(
                [
                    html.H2(
                        "Indicadores clave de felicidad por continente",
                        className="my-5 text-center"
                    )
                ]
            ),
        ),
        # Quinta fila de graficos
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="happiness-key-factors",
                        figure=create_multiple_bar_charts_plotly_with_titles_spacing(),
                        className="dark-graph"

                    )
                ),
            ]
        ),

        # Footer
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.P(
                                [
                                    html.Span(
                                        "© 2025 Nombre estudiantes", className="me-2"
                                    ),
                                    html.A(
                                        html.I(className="fab fa-github me-2"),
                                        href="#asdf",
                                    ),
                                    html.A(
                                        html.I(className="fab fa-linkedin me-2"),
                                        href="#",
                                    ),
                                    html.A(
                                        html.I(className="fa-brands fa-instagram"),
                                        href="#asdf",
                                    ),
                                ],
                                className="text-muted text-center py-3 mb-0",
                            )
                        ]
                    ),
                    width=12,
                )
            ],
            className="mt-4 border-top border-dark",
        ),
    ],
    style={
        "backgroundColor": "#1a1a1a",
        "color": "#EEE",
        "minHeight": "100vh",
        "padding": "20px",
    },
)


# =========================================
# 6. Callbacks para interactividad
# =========================================
@callback(
    Output("felicidad-promedio", "children"),
    Input("year-selector", "value"),
)
def update_felicidad_promedio(selected_year):
    happiness_year = dfFelicidad[dfFelicidad["Year"] == selected_year]["Score"].mean()
    if pd.notna(happiness_year):
        return f"{happiness_year:.3f}"
    else:
        return "Año no registrado."
    
@callback(
    Output("continental-happiness", "figure"),
    Input("continent-selector", "value")
)
def update_continental_happiness_chart(selected_continent):
    return create_happiness_per_continent_chart(selected_continent)

@app.callback(
    Output("happiness-key-factors", "figure"),
    Input("continent-selector", "value")
)
def update_multiple_bar_charts(continent):
    return create_multiple_bar_charts_plotly_with_titles_spacing(highlight_continent=continent)


# =========================================
# 7. Estilos CSS personalizados
# =========================================
app.css.append_css(
    {
        "external_url": [
            {
                "selector": ".light-dropdown .Select-menu-outer",
                "rule": """
                    background-color: white !important;
                    color: #333 !important;
                    border: 1px solid #ddd !important;
                """,
            },
            {
                "selector": ".light-dropdown .Select-control",
                "rule": "border: 1px solid #ddd !important;",
            },
            {
                "selector": ".light-dropdown .Select-value-label",
                "rule": "color: #333 !important;",
            },
        ]
    }
)

# =========================================
# 8. Ejecutar la aplicación
# =========================================
if __name__ == "__main__":
    app.run(debug=True)