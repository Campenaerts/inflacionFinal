import pandas as pd
from sqlalchemy import create_engine

usuario = "postgres"
contraseña = "luukde1ong"
host = "localhost"
puerto = "5432"
base_datos = "dataviz_db"

engine = create_engine(f"postgresql+psycopg2://{usuario}:{contraseña}@{host}:{puerto}/{base_datos}")

# Consultas

# 1. Total de registros
query_total = "SELECT COUNT(*) AS total_registros FROM ipc_excel;"
df_total = pd.read_sql(query_total, engine)
print(f"\n Total de registros en la tabla: {df_total.iloc[0]['total_registros']}")