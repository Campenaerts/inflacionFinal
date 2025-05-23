import pandas as pd
from sqlalchemy import create_engine

df = pd.read_excel("inflacion_procesada.xlsx")

usuario = "postgres"
contraseña = "luukde1ong"  
host = "localhost"
puerto = "5432"
base_datos = "dataviz_db"

engine = create_engine(f"postgresql+psycopg2://{usuario}:{contraseña}@{host}:{puerto}/{base_datos}")

df.to_sql("ipc_excel", engine, if_exists="replace", index=False)
print("\n Datos insertados correctamente en la tabla 'ipc_excel'.")
