import pandas as pd
from sqlalchemy import create_engine

df = pd.read_excel("inflacion_procesada.xlsx")

usuario = "dataviz_db_a2qu_user"
contraseña = "aZ1oCNwaqUh2SruEvnFrNSO1Tp2Hk6ti"  
host = "dpg-d0oe98muk2gs73bp2g3g-a.oregon-postgres.render.com"
puerto = "5432"
base_datos = "dataviz_db_a2qu"

engine = create_engine(f"postgresql+psycopg2://{usuario}:{contraseña}@{host}:{puerto}/{base_datos}")

df.to_sql("ipc_excel", engine, if_exists="replace", index=False)
print("\n Datos insertados correctamente en la tabla 'ipc_excel'.")
