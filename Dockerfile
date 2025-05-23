FROM python:3.10-slim

# Establecer modo no interactivo
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema necesarias para compilar Prophet y otras librerías
RUN apt-get update && apt-get install -y \
    build-essential \
    libpython3-dev \
    libatlas-base-dev \
    gfortran \
    libfreetype6-dev \
    libpng-dev \
    libopenblas-dev \
    git \
    curl \
    && apt-get clean

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY . .

# Instalar pip actualizado y dependencias de Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto de la app Dash
EXPOSE 8050

# Comando por defecto
CMD ["python", "dashboard.py"]
