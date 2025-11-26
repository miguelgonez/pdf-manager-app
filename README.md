# Gestor de Documentos PDF con Referencias

Aplicación Streamlit para gestionar documentos PDF con generación automática de referencias, resúmenes con IA y almacenamiento en base de datos SQLite.

## Características

- **Subida de PDFs**: Carga archivos PDF y extrae automáticamente el texto
- **Referencias automáticas**: Genera referencias únicas con formato `Art. XXXX####` (año + número secuencial)
- **Resúmenes con IA**: Crea resúmenes de aproximadamente 300 palabras usando GPT-4.1-mini
- **Base de datos SQLite**: Almacena todos los documentos con sus metadatos
- **Navegación completa**: Visualiza, busca y gestiona todos tus documentos

## Instalación

1. Instala las dependencias:
```bash
pip3 install -r requirements.txt
```

2. Asegúrate de tener configurada la variable de entorno `OPENAI_API_KEY` (ya está configurada en este entorno)

## Uso

1. Ejecuta la aplicación:
```bash
streamlit run app.py
```

2. La aplicación se abrirá en tu navegador (por defecto en http://localhost:8501)

## Funcionalidades

### 📤 Subir Documento
- Selecciona un archivo PDF
- Especifica el año del documento
- Opcionalmente, proporciona un título (si no, se usa el nombre del archivo)
- Haz clic en "Procesar Documento"
- El sistema:
  - Extrae el texto del PDF
  - Genera una referencia única (ej: Art. 20250001)
  - Crea un resumen de 300 palabras con IA
  - Guarda todo en la base de datos SQLite

### 📋 Ver Documentos
- Visualiza todos los documentos registrados
- Expande cada documento para ver:
  - Título, año y fecha de registro
  - Resumen completo
  - Opción para ver el texto completo extraído
- Elimina documentos si es necesario

### 🔍 Buscar Documento
- Busca por:
  - **Referencia**: Encuentra documentos por su código único
  - **Título**: Busca por palabras clave en el título
  - **Año**: Filtra documentos por año
- Visualiza los resultados con toda la información

## Estructura de la Base de Datos

La tabla `documentos` contiene:
- `id`: Identificador único autoincremental
- `referencia`: Código único (Art. XXXX####)
- `titulo`: Título del documento
- `anio`: Año del documento
- `resumen`: Resumen generado por IA (aprox. 300 palabras)
- `texto_completo`: Texto completo extraído del PDF
- `fecha_registro`: Timestamp de cuando se registró el documento

## Formato de Referencia

Las referencias siguen el formato: **Art. XXXX####**
- `XXXX`: Año del documento (4 dígitos)
- `####`: Número secuencial de 4 dígitos (0001, 0002, etc.)

Ejemplo: `Art. 20250001` (primer documento del año 2025)

## Notas Técnicas

- La base de datos SQLite se crea automáticamente en `documentos.db`
- Los resúmenes se generan usando el modelo GPT-4.1-mini de OpenAI
- El texto completo del PDF se almacena para futuras referencias
- La aplicación es completamente funcional y operativa, no es una página estática
