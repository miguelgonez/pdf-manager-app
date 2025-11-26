import streamlit as st
import sqlite3
import PyPDF2
from datetime import datetime
from openai import OpenAI
import io
import os

# Configuración de la página
st.set_page_config(
    page_title="Gestor de PDFs con Referencias",
    page_icon="📚",
    layout="wide"
)

# Sistema de autenticación simple con Google
def check_authentication():
    """Verifica si el usuario está autenticado"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔐 Acceso Restringido")
        st.markdown("---")
        st.info("Esta aplicación requiere autenticación. Por favor, ingresa tu email autorizado.")
        
        # Obtener emails autorizados desde secrets
        authorized_emails = []
        if "AUTHORIZED_EMAILS" in st.secrets:
            authorized_emails = [email.strip() for email in st.secrets["AUTHORIZED_EMAILS"].split(",")]
        
        email = st.text_input("Email", key="login_email")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Ingresar", type="primary"):
                if email in authorized_emails:
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.rerun()
                else:
                    st.error("❌ Email no autorizado. Contacta al administrador.")
        
        st.markdown("---")
        st.caption("💡 Solo usuarios autorizados pueden acceder a esta aplicación.")
        st.stop()

# Verificar autenticación antes de mostrar la app
check_authentication()

# Mostrar usuario autenticado en el sidebar
st.sidebar.success(f"✅ Autenticado como: {st.session_state.user_email}")
if st.sidebar.button("🚪 Cerrar Sesión"):
    st.session_state.authenticated = False
    st.rerun()

st.sidebar.markdown("---")

# Inicializar cliente OpenAI
client = OpenAI()

# Función para inicializar la base de datos
def init_db():
    conn = sqlite3.connect('documentos.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS documentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            referencia TEXT NOT NULL UNIQUE,
            titulo TEXT NOT NULL,
            anio INTEGER NOT NULL,
            resumen TEXT NOT NULL,
            texto_completo TEXT,
            fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Función para extraer texto del PDF
def extraer_texto_pdf(archivo_pdf):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(archivo_pdf.read()))
        texto = ""
        for pagina in pdf_reader.pages:
            texto += pagina.extract_text()
        return texto
    except Exception as e:
        st.error(f"Error al extraer texto del PDF: {str(e)}")
        return None

# Función para generar resumen usando IA
def generar_resumen(texto, max_palabras=300):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Eres un asistente especializado en crear resúmenes concisos y precisos de documentos académicos y legales."},
                {"role": "user", "content": f"Por favor, crea un resumen de aproximadamente {max_palabras} palabras del siguiente texto:\n\n{texto[:8000]}"}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al generar resumen: {str(e)}")
        return None

# Función para obtener el siguiente número de artículo
def obtener_siguiente_numero(anio):
    conn = sqlite3.connect('documentos.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM documentos WHERE anio = ?', (anio,))
    count = c.fetchone()[0]
    conn.close()
    return count + 1

# Función para insertar documento en la base de datos
def insertar_documento(referencia, titulo, anio, resumen, texto_completo):
    try:
        conn = sqlite3.connect('documentos.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO documentos (referencia, titulo, anio, resumen, texto_completo)
            VALUES (?, ?, ?, ?, ?)
        ''', (referencia, titulo, anio, resumen, texto_completo))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        st.error("Esta referencia ya existe en la base de datos.")
        return False
    except Exception as e:
        st.error(f"Error al insertar documento: {str(e)}")
        return False

# Función para obtener todos los documentos
def obtener_documentos():
    conn = sqlite3.connect('documentos.db')
    c = conn.cursor()
    c.execute('SELECT id, referencia, titulo, anio, resumen, fecha_registro FROM documentos ORDER BY id DESC')
    documentos = c.fetchall()
    conn.close()
    return documentos

# Función para obtener un documento por ID
def obtener_documento_por_id(doc_id):
    conn = sqlite3.connect('documentos.db')
    c = conn.cursor()
    c.execute('SELECT * FROM documentos WHERE id = ?', (doc_id,))
    documento = c.fetchone()
    conn.close()
    return documento

# Función para eliminar un documento
def eliminar_documento(doc_id):
    conn = sqlite3.connect('documentos.db')
    c = conn.cursor()
    c.execute('DELETE FROM documentos WHERE id = ?', (doc_id,))
    conn.commit()
    conn.close()

# Inicializar la base de datos
init_db()

# Título principal
st.title("📚 Gestor de Documentos PDF")
st.markdown("---")

# Menú de navegación
menu = st.sidebar.selectbox(
    "Navegación",
    ["📤 Subir Documento", "📋 Ver Documentos", "🔍 Buscar Documento"]
)

# SECCIÓN: Subir Documento
if menu == "📤 Subir Documento":
    st.header("Subir Nuevo Documento PDF")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        archivo_pdf = st.file_uploader("Selecciona un archivo PDF", type=['pdf'])
    
    with col2:
        anio = st.number_input("Año del documento", min_value=1900, max_value=2100, value=datetime.now().year)
    
    titulo = st.text_input("Título del documento (opcional)")
    
    if st.button("Procesar Documento", type="primary"):
        if archivo_pdf is not None:
            with st.spinner("Procesando documento..."):
                # Extraer texto
                st.info("Extrayendo texto del PDF...")
                texto = extraer_texto_pdf(archivo_pdf)
                
                if texto:
                    # Generar título si no se proporcionó
                    if not titulo:
                        titulo = archivo_pdf.name.replace('.pdf', '')
                    
                    # Generar referencia
                    numero = obtener_siguiente_numero(anio)
                    referencia = f"Art. {anio}{numero:04d}"
                    
                    # Generar resumen
                    st.info("Generando resumen con IA...")
                    resumen = generar_resumen(texto, 300)
                    
                    if resumen:
                        # Insertar en la base de datos
                        if insertar_documento(referencia, titulo, anio, resumen, texto):
                            st.success(f"✅ Documento procesado exitosamente!")
                            st.success(f"**Referencia asignada:** {referencia}")
                            
                            # Mostrar vista previa
                            st.subheader("Vista Previa")
                            st.write(f"**Título:** {titulo}")
                            st.write(f"**Año:** {anio}")
                            st.write(f"**Referencia:** {referencia}")
                            st.write(f"**Resumen:**")
                            st.write(resumen)
                        else:
                            st.error("No se pudo guardar el documento en la base de datos.")
                else:
                    st.error("No se pudo extraer texto del PDF.")
        else:
            st.warning("Por favor, selecciona un archivo PDF.")

# SECCIÓN: Ver Documentos
elif menu == "📋 Ver Documentos":
    st.header("Documentos Registrados")
    
    documentos = obtener_documentos()
    
    if documentos:
        st.write(f"**Total de documentos:** {len(documentos)}")
        st.markdown("---")
        
        for doc in documentos:
            doc_id, referencia, titulo, anio, resumen, fecha_registro = doc
            
            with st.expander(f"**{referencia}** - {titulo}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Título:** {titulo}")
                    st.write(f"**Año:** {anio}")
                    st.write(f"**Fecha de registro:** {fecha_registro}")
                    st.write(f"**Resumen:**")
                    st.write(resumen)
                
                with col2:
                    if st.button(f"Ver completo", key=f"ver_{doc_id}"):
                        doc_completo = obtener_documento_por_id(doc_id)
                        if doc_completo:
                            st.session_state[f'mostrar_completo_{doc_id}'] = True
                    
                    if st.button(f"🗑️ Eliminar", key=f"del_{doc_id}"):
                        eliminar_documento(doc_id)
                        st.success("Documento eliminado")
                        st.rerun()
                
                # Mostrar texto completo si se solicitó
                if st.session_state.get(f'mostrar_completo_{doc_id}', False):
                    doc_completo = obtener_documento_por_id(doc_id)
                    if doc_completo and doc_completo[5]:
                        st.markdown("**Texto Completo:**")
                        st.text_area("", doc_completo[5], height=300, key=f"texto_{doc_id}")
    else:
        st.info("No hay documentos registrados. Sube tu primer documento en la sección 'Subir Documento'.")

# SECCIÓN: Buscar Documento
elif menu == "🔍 Buscar Documento":
    st.header("Buscar Documento")
    
    tipo_busqueda = st.radio("Buscar por:", ["Referencia", "Título", "Año"])
    
    if tipo_busqueda == "Referencia":
        busqueda = st.text_input("Ingresa la referencia (ej: Art. 20240001)")
        if busqueda:
            conn = sqlite3.connect('documentos.db')
            c = conn.cursor()
            c.execute('SELECT * FROM documentos WHERE referencia LIKE ?', (f'%{busqueda}%',))
            resultados = c.fetchall()
            conn.close()
    
    elif tipo_busqueda == "Título":
        busqueda = st.text_input("Ingresa el título o parte de él")
        if busqueda:
            conn = sqlite3.connect('documentos.db')
            c = conn.cursor()
            c.execute('SELECT * FROM documentos WHERE titulo LIKE ?', (f'%{busqueda}%',))
            resultados = c.fetchall()
            conn.close()
    
    else:  # Año
        busqueda = st.number_input("Ingresa el año", min_value=1900, max_value=2100, value=datetime.now().year)
        if busqueda:
            conn = sqlite3.connect('documentos.db')
            c = conn.cursor()
            c.execute('SELECT * FROM documentos WHERE anio = ?', (busqueda,))
            resultados = c.fetchall()
            conn.close()
    
    if 'resultados' in locals() and resultados:
        st.success(f"Se encontraron {len(resultados)} resultado(s)")
        st.markdown("---")
        
        for doc in resultados:
            doc_id, referencia, titulo, anio, resumen, texto_completo, fecha_registro = doc
            
            with st.expander(f"**{referencia}** - {titulo}", expanded=True):
                st.write(f"**Título:** {titulo}")
                st.write(f"**Año:** {anio}")
                st.write(f"**Fecha de registro:** {fecha_registro}")
                st.write(f"**Resumen:**")
                st.write(resumen)
                
                if st.button(f"Ver texto completo", key=f"buscar_{doc_id}"):
                    if texto_completo:
                        st.text_area("Texto Completo:", texto_completo, height=300)
    elif 'resultados' in locals():
        st.warning("No se encontraron resultados.")

# Pie de página
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Los resúmenes se generan automáticamente usando IA.")
