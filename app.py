import streamlit as st
import sqlite3
import PyPDF2
from datetime import datetime
from openai import OpenAI
import io
import os
import json
import re

# Imports para RAG
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

# Configuración de la página
st.set_page_config(
    page_title="Gestor de PDFs con Referencias",
    page_icon="📚",
    layout="wide"
)

# Sistema de autenticación simple
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

# Inicializar cliente de API
client_openai = OpenAI()

# Inicializar ChromaDB para RAG
@st.cache_resource
def init_chromadb():
    """Inicializa ChromaDB para almacenamiento vectorial"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="documentos_pdf",
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection

# Inicializar embeddings de OpenAI
@st.cache_resource
def init_embeddings():
    """Inicializa el modelo de embeddings de OpenAI"""
    return OpenAIEmbeddings(model="text-embedding-3-small")

# Inicializar LLM para RAG
@st.cache_resource
def init_llm():
    """Inicializa el modelo de lenguaje para RAG"""
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)

chroma_client, chroma_collection = init_chromadb()
embeddings_model = init_embeddings()
llm_rag = init_llm()

# Función para añadir documento a ChromaDB
def add_documento_to_vectordb(referencia, titulo, anio, texto_completo, resumen):
    """Añade un documento a ChromaDB evitando duplicados"""
    try:
        # Verificar si ya existe
        existing = chroma_collection.get(ids=[referencia])
        if existing['ids']:
            return False  # Ya existe
        
        # Dividir texto en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Crear documento completo para dividir
        texto_completo_formateado = f"""Título: {titulo}
Año: {anio}
Referencia: {referencia}

Resumen:
{resumen}

Texto completo:
{texto_completo}"""
        
        chunks = text_splitter.split_text(texto_completo_formateado)
        
        # Generar embeddings para cada chunk
        chunk_embeddings = embeddings_model.embed_documents(chunks)
        
        # Preparar IDs y metadatos
        ids = [f"{referencia}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "referencia": referencia,
                "titulo": titulo,
                "anio": str(anio),
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            for i in range(len(chunks))
        ]
        
        # Añadir a ChromaDB
        chroma_collection.add(
            ids=ids,
            embeddings=chunk_embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        return True
    except Exception as e:
        st.error(f"Error al añadir a ChromaDB: {str(e)}")
        return False

# Función para sincronizar SQLite con ChromaDB
def sync_sqlite_to_chromadb():
    """Sincroniza todos los documentos de SQLite a ChromaDB"""
    conn = sqlite3.connect('documentos.db')
    c = conn.cursor()
    c.execute('SELECT referencia, titulo, anio, resumen, texto_completo FROM documentos')
    documentos = c.fetchall()
    conn.close()
    
    sincronizados = 0
    for doc in documentos:
        referencia, titulo, anio, resumen, texto_completo = doc
        if add_documento_to_vectordb(referencia, titulo, anio, texto_completo, resumen):
            sincronizados += 1
    
    return sincronizados, len(documentos)

# Estado para LangGraph
class RAGState(TypedDict):
    question: str
    context: Annotated[list, operator.add]
    answer: str
    sources: list

# Función para recuperar contexto relevante
def retrieve_context(state: RAGState) -> RAGState:
    """Recupera contexto relevante de ChromaDB"""
    question = state["question"]
    
    # Generar embedding de la pregunta
    question_embedding = embeddings_model.embed_query(question)
    
    # Buscar documentos similares
    results = chroma_collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )
    
    # Extraer contexto y fuentes
    context = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []
    
    # Crear lista de fuentes únicas
    sources = list(set([m['referencia'] for m in metadatas if 'referencia' in m]))
    
    return {
        "question": question,
        "context": context,
        "answer": "",
        "sources": sources
    }

# Función para generar respuesta
def generate_answer(state: RAGState) -> RAGState:
    """Genera respuesta basada en el contexto recuperado"""
    question = state["question"]
    context = state["context"]
    
    if not context:
        return {
            "question": question,
            "context": context,
            "answer": "No encontré información relevante en los documentos para responder a tu pregunta.",
            "sources": []
        }
    
    # Crear prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente experto que responde preguntas basadas EXCLUSIVAMENTE en el contexto proporcionado.

REGLAS ESTRICTAS:
1. Solo usa información del contexto proporcionado
2. Si la información no está en el contexto, di claramente que no la tienes
3. NO inventes ni añadas información externa
4. Cita las referencias cuando sea relevante
5. Sé preciso y conciso

Contexto:
{context}"""),
        ("human", "{question}")
    ])
    
    # Formatear contexto
    context_text = "\n\n".join(context)
    
    # Generar respuesta
    chain = prompt_template | llm_rag | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": question})
    
    return {
        "question": question,
        "context": context,
        "answer": answer,
        "sources": state["sources"]
    }

# Crear grafo de LangGraph
def create_rag_graph():
    """Crea el grafo de LangGraph para RAG"""
    workflow = StateGraph(RAGState)
    
    # Añadir nodos
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_answer)
    
    # Añadir edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# Inicializar grafo RAG
rag_graph = create_rag_graph()

# Función para consultar RAG
def query_rag(question: str):
    """Consulta el sistema RAG con una pregunta"""
    initial_state = {
        "question": question,
        "context": [],
        "answer": "",
        "sources": []
    }
    
    result = rag_graph.invoke(initial_state)
    return result

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

# Función para extraer título y año usando IA
def extraer_metadata_con_ia(texto):
    """Usa IA para extraer título y año del texto del PDF"""
    try:
        # Tomar solo las primeras 2000 caracteres para analizar
        texto_inicial = texto[:2000]
        
        response = client_openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente experto en análisis de documentos. Tu tarea es extraer el título y el año de publicación de documentos. Responde ÚNICAMENTE en formato JSON válido con las claves 'titulo' y 'anio'. El título suele estar al principio del documento. Si no encuentras el año, usa el año actual (2025)."
                },
                {
                    "role": "user",
                    "content": f"Analiza el siguiente texto del inicio de un documento PDF y extrae:\n1. El título del documento (generalmente es el texto más prominente al inicio, puede estar en mayúsculas o ser el primer texto significativo)\n2. El año de publicación (busca años en formato YYYY, fechas, o menciones temporales)\n\nTexto:\n{texto_inicial}\n\nResponde SOLO con JSON en este formato exacto: {{\"titulo\": \"texto del titulo\", \"anio\": 2025}}"
                }
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        # Extraer respuesta
        respuesta = response.choices[0].message.content.strip()
        
        # Intentar parsear JSON
        try:
            # Buscar JSON en la respuesta
            json_match = re.search(r'\{.*\}', respuesta, re.DOTALL)
            if json_match:
                respuesta = json_match.group()
            
            datos = json.loads(respuesta)
            titulo = datos.get("titulo", "").strip()
            anio = datos.get("anio", datetime.now().year)
            
            # Validar título
            if not titulo or len(titulo) < 3:
                titulo = "Documento sin título"
            
            # Validar año
            if isinstance(anio, str):
                anio_match = re.search(r'\d{4}', str(anio))
                if anio_match:
                    anio = int(anio_match.group())
                else:
                    anio = datetime.now().year
            
            anio = int(anio)
            if anio < 1900 or anio > 2100:
                anio = datetime.now().year
            
            return titulo, anio
            
        except json.JSONDecodeError as e:
            st.warning(f"No se pudo parsear la respuesta JSON: {respuesta}")
            return extraer_metadata_fallback(texto_inicial)
            
    except Exception as e:
        st.error(f"Error al extraer metadatos con IA: {str(e)}")
        return extraer_metadata_fallback(texto[:2000])

# Función de respaldo para extraer título y año
def extraer_metadata_fallback(texto):
    """Extrae título y año usando expresiones regulares como respaldo"""
    try:
        # Buscar año (formato YYYY)
        anio_match = re.search(r'\b(19|20)\d{2}\b', texto)
        anio = int(anio_match.group()) if anio_match else datetime.now().year
        
        # Extraer título (primeras líneas no vacías)
        lineas = [l.strip() for l in texto.split('\n') if l.strip()]
        titulo = lineas[0] if lineas else "Documento sin título"
        
        # Limpiar título (máximo 150 caracteres)
        if len(titulo) > 150:
            titulo = titulo[:150] + "..."
        
        return titulo, anio
    except:
        return "Documento sin título", datetime.now().year

# Función para generar resumen usando IA
def generar_resumen(texto, max_palabras=300):
    try:
        response = client_openai.chat.completions.create(
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
    ["📤 Subir Documento", "📦 Procesamiento en Bloque", "📋 Ver Documentos", "🔍 Buscar Documento", "🤖 Consulta RAG"]
)

# SECCIÓN: Subir Documento
if menu == "📤 Subir Documento":
    st.header("Subir Nuevo Documento PDF")
    st.info("📄 El sistema extraerá automáticamente el título y año del documento usando IA")
    
    archivo_pdf = st.file_uploader("Selecciona un archivo PDF", type=['pdf'])
    
    if st.button("Procesar Documento", type="primary"):
        if archivo_pdf is not None:
            with st.spinner("Procesando documento..."):
                # Extraer texto completo
                st.info("📝 Extrayendo texto del PDF...")
                texto = extraer_texto_pdf(archivo_pdf)
                
                if texto and len(texto.strip()) > 50:
                    # Extraer título y año con IA
                    st.info("🤖 Analizando documento para extraer título y año...")
                    titulo, anio = extraer_metadata_con_ia(texto)
                    
                    if titulo and anio:
                        st.success(f"✅ Título detectado: **{titulo}**")
                        st.success(f"✅ Año detectado: **{anio}**")
                        
                        # Generar referencia
                        numero = obtener_siguiente_numero(anio)
                        referencia = f"Art. {anio}{numero:04d}"
                        
                        # Generar resumen
                        st.info("📋 Generando resumen con IA...")
                        resumen = generar_resumen(texto, 300)
                        
                        if resumen:
                            # Insertar en la base de datos
                            if insertar_documento(referencia, titulo, anio, resumen, texto):
                                # Añadir a ChromaDB para RAG
                                st.info("📋 Añadiendo a base de datos vectorial...")
                                if add_documento_to_vectordb(referencia, titulo, anio, texto, resumen):
                                    st.success("✅ Añadido a ChromaDB para consultas RAG")
                                
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
                            st.error("No se pudo generar el resumen del documento.")
                    else:
                        st.error("No se pudo extraer el título y año del documento.")
                else:
                    st.error("No se pudo extraer suficiente texto del PDF. El documento puede estar vacío o ser una imagen escaneada.")
        else:
            st.warning("Por favor, selecciona un archivo PDF.")

# SECCIÓN: Procesamiento en Bloque
elif menu == "📦 Procesamiento en Bloque":
    st.header("Procesamiento en Bloque de PDFs")
    st.info("📦 Sube múltiples archivos PDF para procesarlos automáticamente")
    
    archivos_pdf = st.file_uploader(
        "Selecciona uno o más archivos PDF",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if archivos_pdf:
        st.write(f"📄 **Archivos seleccionados:** {len(archivos_pdf)}")
        
        # Mostrar lista de archivos
        with st.expander("Ver lista de archivos"):
            for i, archivo in enumerate(archivos_pdf, 1):
                st.write(f"{i}. {archivo.name}")
        
        if st.button("Procesar Todos los Documentos", type="primary"):
            # Barra de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Contenedor para resultados
            resultados_container = st.container()
            
            exitosos = 0
            fallidos = 0
            
            for i, archivo_pdf in enumerate(archivos_pdf):
                # Actualizar progreso
                progreso = (i) / len(archivos_pdf)
                progress_bar.progress(progreso)
                status_text.text(f"Procesando {i+1}/{len(archivos_pdf)}: {archivo_pdf.name}")
                
                try:
                    with resultados_container:
                        with st.expander(f"📄 {archivo_pdf.name}", expanded=False):
                            # Extraer texto
                            st.write("📝 Extrayendo texto...")
                            texto = extraer_texto_pdf(archivo_pdf)
                            
                            if texto and len(texto.strip()) > 50:
                                # Extraer título y año
                                st.write("🤖 Extrayendo título y año...")
                                titulo, anio = extraer_metadata_con_ia(texto)
                                
                                if titulo and anio:
                                    st.write(f"✅ **Título:** {titulo}")
                                    st.write(f"✅ **Año:** {anio}")
                                    
                                    # Generar referencia
                                    numero = obtener_siguiente_numero(anio)
                                    referencia = f"Art. {anio}{numero:04d}"
                                    st.write(f"🏷️ **Referencia:** {referencia}")
                                    
                                    # Generar resumen
                                    st.write("📋 Generando resumen...")
                                    resumen = generar_resumen(texto, 300)
                                    
                                    if resumen:
                                        # Insertar en la base de datos
                                        if insertar_documento(referencia, titulo, anio, resumen, texto):
                                            # Añadir a ChromaDB
                                            add_documento_to_vectordb(referencia, titulo, anio, texto, resumen)
                                            st.success(f"✅ Procesado exitosamente")
                                            st.write(f"**Resumen:** {resumen[:200]}...")
                                            exitosos += 1
                                        else:
                                            st.error("❌ Error al guardar en la base de datos")
                                            fallidos += 1
                                    else:
                                        st.error("❌ No se pudo generar el resumen")
                                        fallidos += 1
                                else:
                                    st.error("❌ No se pudo extraer título y año")
                                    fallidos += 1
                            else:
                                st.error("❌ Texto insuficiente en el PDF")
                                fallidos += 1
                                
                except Exception as e:
                    with resultados_container:
                        with st.expander(f"❌ {archivo_pdf.name}", expanded=False):
                            st.error(f"Error: {str(e)}")
                    fallidos += 1
            
            # Completar progreso
            progress_bar.progress(1.0)
            status_text.text("✅ Procesamiento completado")
            
            # Mostrar resumen final
            st.markdown("---")
            st.subheader("📊 Resumen del Procesamiento")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total", len(archivos_pdf))
            with col2:
                st.metric("Exitosos", exitosos)
            with col3:
                st.metric("Fallidos", fallidos)

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

# SECCIÓN: Consulta RAG
elif menu == "🤖 Consulta RAG":
    st.header("🤖 Consulta Inteligente con RAG")
    st.info("📚 Haz preguntas sobre todos los documentos almacenados. El sistema usará LangGraph y agentes de LangChain para responder sin alucinar.")
    
    # Botón para sincronizar
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Sincronizar BD"):
            with st.spinner("Sincronizando documentos..."):
                sincronizados, total = sync_sqlite_to_chromadb()
                st.success(f"✅ {sincronizados} documentos nuevos sincronizados de {total} totales")
    
    # Mostrar estadísticas de ChromaDB
    try:
        total_docs = chroma_collection.count()
        st.metric("📊 Documentos en ChromaDB", total_docs)
    except:
        st.warning("⚠️ No se pudo obtener el conteo de documentos")
    
    st.markdown("---")
    
    # Historial de chat
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Mostrar historial
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat['question'])
        with st.chat_message("assistant"):
            st.write(chat['answer'])
            if chat['sources']:
                st.caption(f"📚 **Fuentes:** {', '.join(chat['sources'])}")
    
    # Input de pregunta
    question = st.chat_input("💬 Haz una pregunta sobre los documentos...")
    
    if question:
        # Mostrar pregunta del usuario
        with st.chat_message("user"):
            st.write(question)
        
        # Procesar con RAG
        with st.chat_message("assistant"):
            with st.spinner("🤔 Buscando en los documentos..."):
                try:
                    result = query_rag(question)
                    
                    # Mostrar respuesta
                    st.write(result['answer'])
                    
                    # Mostrar fuentes
                    if result['sources']:
                        st.caption(f"📚 **Fuentes consultadas:** {', '.join(result['sources'])}")
                        
                        # Mostrar contexto usado (opcional, en expander)
                        with st.expander("🔍 Ver contexto utilizado"):
                            for i, ctx in enumerate(result['context'][:3], 1):
                                st.text_area(f"Fragmento {i}", ctx, height=100, key=f"ctx_{i}")
                    else:
                        st.warning("⚠️ No se encontraron documentos relevantes")
                    
                    # Guardar en historial
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': result['answer'],
                        'sources': result['sources']
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error al procesar la consulta: {str(e)}")
    
    # Botón para limpiar historial
    if st.session_state.chat_history:
        if st.button("🗑️ Limpiar historial"):
            st.session_state.chat_history = []
            st.rerun()

# Pie de página
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** El título y año se extraen automáticamente con IA.")
