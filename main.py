#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as mp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import findspark
findspark.init()
from pyspark import SparkContext
sc = SparkContext.getOrCreate()

def descargarDatosDB(lexicon_terminos_negativos, lexicon_terminos_positivos, stopwords):
    from pymongo import MongoClient
    import urllib
    usuario = urllib.parse.quote("amgc")
    contrasena = urllib.parse.quote("am1996")
    connection = MongoClient("mongodb+srv://" + usuario + ":" + contrasena + "@cluster0.uxnqk.mongodb.net/tfm-amgc?retryWrites=true&w=majority")
    db = connection['tfm-amgc']
    for termino_neg in db['lexicon-neg'].find({}, {'_id' : 0}):
        lexicon_terminos_negativos.append(termino_neg['termino'])
    for termino_pos in db['lexicon-pos'].find({}, {'_id' : 0}):
        lexicon_terminos_positivos.append(termino_pos['termino'])
    for sw in db['stopwords'].find({}, {'_id':0}):
        if (sw['termino'] not in ['no', 'nada', 'nunca', 'ni', 'jamas', 'tampoco', 'siquiera']):
            stopwords.append(sw['termino'])
            
    connection.close()

@st.cache
def cargarDatasetEntrenamiento(lexicon_terminos_negativos, lexicon_terminos_positivos, stopwords):
    mensajesDF = pd.read_csv('mensajes.csv', sep=',')

    # Eliminamos las columnas anómalas, sin datos, de la estructura DataFrame. 
    mensajesDF.drop(mensajesDF.columns[[2,3,4]], axis = 1, inplace = True)

    # Eliminamos registros vacíos, con valores no disponibles
    mensajesDF.dropna(inplace=True)

    # Eliminamos los marcadores de tabulación, retorno y tabulación de las cadenas
    mensajesDF['mensaje'].replace(to_replace=[r"\\n|\\t|\\r", "\t|\n|\r"], value="", regex = True, inplace = True)
    # Eliminamos los nicks de usuario de Twitter así como los hashtags
    mensajesDF['mensaje'].replace(to_replace=[r"(@|#)[^\s]+", "(@|#)[^\s]+"], value="", regex = True, inplace = True)
    # Eliminamos los enlaces externos (URLs) 
    mensajesDF['mensaje'].replace(to_replace=[r"^http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", 
                                    "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"], value="", regex = True, inplace = True)

    # Eliminamos los signos de puntuación
    mensajesDF['mensaje'].replace(to_replace=[r"[...|.|,|;]+"], value="", regex = True, inplace = True)

    # Eliminamos los indicadores de retuiteo (RT)
    mensajesDF['mensaje'].replace(to_replace=[r"RT"], value="", regex = True, inplace = True)

    # Eliminamos los EMOJIS
    import re
    regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
                            "]+", flags = re.UNICODE)

    mensajesDF['mensaje'] = mensajesDF['mensaje'].apply(lambda x : regrex_pattern.sub(r'',x))

    # Ponemos todo texto en minúsculas
    mensajesDF['mensaje'] = mensajesDF['mensaje'].str.lower()

    # Eliminamos las llamadas STOPWORDS

    #print(mensajesDF[mensajesDF.sentimiento == "positivo"].count)
    mensajesDF['mensaje'] = mensajesDF['mensaje'].str.split(' ').apply(lambda x: ' '.join(k for k in x if k not in stopwords))

    # print(pd.isnull(mensajesDF['mensaje']).describe())

    # Optimizamos el dataset en cuanto a categorización de sentimientos
    from nltk.util import ngrams
    i = 0
    for indice, registro in mensajesDF.iterrows():
        registro_aux = re.sub("[^\w\s]", "", str(registro['mensaje'])).split()
        terminos_positivos = 0
        terminos_negativos = 0
        terminos_en_total = len(registro_aux)
        
        if terminos_en_total > 0:
            fragmentos_mensaje = list(ngrams(registro_aux, 3, pad_left=True))
            hay_termino_positivo_relevante = False
            terminos_positivos_relevantes = 0
            hay_termino_negativo_relevante = False
            terminos_negativos_relevantes = 0
            for fragmento in fragmentos_mensaje:
                terminos_previos = fragmento[:2]
                termino_actual = fragmento[2]
                
                if termino_actual in lexicon_terminos_positivos:
                    if any(termino_ in terminos_previos for termino_ in ['no', 'nada', 'nunca', 'ni', 'jamas', 'tampoco', 'siquiera']):
                        terminos_negativos += 1
                    else:
                        terminos_positivos += 1
                        # Tratamos de buscar un término relevante, que detemine valores extremos
                        #if (hay_termino_positivo_relevante != True):
                        emocion = re.match(r"(emoci|emotiv)[a-zA-Z]*", termino_actual)
                        alegria = re.match(r"(alegr)[a-zA-Z]*", termino_actual)
                        felicidad = re.match(r"(felic|feliz)[a-zA-Z]*", termino_actual)
                        maravilla = re.match(r"(maravill)[a-zA-Z]*", termino_actual)
                        encanto = re.match(r"(encant)[a-zA-Z]*", termino_actual)
                        pasion = re.match(r"[a]*(pasion)[a-zA-Z]*", termino_actual)
                        amor = re.match(r"(enamor|am(a|e|o)|amab|amar[^g])[a-zA-Z]*", termino_actual)
                        gustar = re.match(r"(gust)[a-zA-Z]*", termino_actual)
                        agrado_paladar = re.match(r"(sabr|delici)[a-zA-Z]*", termino_actual)
                        excitar = re.match(r"(excit)[a-zA-Z]*", termino_actual)
                        impresionar = re.match(r"(impresi[o-ó]n)[a-zA-Z]*", termino_actual)
                        agradecido = re.match(r"(agradez|agradec)[a-zA-Z]*", termino_actual)
                        querer = re.match(r"(quiero|querem|querre)[a-zA-Z]*", termino_actual)
                        adorar = re.match(r"(adoro|adora|adore)[a-zA-Z]*", termino_actual)
                        divertir = re.match(r"(diverti|diviert|divertid)[a-zA-Z]*", termino_actual)
                        hay_termino_positivo_relevante = divertir or adorar or querer or impresionar or agradecido or emocion or alegria or felicidad or maravilla or encanto or pasion or amor or gustar or agrado_paladar or excitar          
                        if (hay_termino_positivo_relevante):
                            terminos_positivos_relevantes+=1
                elif termino_actual in lexicon_terminos_negativos:
                    if any(termino_ in terminos_previos for termino_ in ['no', 'nada', 'nunca', 'ni', 'jamas', 'tampoco', 'siquiera']):
                        terminos_positivos += 1
                    else:
                        terminos_negativos += 1
                        depresion = re.match(r"(depri|deprim|depresi)[a-zA-Z]*", termino_actual)
                        angustia = re.match(r"(angust)[a-zA-Z]*", termino_actual)
                        desanimo = re.match(r"(desanim)[a-zA-Z]*", termino_actual)
                        decepcion = re.match(r"(decepcion)[a-zA-Z]*", termino_actual)
                        tristeza = re.match(r"(trist)[a-zA-Z]*", termino_actual) 
                        desesperacion = re.match(r"(desespe)[a-zA-Z]*", termino_actual)
                        descontento = re.match(r"(descont)[a-zA-Z]*", termino_actual)
                        morir = re.match(r"(morir)[a-zA-Z]*", termino_actual)
                        suicidarse = re.match(r"(suicid)[a-zA-Z]*", termino_actual)
                        horrorizar = re.match(r"(horrori)[a-zA-Z]*", termino_actual)
                        atemorizado = re.match(r"(atemori)[a-zA-Z]*", termino_actual)
                        amenazas = re.match(r"(amenaz)[a-zA-Z]*", termino_actual)
                        aburrir = re.match(r"(aburr)[a-zA-Z]*", termino_actual)
                        susto = re.match(r"[a*](sust)[ad|ar|e]*[a-zA-Z]*", termino_actual)
                        enfado = re.match(r"(enfad|enoj(a|o))[a-zA-Z]*", termino_actual)
                        estres = re.match(r"(estres)[a-zA-Z]*", termino_actual)
                        alterar = re.match(r"(alter)[a-zA-Z]*", termino_actual)
                        nervios = re.match(r"(enerv|nervios)[a-zA-Z]*", termino_actual)
                        enfu_erse = re.match(r"(enfurec|enfurr)[a-zA-Z]*", termino_actual)
                        hay_termino_negativo_relevante = enfu_erse or alterar or nervios or enfado or decepcion or susto or amenazas or aburrir or horrorizar or morir or suicidarse or depresion or angustia or desanimo or tristeza or desesperacion or descontento
                        if (hay_termino_negativo_relevante):
                            terminos_negativos_relevantes+=1
                    
                        
            if (registro['sentimiento'] == 'positivo' and 
                (terminos_negativos_relevantes < terminos_positivos_relevantes or terminos_positivos > terminos_negativos)):
                registro['sentimiento'] = 'muy positivo'
            
            if(registro['sentimiento'] == 'negativo' and 
                (terminos_positivos_relevantes < terminos_negativos_relevantes or terminos_negativos > terminos_positivos)):
                registro['sentimiento'] = 'muy negativo'
                
    return mensajesDF

@st.cache
def prepararPipeline(data_train, data_test, target_train, target_test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Create feature vectors
    vectorizer = TfidfVectorizer(use_idf = True, lowercase=True)
    from procesadorlexicon import ProcesadorLexicon
    lexicon = ProcesadorLexicon(lexicon_terminos_negativos, lexicon_terminos_positivos)
    from sklearn.svm import SVC, LinearSVC
    svm = SVC(C = 0.9, kernel='linear', decision_function_shape = 'ovo')
    from sklearn.feature_selection import SelectPercentile, chi2
    selectPercentile = SelectPercentile(chi2, percentile=95)

    from sklearn.pipeline import Pipeline, FeatureUnion
    pipeline = Pipeline([
        ('feats', FeatureUnion([
            ('vectorizer', vectorizer),
            ('lexicon', lexicon)
        ])),
        ('select', selectPercentile),
        ('classifier', svm)
    ])

    pipeline.fit(data_train, target_train)

    preds = pipeline.predict(data_test)

    from sklearn.metrics import accuracy_score
    print("Exactitud =", accuracy_score(target_test,preds))
    from sklearn.metrics import classification_report
    print("Classification report:")
    clas_report = classification_report(target_test,preds)
    print(clas_report)

    from sklearn.metrics import multilabel_confusion_matrix
    print(multilabel_confusion_matrix(target_test,preds,labels=["muy negativo", "muy positivo", "negativo", "positivo"]))

    return pipeline


def hacerRecuentoDatasetUsuario(mensajesUsuario, pipeline):
    terminos_pos = 0
    terminos_neg = 0
    terminos_muypos = 0
    terminos_muyneg = 0
    for mensaje in mensajesUsuario:
        clasificacion_mensaje = pipeline.predict([mensaje])
        if (clasificacion_mensaje == "positivo"):
            terminos_pos+=1
        elif (clasificacion_mensaje == "muy positivo"):
            terminos_muypos+=1
        elif (clasificacion_mensaje == "negativo"):
            terminos_neg+=1
        elif (clasificacion_mensaje == "muy negativo"):
            terminos_muyneg+=1
            
    return [terminos_pos, terminos_neg, terminos_muypos, terminos_muyneg]

data_load_state = st.text('Cargando datos de entrenamiento...')
lexicon_terminos_negativos = []
lexicon_terminos_positivos = []
stopwords = []
descargarDatosDB(lexicon_terminos_negativos, lexicon_terminos_positivos, stopwords)
mensajesDF = cargarDatasetEntrenamiento(lexicon_terminos_negativos, lexicon_terminos_positivos, stopwords)
data_train, data_test, target_train, target_test = train_test_split(mensajesDF['mensaje'],mensajesDF['sentimiento'],train_size=0.75,random_state=0)
data_load_state.text('Comenzando procesamiento...')
pipeline = prepararPipeline(data_train, data_test, target_train, target_test)

filename = st.file_uploader("Elegir dataset de mensajes: ", type=['csv'])

if filename is not None:
    data_load_state.text('Procesando dataset a evaluar...')
    mensajes_usuario = pd.read_csv(filename.name, sep=',')
    st.write("MENSAJES DEL USUARIO")
    st.write(mensajes_usuario["mensaje"])
    total_clasificacion = hacerRecuentoDatasetUsuario(mensajes_usuario["mensaje"], pipeline)
    total_mensajes = len(mensajes_usuario)

    tags_sentimientos = ['POSITIVIDAD', 'NEGATIVIDAD']
    dimensiones = [total_clasificacion[2]+total_clasificacion[0], total_clasificacion[1]+total_clasificacion[3]]
    tags_subniveles = ['Alta', 'Normal', 'No alarmante', 'Preocupante']
    dimensiones_subniveles = [total_clasificacion[2], total_clasificacion[0], total_clasificacion[1], total_clasificacion[3]]
    colores_tags_sentimientos = ['#00FF74', '#FF4C4C']
    colors_tags_subniveles = ['#01B001', '#00FF00', '#FB1818', '#FF2700']
    
    bigger = mp.pie(dimensiones, labels=tags_sentimientos, colors=colores_tags_sentimientos,
                 startangle=45, frame=True, counterclock = False)
    smaller = mp.pie(dimensiones_subniveles, labels=tags_subniveles,
                  colors=colors_tags_subniveles, radius=0.66,
                  startangle=45, labeldistance=0.7, counterclock = False)
    centre_circle = mp.Circle((0, 0), 0.3, color='white')
    fig = mp.gcf()
    fig.gca().add_artist(centre_circle)
    mp.axis('equal')
    mp.tight_layout()
    st.write(fig)
    mp.savefig('sentimientos_usuario_XXX.jpg')
    #mp.show()

    etiquetas_fechas = mensajes_usuario["fecha"].unique()
    clasif_mensajes_porfechas = []
    recuento_positivos = []
    recuento_negativos = []
    recuento_muynegativos = []
    for fecha in etiquetas_fechas:
        recuento = hacerRecuentoDatasetUsuario(mensajes_usuario[mensajes_usuario["fecha"] == fecha]["mensaje"], pipeline)
        clasif_mensajes_porfechas.append([fecha, recuento])
        recuento_positivos.append(recuento[0] + recuento[2])
        recuento_negativos.append(recuento[1] + recuento[3])
        recuento_muynegativos.append(recuento[3])

    anchor_barras = 0.2

    from matplotlib.ticker import MaxNLocator

    fig1, ax = mp.subplots()

    ax.bar(etiquetas_fechas, recuento_positivos, anchor_barras, label='Positivos', color=colores_tags_sentimientos[0])
    ax.bar(etiquetas_fechas, recuento_negativos, anchor_barras, bottom=recuento_positivos, label='Negativos', color=colores_tags_sentimientos[1])
    mp.plot(recuento_muynegativos, 'y')

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('Recuento +/-')
    ax.set_title('Evolución del estado de ánimo')
    ax.legend()
    st.write(fig1)
    mp.savefig('evolucion_sentimientos_XXX.jpg')
    mp.show()

    
    data_load_state.text('He aquí los resultados del procesamiento evaluador de los mensajes:')