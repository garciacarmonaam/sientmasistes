from sklearn import preprocessing
import nltk
import re
from nltk.util import ngrams


class ProcesadorLexicon():
    
    PALABRAS_REVERSION = ['no', 'nada', 'nunca', 'ni', 'jamas', 'tampoco', 'siquiera']
    
    def __init__(self, terminos_negativos, terminos_positivos):
        self.terminos_negativos = terminos_negativos
        self.terminos_positivos = terminos_positivos
        
    def transform(self, relacion_mensajes, y=None):
        resultado_a_normalizar = []
        
        for mensaje in relacion_mensajes:
            puntuacion_neg_pos_mensaje = self.medir_puntuacion_mensaje(mensaje)
            resultado_a_normalizar.append(puntuacion_neg_pos_mensaje)
                
        return preprocessing.normalize(resultado_a_normalizar)
        
    def medir_puntuacion_mensaje(self, mensaje):
        terminos_positivos = 0
        terminos_negativos = 0
        mensaje_tokenizado = re.sub("[^\w\s]", "", str(mensaje)).split()
        
        fragmentos_mensaje = list(ngrams(mensaje_tokenizado, 3, pad_left=True))
        terminos_positivos_relevantes = 0
        terminos_negativos_relevantes = 0
        for fragmento in fragmentos_mensaje:
            terminos_previos = fragmento[:2]
            termino_actual = fragmento[2]
            
            if termino_actual in self.terminos_positivos:
                if any(termino_ in terminos_previos for termino_ in self.PALABRAS_REVERSION):
                    terminos_negativos += 1
                else:
                    terminos_positivos += 1
                    # Tratamos de buscar un término relevante, que detemine valores extremos
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
            elif termino_actual in self.terminos_negativos:
                if any(termino_ in terminos_previos for termino_ in self.PALABRAS_REVERSION):
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
                    
        return [terminos_positivos, terminos_negativos, terminos_positivos_relevantes*2, terminos_negativos_relevantes*2]
        
    
    def fit(self, X, y=None):
        return self
        
            
        
        