import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



#Se carga la DATA
df = pd.read_excel('dataspam.xlsx')

#Se explora el conjunto de datos


#Limpiamos los datos
def limpiar_texto(texto):
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(r'\W', ' ', texto)  # Eliminar caracteres especiales
    texto = re.sub(r'\s+', ' ', texto)  # Eliminar espacios extras
    palabras = texto.split()
    palabras_limpias = [palabra for palabra in palabras if palabra not in stopwords.words('english')]
    return ' '.join(palabras_limpias)


df['limpio'] = df['Body'].apply(limpiar_texto)
print(df[['Body', 'limpio']])

#Convertir el texto en una matriz de características usando TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['limpio']).toarray()
y = df['Spam']

#Division de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entrenando el modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)


#Evaluando el modelo
y_pred = modelo.predict(X_test)

precision = accuracy_score(y_test, y_pred)
print(f'Precisión: {precision * 100:.2f}%')

print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))


#Analizando resultados

#Obteniendo los coeficientes
coeficientes = modelo.coef_[0]
palabras = vectorizer.get_feature_names_out()
#Ordenando palabras por importancia de clasificacion
top_palabras_indices = np.argsort(coeficientes)[-10:] 
top_palabras = [palabras[i] for i in top_palabras_indices]


print("Palabras más indicativas de spam:", top_palabras)