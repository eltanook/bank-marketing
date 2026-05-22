# Equidad en Aprendizaje Automático: Bank Marketing

Este repositorio contiene la resolución del Trabajo Práctico de Equidad en Aprendizaje Automático, aplicado al conjunto de datos de Marketing Bancario. A lo largo del proyecto, analizamos los sesgos presentes en los datos, entrenamos modelos predictivos y aplicamos técnicas de mitigación para garantizar decisiones automatizadas más justas.

---

## 1. Introducción al Conjunto de Datos (Ejercicio 1)

El análisis se fundamenta en el **Bank Marketing Dataset** proveniente del repositorio UCI. 
* **Motivación:** Demostrar cómo las técnicas de minería de datos pueden predecir el éxito del telemarketing bancario, permitiendo priorizar a los clientes con mayor probabilidad de suscribir a un depósito a plazo fijo.
* **Instancias:** Representan contactos telefónicos individuales realizados por una institución bancaria portuguesa entre 2008 y 2010.
* **Variable Protegida (Proxy):** El conjunto original no contiene información directa sobre el género de los clientes. Tras nuestro análisis exploratorio, establecimos el atributo **`job`** como proxy de género, identificando ocupaciones históricamente femeninas (como `housemaid` y `admin.`) frente al resto de trabajos.

---

## 2. Modelo Predictivo (Ejercicio 2)

Construimos un modelo base de clasificación para predecir si un cliente suscribirá (`yes`) o no (`no`) al depósito.
* **Algoritmo elegido:** Random Forest.
* **Prevención de Data Leakage:** Eliminamos la variable `duration` del entrenamiento, ya que la duración exacta de una llamada solo se conoce una vez que esta ha finalizado (momento en el que también se conoce el resultado).
* **Decisión de Negocio:** Nos situamos en la perspectiva del banco, cuyo objetivo es aumentar la tasa de suscripciones. Concluimos que **el error más perjudicial es el Falso Negativo (FN)**, ya que omitir llamar a un cliente que habría aceptado la oferta resulta en una pérdida de negocio irrecuperable, mientras que el costo de un Falso Positivo (una llamada inútil) es marginal.

---

## 3. Análisis de Equidad (Ejercicio 3)

Se definió el marco teórico y matemático para medir la justicia de nuestro modelo usando el proxy `job`.
* **Criterio seleccionado:** Alineado a nuestra prioridad de minimizar los Falsos Negativos sin discriminar entre grupos demográficos, elegimos **Equal Opportunity (TPR)** como el criterio más relevante. Esto asegura que el banco sea igual de efectivo identificando suscriptores interesados en trabajos históricamente femeninos que en el resto.
* **Evaluación Inicial:** Al analizar el modelo base con un umbral de disparidad del 0.1, verificamos que el modelo ya se comportaba de manera equitativa (*Fair*) en la mayoría de sus métricas, incluyendo la Igualdad de Oportunidades.

---

## 4. Mitigación de Sesgos (Ejercicio 4)

Para garantizar la equidad y observar el efecto de la intervención algorítmica, implementamos rutinas de mitigación utilizando la librería profesional **Holistic AI**:
1. **Pre-processing (Reweighing):** Asignamos distintos pesos a las instancias de entrenamiento para balancear la representación y probabilidad cruzada de las clases frente a nuestro proxy de género.
2. **Post-processing (Equalized Odds):** Ajustamos los umbrales de decisión post-predicción, recurriendo a algoritmos de programación lineal para minimizar la diferencia en las tasas de verdaderos y falsos positivos de los grupos.

---

## 5. Comparación y Reflexión Final (Ejercicio 5)

La comparativa entre el modelo base y los modelos mitigados demuestra que **es posible garantizar la equidad algorítmica sin un costo significativo en la performance del modelo**.

* Ambas intervenciones mantuvieron el Accuracy y el F1-Score originales.
* El pre-procesamiento (*Reweighing*) logró reducir aún más la leve disparidad pre-existente en el *True Positive Rate* (TPR), solidificando la Igualdad de Oportunidades.
* **Reflexión:** En aplicaciones financieras del mundo real, integrar estos controles no solo mitiga el riesgo ético, reputacional y legal por discriminación, sino que demuestra que los objetivos comerciales (captar clientes) pueden alinearse armónicamente con políticas de equidad social si el modelado es riguroso e intencionado.

---

*Nota: Los análisis descartados (como pruebas de validación con variables alternativas como `marital`) se conservan a modo de registro histórico en la carpeta `/desestimado`.*
