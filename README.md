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
* **Decisión de Negocio:** Nos situamos en la perspectiva del banco, cuyo objetivo es aumentar la tasa de suscripciones. Concluimos que **el error más perjudicial es el Falso Negativo (FN)**, ya que omitir llamar a un cliente que habría aceptado la oferta resulta en una pérdida de negocio irrecuperable, mientras que el costo de un Falso Positivo (una llamada inútil) es marginal. Obtuvimos un Accuracy global altísimo (89%) pero un Recall pobrísimo (20%) para la clase minoritaria.

---

## 3. Análisis de Equidad (Ejercicio 3)

Se definió el marco teórico y matemático para medir la justicia de nuestro modelo usando el proxy `job`.
* **Criterio seleccionado:** Alineado a nuestra prioridad de minimizar los Falsos Negativos sin discriminar entre grupos demográficos, elegimos **Equal Opportunity (TPR)** como el criterio más relevante. Esto asegura que el banco sea igual de efectivo identificando suscriptores interesados en trabajos históricamente femeninos que en el resto.
* **Evaluación Inicial:** Al analizar el modelo base con un umbral de disparidad del 0.1, verificamos empíricamente que el modelo ya se comportaba de manera equitativa (*Fair*), presentando una brecha de apenas un 2% en TPR.

---

## 4. Mitigación de Sesgos y Ajuste de Umbral (Ejercicio 4)

A pesar de tener un modelo equitativo, el problema del bajo Recall (20%) persistía. Exploramos distintas técnicas:
1. **Pre-processing (Reweighing):** Asignamos distintos pesos a las instancias de entrenamiento. Inesperadamente, esta técnica degradó ligeramente la TPR del grupo femenino (bajó a 15.75%), rompiendo la Igualdad de Oportunidades.
2. **Post-processing (Equalized Odds):** Los optimizadores matemáticos tampoco lograron mejorar el balance, demostrando que aplicar librerías correctivas a ciegas sobre un modelo inherentemente justo no soluciona problemas intrínsecos de clasificación.
3. **Logit Tuning (Ajuste Manual de Umbral):** Nuestra solución óptima fue bajar el umbral de decisión a **0.30**. Con esto logramos duplicar los Verdaderos Positivos sin quebrar en ningún momento la equidad distributiva inicial.

---

## 5. Generalización a Nuevas Variables (Ejercicio 5)

Para probar la escalabilidad de nuestra arquitectura y análisis, replicamos los cálculos evaluando la variable **`education`** (nivel educativo).
* El pipeline demostró ser agnóstico a la variable protegida, permitiendo medir fácilmente las métricas para grupos con educación primaria, secundaria y terciaria.
* **Reflexión Interseccional:** Se deja sentada la base para futuras auditorías de equidad que crucen estas dimensiones simultáneamente (ej. oficios feminizados + educación primaria), garantizando que las decisiones automatizadas no refuercen exclusiones históricas cruzadas.

---

*Nota: Los análisis descartados (como pruebas de validación con variables alternativas como `marital`) se conservan a modo de registro histórico en la carpeta `/desestimado`.*
