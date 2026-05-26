# Guion de Presentación: Equidad en Aprendizaje Automático
**Tiempo estimado:** 20 minutos (~5 minutos por orador)
**Materia:** Equidad en Aprendizaje Automático - UNSAM (Ciencia de Datos)

---

## 🎙️ Parte 1: Introducción y Contexto de los Datos
**Orador:** ... *(Minutos 0 - 5)*

**[Diapositiva 1: Portada]**
"Hola a todos, buenos días/tardes. Somos estudiantes de Ciencia de Datos de la UNSAM y hoy venimos a presentar nuestro Trabajo Práctico Integrador sobre Equidad en Aprendizaje Automático. Nuestro equipo está compuesto por Tomás, Alejandro, Matías y Rocío. Durante el proyecto, nos propusimos analizar a fondo un modelo predictivo, auditar sus sesgos y aplicar técnicas de mitigación reales sobre un conjunto de datos bancario."

**[Diapositiva 2: El Dataset: Bank Marketing]**
"Para ponernos en contexto, trabajamos con el famoso dataset *Bank Marketing*. El objetivo comercial del banco era predecir si un cliente iba a suscribirse a un depósito a plazo fijo, para poder optimizar sus campañas de telemarketing y no llamar en vano. 
Al no tener una variable explícita de género en los 45.000 registros, tuvimos que iterar. Inicialmente intentamos usar el Estado Civil como variable protegida, pero lo desestimamos al ver que inyectaba mucho ruido circunstancial. Por ende, tomamos la decisión final de usar la ocupación (`job`) como nuestra variable protegida o *proxy* de género, separando roles históricamente feminizados como empleadas domésticas o administrativas, frente al resto de los oficios."

**[Diapositiva 3: Datasheets for Datasets]**
"Antes de tocar un algoritmo, aplicamos la metodología de *Datasheets for Datasets*. Descubrimos algo crítico: los datos provienen de registros administrativos reales del 2008 al 2010. Esto trae un riesgo ético importante, ya que usar datos directos del sistema perpetúa los sesgos históricos de exclusión financiera del propio banco. 
También tomamos una decisión de preprocesamiento clave: eliminamos la variable `duration` (la duración de la llamada) porque representaba un caso clarísimo de *Data Leakage*. No podemos saber cuánto dura una llamada antes de hacerla, así que entrenar el modelo con esa ventaja lo invalidaría en la vida real."

---

## 🎙️ Parte 2: Exploración y Desempeño Base
**Orador:** ... *(Minutos 5 - 10)*

**[Diapositiva 4: Sesgos Potenciales Iniciales]**
"Al explorar los datos, nos encontramos con obstáculos estructurales fuertes. El 90% de las llamadas en la base de datos terminaban en un 'No'. Teníamos un dataset brutalmente desbalanceado. Además, detectamos subrepresentación en grupos clave: por ejemplo, los adultos mayores casi no figuraban frente a los adultos jóvenes, a pesar de que los jubilados tenían la mayor tasa real de suscripción (lo que se conoce como la paradoja de la tasa base). Nuestro mayor miedo acá era que el algoritmo terminara ignorando a estas minorías rentables para priorizar la comodidad de predecir a la mayoría."

**[Diapositiva 5: Clasificación Base y Rendimiento Global]**
"Para tener un punto de partida, entrenamos un modelo *Random Forest* estándar. Globalmente dio un *Accuracy* del 89%, lo cual suena excelente... hasta que te das cuenta de que solo es bueno prediciendo a los que van a rechazar la oferta. 
Acá nos hicimos una pregunta de negocio: ¿Cuál es el peor error posible? Llegamos a la conclusión de que el *Falso Negativo* es nuestro enemigo. Si el modelo dice que un cliente no quiere el plazo fijo, no lo llamamos, y perdemos comercialmente a un inversor real. Un *Falso Positivo*, en cambio, solo nos cuesta un par de minutos de un operador telefónico."

**[Diapositiva 6: Evaluación Inicial (Proxy)]**
"Al evaluar cómo este modelo trataba a nuestro proxy de género, notamos que a las ocupaciones feminizadas se les asignaban predicciones negativas de forma desproporcionada. El modelo estaba siendo muy ineficaz para encontrar a las personas de ese grupo que realmente querían invertir."

---

## 🎙️ Parte 3: Auditoría de Equidad y Mitigaciones
**Orador:** ... *(Minutos 10 - 15)*

**[Diapositiva 7: Conceptos de Equidad (Fairness)]**
"¿Qué tal? Sabiendo esto, pasamos a definir matemáticamente qué es 'justo' para este caso. Descartamos la paridad estadística porque la disposición real a suscribirse varía orgánicamente. Como vimos recién, nuestro error crítico era el Falso Negativo, por lo que adoptamos el criterio de **Equal Opportunity** (Igualdad de Oportunidades). Exigimos que el modelo tenga la misma *Tasa de Verdaderos Positivos* (TPR) para todos; si un cliente va a invertir, el modelo debe detectarlo sin importar su ocupación."

**[Diapositiva 8: Análisis Cuantitativo de Sesgo]**
"Acá nos llevamos nuestra primera gran sorpresa. Al auditar matemáticamente el *Random Forest* base, descubrimos que la disparidad era mínima. La diferencia de TPR era de apenas un 1.46%. Es decir, el modelo original ya era sumamente justo y equitativo por naturaleza."

**[Diapositiva 9: Enfoque Alternativo - Ajuste Manual]**
"Viendo que el modelo era justo pero comercialmente pobre (detectaba muy pocos inversores), tomamos una decisión empírica. Entramos a la capa probabilística del algoritmo y bajamos el umbral de decisión de 0.50 a 0.30. El impacto fue masivo: ¡logramos duplicar la cantidad de Verdaderos Positivos, saltando de 214 a 480 clientes captados, sin perder equidad!"

**[Diapositiva 10, 11 & 12: Mitigaciones Teóricas y Resumen]**
"Aun así, probamos técnicas de mitigación académicas para ver si podíamos mejorar la equidad preexistente. Aplicamos *Reweighing* y *Equalized Odds*. Como se ve en la tabla comparativa, forzar a un modelo que ya era justo a cumplir restricciones matemáticas estrictas solo terminó empeorando la disparidad e introduciendo ruido. Concluimos que nuestro mejor modelo era el Base, empoderado con nuestro umbral manual de 0.30."

---

## 🎙️ Parte 4: Generalización y Reflexiones
**Orador:** ... *(Minutos 15 - 20)*

**[Diapositiva 13: Extrapolación (Nivel Educativo)]**
"Hola a todos. Ya con nuestro *pipeline* funcionando, quisimos poner a prueba la flexibilidad técnica del código. Extrapolamos el análisis entero hacia una nueva variable protegida: el Nivel Educativo (universitarios contra no universitarios). Los resultados fueron consistentes, confirmando que el banco puede auditar futuras características demográficas reutilizando nuestra misma base arquitectónica."

**[Diapositiva 14: Lecciones sobre la Mitigación]**
"Esta auditoría nos dejó lecciones fundamentales. Primero: no hay que asumir el sesgo, hay que medirlo. Nuestro modelo resultó ser justo antes de que lo toquemos. Segundo: más mitigación no es mejor. Inyectar algoritmos de caja negra para forzar equidad muchas veces añade ruido y empeora los resultados. Por eso es vital hacer pruebas empíricas rigurosas."

**[Diapositiva 15: Reflexión en el Mundo Real]**
"Para cerrar, es clave entender que un algoritmo sesgado genera ciclos de retroalimentación negativos: si el banco no llama a ciertos grupos, esos grupos no generan historial, y los futuros modelos aprenderán falsamente que 'no son exitosos'. 
A su vez, en nuestro trabajo futuro recomendamos fuertemente realizar análisis interseccionales. Auditar por género o por educación por separado no basta; sabemos que la vulnerabilidad se dispara cuando los ejes de identidad se cruzan de manera concurrente.
Nuestra recomendación operativa formal para la empresa es **modificar sus sistemas de recolección**. Deben pedir datos demográficos explícitos de forma voluntaria para poder abandonar proxies inexactos y medir esta interseccionalidad real. Mitigar sesgos no es solo mover un umbral en Python, es un imperativo ético para los Científicos de Datos."

**[Diapositiva 16: Despedida]**
"¡Muchas gracias por su atención! Quedamos a disposición por si tienen alguna pregunta."
