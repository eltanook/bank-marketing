# Guion de Presentación: Equidad en Aprendizaje Automático
**Tiempo estimado:** 20 minutos (~5 minutos por orador)
**Materia:** Equidad en Aprendizaje Automático - UNSAM (Ciencia de Datos)

---

## 🎙️ Parte 1: Introducción y Contexto de los Datos
**Orador:** Rocío *(Minutos 0 - 5)*

**[Diapositiva 1: Portada]**
"Hola a todos, muy buenas tardes. Somos estudiantes de la carrera de Ciencia de Datos de la UNSAM y hoy venimos a presentarles nuestro Trabajo Práctico Integrador sobre Equidad en Aprendizaje Automático. El equipo está compuesto por Tomás, Alejandro, Matías y yo, Rocío. A lo largo de este cuatrimestre, nos propusimos un objetivo ambicioso: no solo queríamos construir un modelo predictivo bancario que funcionara bien comercialmente, sino que queríamos auditarlo a fondo, buscar sus sesgos ocultos y aplicar técnicas de mitigación algorítmica reales para asegurar que nuestras decisiones automatizadas sean éticamente justas."

**[Diapositiva 2: El Dataset: Bank Marketing]**
"Para ponernos en contexto, decidimos trabajar con un conjunto de datos sumamente famoso en la comunidad académica: el dataset de *Bank Marketing* del repositorio de la UCI (Universidad de California en Irvine). Hablamos de una base de datos con un poco más de 45.000 registros, proveniente de un banco portugués durante los años 2008 a 2010, en plena crisis financiera europea.
El objetivo comercial del banco era claro: predecir si un cliente, tras recibir una llamada telefónica, iba a suscribirse a un depósito a plazo fijo. Como llamar a toda la base de clientes es costoso e ineficiente, el banco necesitaba un modelo de minería de datos para priorizar a quién contactar.
Antes de arrancar, tuvimos que tomar una decisión de preprocesamiento clave y eliminar de cuajo la variable `duration`, que indicaba los minutos que duró la llamada. ¿Por qué? Porque representaba un caso de manual de *Data Leakage* o fuga de datos. Es imposible saber cuánto va a durar una llamada *antes* de levantar el teléfono. Además, una llamada larga suele ser consecuencia directa de que el cliente ya está interesado. Entrenar a nuestro algoritmo con esa ventaja del futuro hubiera invalidado por completo el modelo para un escenario predictivo real.
Otro gran obstáculo fue la ausencia de la variable 'género' en los datos. Tuvimos que iterar. Decidimos usar la ocupación de la persona (la variable `job`) como nuestro *proxy* o aproximación de género. Para ello, separamos roles que histórica y culturalmente han sido feminizados, como las empleadas domésticas o las administrativas, frente al resto de los oficios históricamente masculinizados o neutros."

**[Diapositiva 3: Datasheets for Datasets]**
"Previo a programar una sola línea de código, aplicamos la rigurosa metodología de *Datasheets for Datasets* propuesta por Timnit Gebru. Investigar el origen de la base nos reveló algo crítico: estos no son datos de una encuesta voluntaria. Son registros administrativos directos del banco. Esto acarrea un riesgo ético masivo.
Si bien un plazo fijo no es un préstamo, el banco está decidiendo proactivamente a quién ofrecerle instrumentos de ahorro cuyos rendimientos se pagan con el propio capital del banco. Si nosotros usamos datos históricos sesgados para decidir a quién llamar hoy, corremos el enorme riesgo de perpetuar perfiles discriminatorios y excluir sistemáticamente a grupos vulnerables del acceso a la inclusión financiera. Con esto en mente, pasamos a analizar matemáticamente el comportamiento de estos datos."

---

## 🎙️ Parte 2: Exploración y Desempeño Base
**Orador:** Tomás *(Minutos 5 - 10)*

**[Diapositiva 4: Sesgos Potenciales Iniciales]**
"Gracias Rocío. Al adentrarnos en el Análisis Exploratorio de los Datos (EDA), nos chocamos de frente con obstáculos estructurales inmensos. En primer lugar, la distribución de nuestra variable objetivo: casi el 90% de las llamadas registradas terminaban en un rotundo 'No'. Teníamos un dataset brutalmente desbalanceado, lo que de entrada sabíamos que iba a empujar a cualquier modelo a predecir siempre 'no' para asegurar un error global bajo.
Pero más allá de las etiquetas, detectamos problemas de representación poblacional, como la *paradoja de la tasa base* con la edad. Los adultos mayores, especialmente los jubilados, casi no figuraban en el dataset si los comparamos con los adultos casados de mediana edad. Sin embargo, al cruzar los datos, vimos que los jubilados tenían la tasa de suscripción real más alta de todas, rondando el 27%. Nuestro mayor miedo era que el algoritmo, al ver tan pocos ejemplos de jubilados, ignorara a esta minoría altamente rentable simplemente por comodidad estadística."

**[Diapositiva 5: Clasificación Base y Rendimiento Global]**
"Para tener un *baseline* o punto de partida, preprocesamos los datos aplicando One-Hot Encoding y un StandardScaler—crítico para que los enormes saldos bancarios no distorsionaran la matemática— y entrenamos un modelo de *Random Forest* estándar, ajustando el hiperparámetro de pesos de clase para intentar combatir el desbalance.
El reporte de clasificación nos arrojó una Exactitud o *Accuracy* global del 89%. A simple vista, un número excelente... hasta que miramos la matriz de confusión. Nuestro *Recall* o Sensibilidad para la clase que nos importaba (los que decían 'Sí') era de apenas un 20%. Es decir, el modelo era un genio prediciendo a los que iban a rechazar la oferta, pero se le escapaba el 80% de los verdaderos inversores.
Aquí tuvimos que frenar y hacernos una pregunta de negocio fundamental: ¿Cuál es el peor error posible para este banco? Llegamos a la firme conclusión de que el *Falso Negativo* era nuestro enemigo a abatir. Si el modelo predice que un cliente no quiere invertir (y nos equivocamos), no lo llamamos, perdiendo comercialmente a un inversor real y su capital frente a otro banco. En cambio, equivocarnos con un *Falso Positivo* solo nos cuesta los dos minutos de salario del operador telefónico que hace una llamada infructuosa."

**[Diapositiva 5: Evaluación Inicial (Proxy)]**
"Para cerrar el diagnóstico, evaluamos cómo este modelo base trataba a nuestro proxy de género. Observamos las tasas de Verdaderos Positivos segmentadas por ocupación: el grupo feminizado tenía un 18.49% y el resto un 19.96%. A primera vista, el modelo parecía razonablemente equitativo. Pero el problema de fondo era estructural: **ambos grupos tenían un Recall paupérrimo**, inferior al 20%. El algoritmo era ineficaz para captar a los inversores reales en todos los grupos por igual. Y con ese problema sobre la mesa, pasamos a la auditoría formal de equidad."

---

## 🎙️ Parte 3: Auditoría de Equidad y Mitigaciones
**Orador:** Matías *(Minutos 10 - 15)*

**[Diapositiva 7: Conceptos de Equidad (Fairness)]**
"Buenas tardes. Teniendo en claro que nuestro peor error era el Falso Negativo, necesitábamos definir matemáticamente qué significaba ser 'justos' en este contexto comercial.
Descartamos métricas como la Paridad Estadística porque exigir que llamemos exactamente a la misma proporción de personas en cada grupo no tiene sentido si su disposición real y orgánica al ahorro es diferente. Además, apoyándonos en los Teoremas de Imposibilidad de Equidad de autores como Kleinberg y Chouldechova, sabemos que es matemáticamente imposible satisfacer todos los criterios de equidad al mismo tiempo cuando las tasas base difieren.
Por ende, adoptamos como nuestra estrella polar el criterio de **Equal Opportunity** o Igualdad de Oportunidades. Esto exige que el modelo iguale la *Tasa de Verdaderos Positivos* (TPR) a través de todos los grupos. En criollo: si un cliente, sea del grupo históricamente masculinizado o del grupo feminizado, realmente desea invertir, el algoritmo debe detectarlo con la misma eficacia."

**[Diapositiva 8: Análisis Cuantitativo de Sesgo]**
"Sometimos a nuestro Random Forest a la auditoría cuantitativa y acá nos llevamos nuestra primera gran sorpresa del cuatrimestre. Al medir la diferencia de la Tasa de Verdaderos Positivos entre ocupaciones feminizadas (18.49%) y el resto (19.96%), descubrimos que la disparidad era de apenas un 0.0146, es decir, un 1.46%. Con un umbral de tolerancia estándar del 10%, todas las métricas —Paridad Estadística, Igualdad de Oportunidades, Paridad Predictiva— daban ✅ FAIR. En contra de nuestras hipótesis iniciales, el modelo original ya era sumamente justo y equitativo por naturaleza."

**[Diapositiva 9: Mitigación: Reweighing]**
"A pesar de esto, queríamos ver si podíamos mejorar ese paupérrimo 20% de Recall global sin romper la equidad. Comenzamos probando mitigadores académicos desde el pre-procesamiento, específicamente la técnica de *Reweighing*. Esta técnica le asigna pesos a las instancias de forma inversamente proporcional a su frecuencia, buscando que el modelo no penalice a las minorías.
¿Qué pasó? Como nuestro modelo base ya era equitativo, forzar un cambio en la distribución de los pesos fue contraproducente. La técnica terminó degradando la Tasa de Verdaderos Positivos del grupo femenino, haciéndola caer del 18% al 15%. La descartamos."

**[Diapositiva 9: Enfoque Alternativo - Ajuste Manual]**
"Viendo que los mitigadores de caja negra fallaban o eran contraproducentes, tomamos una decisión puramente empírica y enfocada en el negocio. Entramos a la capa probabilística del algoritmo y realizamos un *Logit Tuning*. Por defecto, el modelo clasifica como 'Sí' si la probabilidad supera el 0.50. Nosotros lo fuimos bajando iterativamente hasta fijar un umbral de 0.30.
El impacto fue masivo: con este simple ajuste manual logramos duplicar la cantidad de Verdaderos Positivos, pasando de 214 a **484** clientes captados, reduciendo los Falsos Negativos de más de 800 a 574, sin que la equidad entre los grupos sufriera variaciones significativas. Nuestra conclusión fue contundente: a veces un ajuste empírico simple y guiado por el negocio es más efectivo que forzar restricciones matemáticas complejas a ciegas sobre el pipeline."

**[Diapositiva 10: Mitigación Equalized Odds y Diapositiva 11: Resumen Comparativo]**
"Para agotar las instancias teóricas, también aplicamos *Equalized Odds* utilizando la librería Holistic AI, una técnica agresiva de post-procesamiento que fuerza matemáticamente la igualdad de tasas alterando las predicciones finales. Como pueden ver en la tabla resumen comparativa de la siguiente diapositiva, forzar a un modelo que ya era equilibrado a cumplir restricciones tan estrictas solo terminó empeorando la disparidad, llevándola de un 1.46% a casi un 6.73%, e introduciendo ruido algorítmico. El Modelo Base tuvo un TPR Difference de 0.0146 (✅ Muy Justo), el Random Forest con Reweighing tuvo 0.0179 (✅ Justo, con leve ruido) y el RF con Equalized Odds llegó a 0.0673 (❌ Empeoró la equidad).
Nuestra gran conclusión de esta etapa fue contundente: nuestro mejor modelo era el baseline original, empoderado operativamente con nuestro umbral manual de 0.30."

---

## 🎙️ Parte 4: Generalización y Reflexiones
**Orador:** Alejandro *(Minutos 15 - 20)*

**[Diapositiva 12: Extrapolación (Nivel Educativo)]**
"Hola a todos, para ir cerrando. Como sabemos que la escalabilidad es un pilar de la ingeniería de *Machine Learning*, quisimos poner a prueba la flexibilidad de nuestro código. Extrapolamos el pipeline entero hacia una nueva variable demográfica: el Nivel Educativo, contrastando a los clientes con educación universitaria frente a los que no la tienen. 
Al correr la misma auditoría sobre este nuevo eje, los resultados se mantuvieron robustos. Adoptamos el umbral de 0.30 como nuevo modelo base comercial —ya que demostró mejorar los Verdaderos Positivos—, aplicamos Reweighing y ajustes de umbral iterativos específicos por grupo educativo, y confirmamos la misma tendencia: el modelo base ya presentaba disparidades ínfimas, por lo que intervenir forzosamente no aportó ganancias significativas. El banco tiene ahora una arquitectura de software lista para auditar futuras variables demográficas reciclando exactamente la misma base de código."

**[Diapositiva 13: Lecciones sobre la Mitigación]**
"Todo este extenso análisis nos dejó lecciones fundamentales. La primera es que no hay que asumir el sesgo, hay que medirlo rigurosamente con un umbral de tolerancia claro —nosotros usamos el estándar del 10%—. Creíamos que íbamos a encontrar un modelo fuertemente discriminador, pero los datos nos demostraron que el modelo base era equitativo en todas las métricas.
La segunda gran lección es que en Machine Learning, 'más mitigación no siempre es mejor'. Inyectar algoritmos complejos de caja negra como *Equalized Odds* o *Reweighing* solo por cumplir con un *checklist* académico, muchas veces añade ruido y empeora métricas que ya funcionaban bien. También aprendimos que el umbral de tolerancia elegido importa: un 10% puede ser permisivo y es clave analizar la robustez con umbrales más estrictos."

**[Diapositiva 14: Reflexión en el Mundo Real]**
"Para finalizar, queremos llevar estos números a la realidad social. Entender por qué esto importa. Un algoritmo sesgado que deniega oportunidades genera ciclos de retroalimentación negativos: si el banco no llama a ciertos grupos marginados, esos grupos jamás generan un historial de éxito. En consecuencia, el modelo de la próxima década aprenderá de esos datos sesgados y confirmará falsamente que 'esos grupos no son propensos a invertir', automatizando la desigualdad a gran escala.
Por eso, en nuestro informe recomendamos realizar *Análisis Interseccionales* en el futuro. Auditar por género o por educación de manera aislada suele ser insuficiente. Imaginen a una mujer que trabaja en un oficio feminizado y que, además, solo pudo acceder a educación primaria. La vulnerabilidad sistémica se multiplica en esa intersección, y la infraestructura de código que construimos soporta plenamente realizar esos cruces interseccionales.
Como recomendación operativa directa, recomendamos que el banco modifique sus sistemas de recolección para pedir datos demográficos —como el género— de manera explícita y voluntaria. Deben abandonar las aproximaciones de los proxies inexactos. Los algoritmos no son entes neutrales; auditar y mitigar sus sesgos no es solo mover un umbral en Python, es un imperativo ético para los profesionales de los datos."

**[Diapositiva 15: Despedida]**
"Esperamos que la presentación haya sido ilustrativa. ¡Muchísimas gracias por su atención y quedamos a total disposición para responder sus preguntas!"
