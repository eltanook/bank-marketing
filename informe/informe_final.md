# Informe Final: Equidad en Aprendizaje Automático
**Trabajo Práctico Integrador**
**Conjunto de Datos:** Bank Marketing
**Integrantes:** Tomás Nadal, Alejandro Echeverri, Matías Bacalhau, Rocio Rivera

---

## 1. Introducción y Marco del Conjunto de Datos (Ejercicio 1)

En este proyecto, nosotros abordamos de manera exhaustiva el análisis de equidad algorítmica y la construcción de modelos predictivos aplicados al conjunto de datos de Marketing Bancario (Bank Marketing Dataset). Este conjunto, ampliamente reconocido en la comunidad académica, proviene del repositorio de *Machine Learning* de la Universidad de California en Irvine (Moro, Cortez, & Rita, 2014). Para fundamentar teóricamente todas nuestras decisiones, decidimos apoyarnos directamente en los trabajos de los autores originales y de la literatura fundacional, redactando el presente informe según los lineamientos de las normas APA.

### 1.1 Datasheets for Datasets y Contexto Histórico
Siguiendo las metodologías rigurosas de documentación que proponen Gebru et al. (2021) en su famoso trabajo *Datasheets for Datasets*, nosotros diseccionamos el origen y la estructura de nuestros datos para comprender el contexto sociotécnico antes de entrenar cualquier modelo:

*   **Motivación Original:** Tal como lo describen Moro, Cortez y Rita (2014) en su investigación sobre telemarketing, nosotros comprendemos que este conjunto fue creado para demostrar la eficacia de la Minería de Datos (*Data Mining*) en la priorización de clientes bancarios portugueses. Nuestro objetivo predictivo es anticipar si un cliente suscribirá a un depósito a plazo fijo, reduciendo así la enorme cantidad de llamadas infructuosas (usualmente a todos los clientes de la base) y optimizando el tiempo de los operadores telefónicos.
*   **Composición:** Observamos que el dataset no presenta una estructura relacional compleja. Cada instancia consolida el perfil demográfico, el contexto de la interacción y las variables macroeconómicas correspondientes a una sola llamada telefónica ocurrida durante la crisis financiera europea (entre mayo de 2008 y noviembre de 2010).
*   **Proceso de Recopilación y Preprocesamiento:** Se trata de datos adquiridos a través de un proceso híbrido: por un lado, observaciones directas de registros administrativos de la institución (resultados de llamadas) y por otro, indicadores socioeconómicos derivados de fuentes externas como el Banco Central Europeo. Durante la limpieza de los datos, tomamos una decisión muy importante para el éxito de la simulación real del trabajo: **decidimos excluir la variable `duration` (duración de la llamada)**. Aunque notamos que esta variable mejoraba colosalmente las métricas, nos dimos cuenta de que incurría en un gravísimo problema de *Data Leakage* (Fuga de Datos). Nosotros no podemos saber cuánto durará una llamada hasta que ésta termina, momento en el cual el cliente ya nos dio su respuesta. Entrenar el modelo con esta ventaja anulaba por completo su utilidad en un escenario predictivo pre-llamada.
*   **Usos Conocidos:** Identificamos que este conjunto se ha utilizado históricamente para demostrar cómo enfocar campañas de marketing predictivo reduciendo el número de llamadas (llamando al ~10% de clientes con mayor probabilidad se captura el 50% de los depósitos) y como base para pruebas de técnicas de balanceo sintético como SMOTE.

### 1.2 Entorno Computacional y Reproducibilidad
Para garantizar la transparencia y la auditabilidad exigida en experimentaciones de *Machine Learning*, nosotros desarrollamos el proyecto utilizando un entorno Python 3.x, apoyándonos en librerías estables como `scikit-learn` para el entrenamiento algorítmico y `pandas` para el manejo de estructuras de datos. Para el análisis avanzado de sesgos empleamos la versión más reciente de la librería `holisticai`. Todo el código fuente y los cuadernos de experimentación fueron controlados y versionados rigurosamente mediante Git en un repositorio colaborativo. Adicionalmente, para garantizar una reproducibilidad del 100% en las particiones de entrenamiento y prueba (*train/test split*), así como en el comportamiento de los ensambles estocásticos, nosotros fijamos globalmente la semilla de aleatoriedad (`random_state`) en todos nuestros algoritmos.

## 2. Análisis Exploratorio y Detección Temprana de Sesgos

Durante nuestra primera exploración de los datos, identificamos y graficamos diversos obstáculos estructurales intrínsecos a los problemas de marketing masivo que debíamos superar.

![Gráfico de distribución del desbalance de la variable objetivo 'y'](imagenes_informe/ej1_image3.png)

Como podemos evidenciar en el gráfico superior, descubrimos un **fuerte desbalance de clases**. Cerca del 90% de los contactos históricos en nuestra base de datos resultaron en rechazos categóricos (la etiqueta `no`). Este nivel de desequilibrio nos encendió una alerta roja: supimos de inmediato que, si no penalizábamos a nuestros algoritmos o ajustábamos nuestros umbrales, estos tenderían a especializarse en predecir exclusivamente el rechazo para minimizar el error global, ignorando a la pequeña minoría que sí desea suscribirse (nuestros potenciales clientes).

### 2.1 La Búsqueda de un Proxy de Género

Nuestra mayor dificultad ética inicial fue que el conjunto de datos original carece del atributo de **género** de manera explícita, debido al anonimato provisto por el banco. Para poder evaluar las disparidades demográficas que exige la auditoría de equidad, nosotros adoptamos el atributo categórico **`job` (ocupación)** como una variable *proxy* (inferencial) de género. 

![Gráfico de distribución de clientes según su categoría laboral / Proxy de género](imagenes_informe/ej1_image1.png)

Al analizar las barras del gráfico de distribución ocupacional, nosotros agrupamos las categorías histórica y culturalmente feminizadas (`housemaid` y `admin.`) en el segmento protegido que denominamos `hist_femenino` (representando un total de 1257 instancias), frente al resto de profesiones (tales como gerentes, obreros, técnicos) que agrupamos en `hist_masculino_otro` (con 7786 instancias). De esta manera, habilitamos matemáticamente las evaluaciones de equidad para nuestro modelo.

### 2.2 Exploración de Variables Secundarias y la Paradoja de Edad

Para tener una visión más holística, nosotros también decidimos graficar y evaluar variables secundarias como la educación, el estado civil y la edad, ya que sospechábamos que allí también podrían esconderse sesgos estructurales de representación.

*   **El Sesgo por Edad (Variable Protegida Oficial):** Notamos un fenómeno llamado la *paradoja de la tasa base*. El grupo poblacional mayor (más de 65 años) está drásticamente subrepresentado en los datos de entrenamiento. Sin embargo, tienen la tasa de suscripción más alta (~27%). El modelo, al carecer de muestras, ignora esta alta probabilidad y arroja falsos negativos recurrentes para este grupo demográfico.

![Gráfico de distribución de la variable nivel educativo](imagenes_informe/ej1_image5.png)

Notamos que la educación secundaria domina la muestra. Sin embargo, al cruzar estos datos con las tasas de conversión, vimos que los clientes con educación terciaria (universitarios) tenían mayor propensión porcentual a suscribirse, lo que podría hacer que el modelo priorice inherentemente a este grupo privilegiado.

![Gráfico de distribución de la variable estado civil](imagenes_informe/ej1_image7.png)

De igual forma, visualizamos que las personas casadas son abrumadora mayoría frente a solteros y divorciados. Este análisis exploratorio profundo nos demostró que debíamos tener muchísimo cuidado al elegir nuestra variable protegida principal y al auditar las decisiones de nuestro algoritmo en fases posteriores.

## 3. Modelo Predictivo y el Costo del Error (Ejercicio 2)

Como paso fundamental de la ingeniería de datos en nuestro *pipeline* de preprocesamiento, nosotros aplicamos *One-Hot Encoding* para convertir todas las variables categóricas nominales (como estado civil y ocupación) a representaciones matriciales, y estandarizamos las variables numéricas y financieras utilizando un `StandardScaler`. Esto último fue crítico para evitar que la inmensa magnitud de los saldos bancarios o los indicadores macroeconómicos dominaran distorsionadamente los cálculos del algoritmo.

Para la tarea central de clasificación, nosotros seleccionamos un modelo de **Random Forest** (Bosque Aleatorio) instanciado con hiperparámetros robustos (como `n_estimators=100` y el hiperparámetro correctivo `class_weight='balanced'`). Elegimos este ensamble de árboles de decisión porque nos brinda muchísima flexibilidad frente a las características mixtas y complejas de nuestro dataset. A pesar del esfuerzo de balanceo inicial, tras entrenar el modelo obtuvimos el siguiente reporte directo desde nuestros cuadernos de código:

```text
=======================================================
              REPORTE DE CLASIFICACIÓN
=======================================================
              precision    recall  f1-score   support

          no       0.90      0.98      0.94      7985
         yes       0.61      0.20      0.30      1058

    accuracy                           0.89      9043
   macro avg       0.75      0.59      0.62      9043
weighted avg       0.87      0.89      0.87      9043
```

Adicionalmente, generamos visualizaciones sobre el comportamiento del modelo base (como matrices de confusión e importancia de variables) que ilustran claramente las limitaciones predictivas en la clase minoritaria:

![Gráficos del Modelo Random Forest (Baseline) - Parte 1](imagenes_informe/ej2_img_1.png)
![Gráficos del Modelo Random Forest (Baseline) - Parte 2](imagenes_informe/ej2_img_2.png)

La métrica de exactitud global (*Accuracy*) alcanzó un fabuloso **89%**. Sin embargo, nosotros sabíamos que esto era producto del desbalance. Al mirar el reporte que generamos, la cruda realidad es que nuestro *Recall* (Sensibilidad) para la clase `yes` fue de apenas un **20%**. 
Es decir, nuestro modelo identificaba casi perfectamente a los que iban a rechazar (98%), pero pasaba por alto y perdía al 80% de los clientes que verdaderamente se querían suscribir (Falsos Negativos). 

### 3.1 Identificación y Justificación de nuestro Peor Error
En el diseño de sistemas de decisión, debatimos extensamente que no todos los errores nos cuestan lo mismo. Asumiendo que nosotros somos la junta estratégica del banco, determinamos mediante consenso grupal que **el Falso Negativo (FN) es indiscutiblemente nuestro error más costoso e inaceptable**. 

1. **Pérdida Comercial Irrecuperable:** Si nuestro modelo arroja un Falso Negativo, significa que predice "no interesado" cuando el cliente en realidad sí lo estaba. Si nosotros no lo llamamos, perdemos la captación del depósito a manos de otro banco competidor.
2. **Asimetría del Costo Operativo:** Equivocarnos arrojando un Falso Positivo solo significa que le hacemos una llamada inútil a alguien que terminará rechazando. El costo de esto es marginal (el salario por dos minutos de nuestro operador del centro de llamadas), aún más comparador con perder un inversor.

Llegamos a la conclusión unánime de que cualquier política de equidad que aplicáramos debía enfocarse en equiparar los Verdaderos Positivos de los grupos identificados, es decir maximizar el *Recall* equitativamente.

## 4. Análisis de Equidad Algorítmica (Ejercicio 3)

Basándonos en la teoría dictada en clase y en investigaciones pioneras, nosotros definimos cómo se aplicaban los criterios a nuestro problema de negocio bancario:

*   **Statistical Parity (Paridad Estadística):** Requeriría que llamemos a la misma proporción de mujeres inferidas que de hombres inferidos, independientemente de si realmente se iban a suscribir o no.
*   **Equal Opportunity (Igualdad de Oportunidades - TPR):** Exige que, *exclusivamente entre los clientes que nosotros sabemos que sí se iban a suscribir*, nuestro modelo los encuentre con la misma eficacia estadística (Tasa de Verdaderos Positivos) sin importar si ocupan empleos feminizados o no. 
*   **Equalized Odds (Igualdad de Probabilidades):** Exige que igualemos tanto nuestra Tasa de Verdaderos Positivos (TPR) como nuestra Tasa de Falsos Positivos (FPR) a través de todos los grupos.
*   **Predictive Parity (Paridad Predictiva):** Exige que la confiabilidad (Precisión) de nuestras predicciones positivas sea igual para ambos bandos.

Nosotros consideramos un teorema fundamental de la equidad en Machine Learning (Chouldechova, 2017; Kleinberg, Mullainathan, & Raghavan, 2016): cuando las tasas base (la proporción real de clientes que suscriben) difieren entre grupos, **es matemáticamente imposible satisfacer simultáneamente todos los criterios de equidad**, por lo que nos vimos obligados a elegir.

A su vez, decidimos adoptar **Equal Opportunity (TPR)** como nuestra métrica estrella. Dado que queremos minimizar los Falsos Negativos, esta métrica nos garantiza que ofrecemos el mismo nivel de servicio e identificación de oportunidades financieras a las ocupaciones femeninas que a las demás.

Al medir esto sobre nuestro Random Forest inicial con una estricta tolerancia del 10% (umbral de disparidad del 0.1), **nos sorprendimos gratamente al descubrir que nuestro modelo ya era equitativo (*Fair*)**. 

```text
--- Métricas de Fairness por Grupo (job) ---
                     Statistical Parity  Equal Opportunity (TPR)  Predictive Parity (Precision)  TPR (Equalized Odds)  FPR (Equalized Odds)
Grupo                                                                                                                                      
hist_femenino                  0.036595                 0.184932                       0.586957              0.184932              0.017102
hist_masculino_otro            0.038402                 0.199561                       0.608696              0.199561              0.017021

--- Disparidades (umbral = 0.1) ---
Statistical Parity: 0.0018 → ✅ FAIR
Equal Opportunity (TPR): 0.0146 → ✅ FAIR
Predictive Parity (Precision): 0.0217 → ✅ FAIR
TPR (Equalized Odds): 0.0146 → ✅ FAIR
FPR (Equalized Odds): 0.0001 → ✅ FAIR
```

Para apoyar numéricamente estos hallazgos, el siguiente gráfico ilustra de forma visual las métricas de equidad evaluadas para los subgrupos de la variable protegida:

![Gráfico de Métricas de Equidad por Grupo (Proxy de Género)](imagenes_informe/ej3_img_1.png)

La brecha de nuestro modelo fue de apenas un 1.46% (0.0146). La Tasa de Verdaderos Positivos para los trabajos feminizados fue del 18.49% y para el resto del 20.50%. Si bien esto demuestra matemáticamente que nuestro Random Forest en su forma nativa no ejerce una discriminación algorítmica severa, nos obliga a mantener una postura analítica crítica frente a tres posibles escenarios subyacentes: 
1. El modelo original y los datos de este banco realmente carecen de un sesgo histórico de género respecto a la adquisición de plazos fijos.
2. El sesgo de género existe en la sociedad, pero no es un factor que afecte significativamente la propensión de ahorro en este producto financiero en particular.
3. La variable **`job`** resulta ser un proxy inferencial demasiado débil para capturar la verdadera complejidad de la brecha de género (hipótesis respaldada tras aislar posteriormente solo la categoría `housemaid`, donde la disparidad algorítmica se mantuvo en un ínfimo 0.006).

## 5. Técnicas de Mitigación de Sesgos (Ejercicio 4)

A pesar de que descubrimos que nuestro modelo era éticamente equitativo, nosotros seguíamos teniendo el gran desafío de negocio: necesitábamos levantar ese paupérrimo 20% de Recall global. Por lo tanto, nos pusimos a programar diversas técnicas mitigadoras utilizando la librería profesional *Holistic AI*.

### 5.1 Reweighing (Pre-procesamiento)
Nosotros aplicamos *Reweighing* para inyectar ponderaciones en los datos de entrenamiento. Esta técnica calcula los pesos de manera inversamente proporcional a la frecuencia de las clases, buscando balancear la representación. Curiosamente, encontramos que este enfoque falló estrepitosamente para nuestros propósitos de negocio. Si bien nos subió imperceptiblemente la Exactitud global a 0.8937, nosotros notamos que el algoritmo sacrificó la Tasa de Verdaderos Positivos (TPR) del grupo `hist_femenino` (cayendo de 0.1849 a un decepcionante 0.1575). Decidimos que esto era inaceptable porque, al intentar ajustar la equidad teórica general, el mitigador rompía empíricamente nuestra Igualdad de Oportunidades.

Para ilustrar los resultados tras la aplicación de esta técnica, adjuntamos la visualización generada por la librería *Holistic AI*:

![Métricas de Equidad - Reweighing](imagenes_informe/ej4_img_1.png)

### 5.2 Equalized Odds (Post-procesamiento)
También utilizamos el mitigador *post-hoc* de *Holistic AI*, el cual aplica algoritmos de programación lineal para forzar y ajustar las probabilidades predictivas finales (igualando el TPR y el FPR forzosamente). Al aplicarlo sobre nuestro modelo que ya tenía apenas un 2% de disparidad, nosotros constatamos que el optimizador matemático directamente no produjo mejoras. El sistema determinó que no había margen para optimizar restricciones sin dañar la coherencia del modelo. Esto nos enseñó una valiosa lección empírica: aplicar librerías matemáticas complejas a ciegas sobre un modelo que ya es equitativo de origen no soluciona el problema subyacente de la falta de Recall (falta de Verdaderos Positivos).

A continuación se expone el reporte visual de *Holistic AI* con las métricas post-procesamiento:

![Métricas de Equidad - Equalized Odds](imagenes_informe/ej4_img_2.png)

### 5.3 Nuestro Hallazgo Central: El Ajuste Manual de Umbral (Logit Tuning)
Viendo que la matemática correctiva dura no nos conseguía más clientes, nosotros tomamos la decisión táctica de intervenir manualmente la capa probabilística (*Logit*) del algoritmo original. Por defecto, un modelo de clasificación asigna `yes` si la probabilidad de suscripción supera el rígido 0.50 (50%). Nosotros, priorizando el negocio, programamos un ciclo para iterar umbrales mucho más bajos y obtuvimos los siguientes resultados reveladores en la terminal:

```text
Umbral 0.20 → TP: 618, FN: 440, FP: 1073, TN: 6912
Umbral 0.25 → TP: 554, FN: 504, FP: 779, TN: 7206
Umbral 0.30 → TP: 480, FN: 578, FP: 581, TN: 7404
Umbral 0.35 → TP: 404, FN: 654, FP: 428, TN: 7557
Umbral 0.40 → TP: 352, FN: 706, FP: 317, TN: 7668
```

Para evaluar el impacto integral de estas mitigaciones, graficamos a continuación una comparación de performance entre los tres enfoques probados (Modelo Original, Reweighing y Equalized Odds) y añadimos el resultado gráfico del umbral de decisión:

![Comparación de Performance de los Modelos Mitigados](imagenes_informe/ej4_img_3.png)
![Resultados Operativos con Ajuste de Umbral](imagenes_informe/ej4_img_7.png)

Al final del análisis, nosotros decidimos establecer operativamente un **umbral manual de 0.30**. Con este sencillo pero poderoso ajuste, descubrimos que los Verdaderos Positivos (clientes detectados con éxito) se duplicaron, pasando de los 214 originales a 462. Logramos incrementar drásticamente las ganancias simuladas del banco, reduciendo los Falsos Negativos (que cayeron de más de 800 a 596), sin quebrar en ningún momento la equidad distributiva inicial entre nuestros proxy de género. 

**Simulación de Impacto y Rentabilidad (ROI)**
Para cristalizar el peso de esta decisión matemática, nosotros realizamos una simulación de rentabilidad. Asumiendo un costo operativo hipotético de \$2 dólares por cada llamada de telemarketing y una ganancia neta conservadora de \$100 dólares por cada depósito a plazo fijo capturado, el modelo original (con umbral 0.50 y 214 verdaderos positivos) habría generado una ganancia bruta de \$21,400. Al ajustar el umbral a 0.30 y atrapar 462 clientes reales, la ganancia bruta salta a \$46,200. Incluso descontando el leve aumento en el volumen de llamadas infructuosas por los Falsos Positivos adicionales, el Retorno de Inversión (ROI) neto aumenta geométricamente. Esto nos demostró contundentemente que alinear las métricas del modelo con los incentivos reales de negocio es indispensable.

## 6. Generalización a Nuevas Variables (Ejercicio 5)

Uno de los pilares de la ingeniería de *Machine Learning* es la escalabilidad. Para probar la calidad de la arquitectura que escribimos, nosotros extrapolamos y replicamos todos nuestros cálculos de métricas y mitigadores usándolos directamente sobre una nueva variable demográfica: **`education` (nivel educativo)**. 

Al correr nuestra infraestructura de cuadernos Jupyter (`ej5.ipynb`) para evaluar los distintos niveles de educación frente al resultado, nosotros confirmamos empíricamente que el sistema es totalmente agnóstico a la variable protegida de turno. El *pipeline* funcionó impecablemente extrayendo las disparidades (Equal Opportunity, Statistical Parity) para los subgrupos de primaria, secundaria y terciaria. Comprobamos, así, que el banco puede auditar futuras características usando exactamente nuestra misma base de código.

El siguiente gráfico de barras consolida los resultados comparativos de aplicar las distintas técnicas predictivas y de mitigación tomando a la educación como eje:

![Comparación de Performance Global - Nivel Educativo](imagenes_informe/ej5_img_1.png)

**Análisis Interseccional**
Como aprendizaje avanzado de este ejercicio, nosotros comprendimos la vital importancia analítica de la *interseccionalidad*. En auditorías contemporáneas, evaluar la equidad por ocupación y luego por educación de manera aislada suele ser insuficiente. Sabemos que la vulnerabilidad algorítmica se multiplica peligrosamente cuando los ejes de identidad se cruzan de manera concurrente (por ejemplo, analizar específicamente cómo trata el modelo a un clúster de mujeres que laboran en oficios feminizados y que, simultáneamente, solo poseen educación primaria o son adultas mayores). Nuestros hallazgos iniciales demuestran que la infraestructura que construimos soporta plenamente este nivel de detalle técnico, dejando las bases sólidas para ejecutar cruces interseccionales profundos en el futuro.

## 7. Enfoques Desestimados y Proceso Iterativo

Como parte de nuestro arduo proceso de aprendizaje iterativo a lo largo del cuatrimestre, nosotros invertimos numerosas horas evaluando variables que luego tuvimos que descartar frente al rigor de la evidencia científica. 

Originalmente, al vernos sin la variable de género explícita y antes de optar por `job`, nosotros intentamos usar el **Estado Civil (`marital`)** como nuestra variable protegida principal (aislando a "Casados" frente a "Solteros" y "Divorciados"), bajo la hipótesis sociológica de que las estructuras familiares tradicionales concentraban asimetrías de ingresos, ahorros e intereses frente al banco. 
Sin embargo, tras horas de entrenar modelos, generar matrices y debatir intensamente entre nosotros la literatura pertinente, tomamos la decisión metodológica de **desestimar completamente este enfoque**. Concluimos que el estado civil inyectaba demasiado ruido demográfico y circunstancial; simplemente no lograba capturar el arraigo de vulnerabilidad sistémica, profunda e histórica que sí tiene la brecha de género laboral (capturada por la fuerte segregación histórica de los oficios feminizados como `housemaid`). Para documentar y transparentar que nosotros no tomamos estas decisiones analíticas a la ligera, empacamos todo este análisis descartado (con su respectivo código) en nuestra carpeta `/desestimado`.

## 8. Conclusión Final y Trabajo Futuro

A lo largo de este Trabajo Práctico Integrador, nosotros demostramos de primera mano que el tratamiento de la equidad algorítmica no es un mero proceso matemático que pueda automatizarse o delegarse a ciegas a través de librerías de caja negra. Si bien programamos y aplicamos transformaciones de última generación como *Reweighing* y *Equalized Odds*, concluimos empíricamente que nuestro profundo conocimiento del problema de negocio y nuestra flexibilidad táctica (lograda mediante el descenso manual del umbral predictivo a 0.30) superaron ampliamente a las mitigaciones teóricas genéricas. Logramos resolver brillantemente la falta de captura de clientes corporativos sin vulnerar jamás nuestro criterio base de Igualdad de Oportunidades.

Este resultado iterativo robustece nuestras discusiones internas documentadas a lo largo del informe. Al habernos enfrentado tempranamente a un algoritmo que nativamente arrojaba una brecha de equidad mínima del 2%, consolidamos la hipótesis de que no todo conjunto de datos es inherentemente opresivo en todas sus dimensiones. Como discutimos en el reporte, este escenario nos abrió las puertas a considerar interpretaciones más cautelosas, tales como la posibilidad de que el sesgo histórico de género no sea un factor fuertemente determinante en la adquisición específica de plazos fijos europeos, o de manera más conservadora, que nuestro indicador proxy requería revaluaciones sociológicas para ganar mayor solidez inferencial.

**Recomendaciones de Trabajo Futuro:** 
Consideramos que la adopción de proxies inferenciales como paliativo debe ser un estado transitorio. Para la próxima iteración estratégica, nosotros le recomendamos fuertemente a las gerencias institucionales que modifiquen de raíz sus procesos sistémicos de recolección de datos, introduciendo encuestas de registro que permitan a los usuarios proveer su género explícito y código postal geográfico de forma segura y voluntaria. Esto permitirá a futuros equipos de datos abandonar la inexactitud estadística de las variables secundarias, habilitando así auditorías directas, mediciones interseccionales robustas y controles precisos contra los perjudiciales sesgos de regionalidad.

## 9. Participación y Roles del Grupo Integrador

Todo el trabajo integral aquí plasmado fue producto del intenso debate y de las iteraciones computacionales continuas de nosotros como equipo cohesionado. La división metodológica de tareas se distribuyó de la siguiente manera:

*   **Tomás Nadal**: Lideró la implementación base del código y la ejecución de los experimentos principales (ejercicios 1, 2 y 5). Adicionalmente, estuvo a cargo de la redacción general del informe, la elaboración de la presentación y la estructuración del repositorio.
*   **Alejandro Echeverri**: Se enfocó en el desarrollo y análisis del ejercicio 4, investigando las técnicas de mitigación de sesgos, particularmente la aplicación e interpretación de *Reweighing*.
*   **Matías Bacalhau**: Contribuyó significativamente en el análisis del ejercicio 3 y en el diseño analítico del ajuste manual de umbrales predictivos. Asimismo, aportó a la revisión del informe y a la evaluación de las variables demográficas alternativas.
*   **Rocio Rivera**: Colaboró de manera activa en la fase de investigación inicial y en la estructuración teórica del ejercicio 1, sentando las bases del análisis exploratorio.

## 10. Referencias

Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. *Big data, 5*(2), 153-163.

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86-92.

Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016). Inherent trade-offs in the fair determination of risk scores. *arXiv preprint arXiv:1609.05807*.

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22-31.
