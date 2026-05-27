---
marp: true
theme: custom
paginate: true
math: mathjax
header: "![w:150](../assets/Logo_UNSAM.png)"
style: |
  section {
    background-color: #E1F5FE;
    color: #002244;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    padding: 60px 80px 80px 80px;
    font-size: 25.5px; /* Aumento leve para mejor legibilidad */
  }
  header {
    position: absolute;
    top: 40px;
    left: auto;
    right: 50px;
    width: auto;
    text-align: right;
    z-index: 10;
  }
  header img {
    box-shadow: none !important;
    background: transparent !important;
    border-radius: 0 !important;
  }
  
  /* Ocultar page control a los 4s sin hover */
  .bespoke-marp-osc {
    opacity: 0 !important;
    transition: opacity 0.5s ease-in-out 4s !important;
    pointer-events: none; /* Evita que el OSC invisible intercepte clics */
  }
  .bespoke-marp-parent:hover .bespoke-marp-osc {
    opacity: 1 !important;
    transition-delay: 0s !important;
    pointer-events: auto;
  }
  
  h1, h2, h3 {
    color: #0047AB;
    margin-bottom: 0.2em;
  }
  h1 { font-size: 2em; font-weight: bold; }
  h2 { font-size: 1.4em; border-bottom: 2px solid #0047AB; padding-bottom: 5px; margin-top: 0; }
  p, li { font-size: 1em; line-height: 1.35; }
  ul { padding-left: 1.2em; }
  strong { color: #003366; }
  
  /* Glassmorphism Cards */
  .card {
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 34, 68, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.5);
  }
  
  .columns { display: flex; gap: 2em; align-items: stretch; }
  .column { flex: 1; }
  
  img {
    max-width: 100%;
    max-height: 300px;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 4px 10px rgba(0,34,68,0.1);
  }
  
  .center { text-align: center; }
  .highlight-box {
    background: rgba(0, 71, 171, 0.05);
    border-left: 4px solid #0047AB;
    padding: 15px 20px;
    margin-top: 15px;
    border-radius: 4px;
  }

  /* Portada Styling */
  section.portada {
    background: linear-gradient(135deg, #001f3f 0%, #0047AB 100%);
    color: white;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }
  section.portada h1 { color: white; border: none; font-size: 3em; margin-bottom: 0.2em; }
  section.portada h2 { color: #87CEFA; border: none; font-size: 1.5em; font-weight: normal; }
  section.portada .details { margin-top: 2em; font-size: 1.2em; color: #E0FFFF; }
  section.portada strong { color: white; }

  /* Preguntas Styling */
  section.preguntas {
    background-color: #0047AB;
    background-image: none;
    color: white;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }
  section.preguntas h1 { color: white; border: none; font-size: 3.5em; margin-bottom: 0.1em; }
  section.preguntas h2 { color: #87CEFA; border: none; font-size: 1.8em; font-weight: normal; margin-top: 0; }

---

<!-- _class: portada -->
<!-- _header: "" -->
<!-- _paginate: false -->

<h1>Equidad en Aprendizaje Automático</h1>
<h2>Análisis de Sesgos y Mitigación en Bank Marketing</h2>

<div class="details">
  <p>Trabajo Práctico Integrador</p>
  <p><strong>Integrantes:</strong> Tomás Nadal, Alejandro Echeverri, Matías Bacalhau, Rocío Rivera</p>
  <p>1er Cuatrimestre 2026 | Universidad Nacional de San Martín (UNSAM)</p>
</div>

---

## 1. El Dataset: Bank Marketing

<div class="card">

**Motivación**
Predecir si un cliente **suscribirá a un plazo fijo** para optimizar campañas de marketing telefónico.

**Naturaleza de los Datos**
- **45,211 instancias** (llamadas telefónicas).
- Variables sociodemográficas y macroeconómicas (Euribor, tasas de empleo).
- **Preprocesamiento Clave:** Exclusión de la variable `duration` para evitar el riesgo de *Data Leakage*.

**Variable Protegida (Foco de Equidad)**
- **`job` (Ocupación) como proxy de género** (ej. roles históricamente feminizados como "housemaid" vs masculinizados como "blue-collar").

</div>

---

## 2. Datasheets for Datasets (Contexto)

<div class="columns">
<div class="column">
<div class="card" style="height: 100%;">

**Origen y Validación**
- Obtenido del UCI Machine Learning Repository.
- Los datos no provienen de encuestas, sino de **registros administrativos reales** de un banco portugués (2008-2010).
- La validación del éxito de la campaña fue determinística (cruce con bases transaccionales).

</div>
</div>
<div class="column">
<div class="card" style="height: 100%;">

**Implicancias Éticas**
- La recolección directa hereda los sesgos de perfilamiento del banco. Al decidir a quién ofrecer instrumentos de ahorro (cuyos rendimientos se pagan con capital del banco), se discrimina a grupos vulnerables.
- Los "descartados" por el modelo no volverán a ser llamados, perpetuando su exclusión de herramientas de ahorro e inclusión financiera.

</div>
</div>
</div>

---

## 3. Sesgos Potenciales Iniciales

Existen fuertes desbalances estructurales en los datos:

<div class="columns">
<div class="column">
<div class="card" style="height: 100%;">

- **Etiquetas**: ~90% de los contactos dicen "no".
- **Educación**: El nivel secundario es mayoría absoluta.
- **Demografía**: Grupos como solteros o jubilados están fuertemente subrepresentados frente a adultos casados.

</div>
</div>
<div class="column">
<div class="card" style="height: 100%;">
<div class="highlight-box">
<strong>Riesgo Identificado:</strong><br>
El modelo puede priorizar inadvertidamente a los grupos mayoritarios (aprendiendo sus patrones), siendo ineficaz para captar a las minorías.
</div>
</div>
</div>
</div>

---

## 4. Clasificación Base y Rendimiento Global

Antes de analizar la equidad, entrenamos un modelo **Random Forest Classifier** estándar.

<div class="card">

**Métricas del Modelo Base (Global)**
- Es excelente prediciendo el "No" (alta Accuracy global empujada por la clase mayoritaria).
- Sufre para predecir correctamente el "Sí" en casos ambiguos.

**El Costo del Error: Falsos Negativos**
- **Falso Negativo (FN)**: Predecir que el cliente *no* aceptará, cuando en realidad *sí* lo haría.
- **Impacto**: Pérdida directa de un cliente potencial (costo de oportunidad real).
- *Un Falso Positivo solo cuesta el tiempo operativo de hacer una llamada inútil.*

</div>

---

## 5. Evaluación Inicial (`job` como proxy)

Evaluación del modelo base segmentando por ocupación (proxy de género).

<div class="columns">
<div class="column">
<div class="card">

  - **Hallazgo clave:** La TPR del grupo feminizado fue **18.49%** vs **19.96%** del resto — una diferencia de solo **1.46%**.
  - El problema real no era discriminación: **ambos grupos tenían un Recall inferior al 20%**. El algoritmo era ineficaz para captar inversores reales en *todos* los grupos por igual.

</div>
</div>
<div class="column">
  <img src="../assets/ej2_img1.png" alt="Evaluación Inicial 1">
</div>
</div>

---

## 6. Conceptos de Equidad (Fairness)

<div class="columns">
<div class="column">
<div class="card" style="font-size: 0.9em;">

**Statistical Parity**
Misma proporción de predicciones positivas para todos los grupos. *Irreal si la disposición real a suscribirse varía.*

**Equalized Odds**
Mismas tasas de verdaderos positivos (TPR) y falsos positivos (FPR) en todos los grupos. *Muy estricto.*

**Predictive Parity**
Mismo Precision (Valor Predictivo Positivo) entre grupos.

</div>
</div>
<div class="column">
<div class="card" style="background-color: #e6f0fa; border-color: #0047AB;">

**Criterio Elegido: Equal Opportunity**
(Igualdad de Oportunidades)

Como el **Falso Negativo** es el error más costoso, exigimos que la **Tasa de Verdaderos Positivos (TPR) sea igual para todos**. 

*Si un cliente realmente va a suscribirse, el modelo debe detectarlo sin importar su ocupación.*

</div>
</div>
</div>

---

## 7. Análisis Cuantitativo de Sesgo (Pre-mitigación)

Evaluación matemática de la disparidad antes de intervenir.

<div class="card" style="margin-bottom: 20px;">

Utilizando el módulo de la diferencia como medida de disparidad:
- **Hallazgo inesperado:** El modelo original (Baseline) resulta ser muy equitativo, cumpliendo con holgura **todas** las métricas con un umbral de tolerancia del 10%:
  - Statistical Parity: **0.0018** ✅
  - Equal Opportunity (TPR): **0.0146** ✅
  - Predictive Parity: **0.0217** ✅

</div>

<div class="center">
  <img src="../assets/ej3_img1.png" alt="Criterios Cuantitativos" style="max-height: 340px; width: auto;">
</div>

---

## 8. Mitigación: Reweighing (Pre-procesamiento)

**Técnica:** Asigna "pesos" a los datos de entrenamiento para balancear la importancia empírica de grupos desfavorecidos.

<div class="columns">
<div class="column">
<div class="card">

- **No altera los datos**, solo modifica su distribución de peso interno.
- Evita que el modelo penalice a grupos minoritarios durante el aprendizaje.
- *Resultado Empírico*: Como el modelo base ya era equitativo, aplicar este pre-procesamiento generó fluctuaciones menores, degradando levemente algunas métricas en el set de prueba.

</div>
</div>
<div class="column">
  <img src="../assets/ej4_img1.png" alt="Resultados Reweighing">
</div>
</div>

---

## 9. Enfoque Alternativo: Ajuste Manual de Umbral

Antes de recurrir a un post-procesamiento complejo, evaluamos una intervención directa para priorizar la captación comercial (reducir Falsos Negativos).

<div class="card" style="margin-bottom: 20px;">

- **Técnica:** Analizamos las métricas internas del Random Forest y bajamos el umbral de decisión para clasificar como "Sí" de `0.50` a **`0.30`**.
  - **Resultado:** Se **duplicaron** los Verdaderos Positivos (**214 → 484** clientes captados), reduciendo los Falsos Negativos de más de 800 a 574, **sin afectar la equidad**.
  - **Conclusión:** A veces un ajuste empírico simple y guiado por el negocio es más transparente y efectivo que forzar restricciones matemáticas a ciegas en todo el pipeline.

</div>

---

## 10. Mitigación: Equalized Odds (Post-procesamiento)

**Técnica:** Modifica los umbrales de decisión del modelo *después* de que fue entrenado, forzando matemáticamente la igualdad de TPR.

<div class="columns">
<div class="column">
<div class="card">

- Intervención directa y potente sobre las predicciones emitidas.
- **¿Qué ocurrió aquí?**: Aplicar este mitigador estricto sobre un modelo que *ya era justo* terminó **empeorando** la equidad (aumentó la diferencia de TPR).
- *Conclusión*: Forzar la igualdad matemática alterando (haciendo "flip" a) predicciones de un modelo balanceado introduce ruido y degrada los resultados.

</div>
</div>
<div class="column">
  <img src="../assets/ej4_img2.png" alt="Resultados Equalized Odds">
</div>
</div>

---

## 11. Resumen Numérico Comparativo

<div class="card">

Al aplicar las técnicas utilizando la librería **Holistic AI**, observamos los siguientes resultados en las métricas de disparidad:

| Modelo | TPR Difference | Disparate Impact | Evaluación de Equidad |
|:---|:---:|:---:|:---:|
| **Random Forest Base** | **0.0146** | **1.049** | ✅ **Muy Justo** |
| **RF + Reweighing** | 0.0179 | 0.972 | ✅ Justo (leve ruido) |
| **RF + Equalized Odds** | 0.0673 | 1.254 | ❌ **Empeora la equidad** |

*(Nota: El baseline resultó ser el mejor modelo. Aplicar mitigadores por defecto puede ser contraproducente).*

</div>

---

## 12. Extrapolación: Nivel Educativo (Ej 5)

Para validar la robustez de la metodología, replicamos el análisis de equidad utilizando la variable **`education`** (Universitarios vs No Universitarios).

<div class="card" style="margin-bottom: 20px;">

- **El Umbral 0.30 como Base:** Adoptamos el umbral ajustado empíricamente como nuestro nuevo modelo base comercial, ya que demostró mejorar los Verdaderos Positivos.
- **Prueba de Mitigadores:** Aplicamos *Reweighing* y un *Threshold Adjustment* iterativo específico por grupo educativo.
- **Conclusión Consistente:** Confirmamos la misma tendencia que con la variable `job`. Nuestro modelo base ya presentaba disparidades ínfimas, por lo que intervenir forzosamente por subgrupos no aportó ganancias significativas de equidad.

</div>

---

## 13. Lecciones sobre la Mitigación

Nuestro análisis nos dejó una enseñanza contra-intuitiva pero fundamental en la práctica:

<div class="columns">
<div class="column">
<div class="card">

- **No asumas el sesgo, mídelo**: El modelo base resultó ser altamente equitativo de forma natural.
- **Más mitigación no siempre es mejor**: Intervenir agresivamente con algoritmos de *post-processing* terminó añadiendo ruido y empeorando las métricas de sesgo.
- **Pruebas de estrés**: Un umbral estándar (0.1) puede ser muy permisivo; es clave analizar la robustez con umbrales más estrictos (0.01).

</div>
</div>
<div class="column">
  <img src="../assets/ej5_img1.png" alt="Trade-off Plot">
</div>
</div>

---

## 14. Reflexión en el Mundo Real

**Más allá de los números**
Un modelo sesgado que deniega sistemáticamente oportunidades genera **ciclos de retroalimentación negativos**:
Grupos marginados no son contactados $\rightarrow$ no generan historial $\rightarrow$ el modelo futuro aprende que "no son propensos al éxito".

**Recomendación Operativa para el Banco:**
Modificar procesos de recolección para obtener el género explícito voluntariamente, abandonando la inexactitud de los proxies.

<div class="highlight-box">
<strong>Conclusión Final de la Materia:</strong><br>
Los algoritmos y los datos no son neutros. Mitigar sesgos no es solo un ajuste técnico; es un imperativo ético para no amplificar ni automatizar inequidades estructurales a gran escala.
</div>

---

<!-- _class: preguntas -->
<!-- _header: "" -->
<!-- _paginate: false -->

<h1>¡Muchas gracias!</h1>
<h2>¿Tienen alguna pregunta?</h2>
