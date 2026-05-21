# TP Equidad en Aprendizaje Automático

## Contexto y Dataset

**Dataset**: Bank Marketing (UCI) — 41.188 registros de campañas de telemarketing de un banco portugués (2008–2010).  
**Objetivo de Negocio**: Incrementar la tasa de suscripción a depósitos a plazo fijo, prediciendo si un cliente suscribirá o no (`y`: yes/no).  
**Variable objetivo**: Clase muy desbalanceada (~89% "no", ~11% "yes").

---

## Variables Protegidas y Proxies Analizados

La consigna establece que las variables protegidas oficiales para este dominio bancario son el **estado civil** y la **edad**, sumado a un enfoque particular en el **género**. Debido a la composición del dataset, el análisis se estructuró de la siguiente forma:

| Variable / Proxy | Nombre en Dataset | Justificación y Uso |
|-------|----------|---------------|
| **Género** (Proxy) | `job` | El dataset no incluye género directo. Se utilizó la ocupación como proxy, agrupando trabajos históricamente femeninos (`housemaid`, `admin.`) versus el resto. |
| **Estado civil / Edad** | `marital` | Variable protegida oficial asignada por la consigna. Se utiliza porque el estado civil correlaciona fuertemente con la etapa de la vida (solteros≈jóvenes, casados≈mediana edad, divorciados≈mayores). |

> **Nota Crítica sobre Data Leakage**: La variable `duration` (duración de la llamada) fue **excluida de todos los modelos predictivos**. Como advierten los autores del dataset, esta variable solo se conoce *después* de que la llamada ha finalizado y determina directamente el éxito de la misma. Incluirla crearía un modelo sesgado e irrealizable en un entorno de producción.

---

## Resumen de la Implementación por Ejercicio

### Ejercicio 1 — Análisis Exploratorio y Marco Conceptual (`ej1.ipynb`)

Se realizó una investigación profunda sobre el origen y propósito del dataset siguiendo la metodología *Datasheets for Datasets*.

**Hallazgos principales:**

- **Motivación**: El dataset fue creado para demostrar la viabilidad del Data Mining aplicado a campañas de telemarketing bancario, orientado a la selección eficiente de clientes a contactar.
- **Creadores**: Investigadores portugueses (Moro, Cortez, Rita), con datos de una institución bancaria bajo confidencialidad.
- **Proceso de recolección**: Híbrido entre observación directa (registros del call center) y datos derivados (indicadores macroeconómicos externos).
- **Sesgos identificados**:
  - **Desbalance de clases severo**: casi el 90% de los datos son "no", lo que puede llevar a un modelo que simplemente prediga siempre negativamente.
  - **Sub-representación de solteros y divorciados** frente a casados (estado civil).
  - **Sub-representación de grupos educativos extremos** (primaria y sin educación).
  - **Grupos etarios desiguales**: estudiantes y jubilados están sub-representados frente a trabajadores activos.
- **Variable proxy de género**: Se identificó `job` como proxy, dado que ocupaciones como `housemaid` y `admin.` tienen composición históricamente femenina.

---

### Ejercicio 2 — Modelo Base y Evaluación del Error (`ej2.ipynb`)

Se entrenó un modelo clásico de Machine Learning como punto de partida (baseline).

**Modelo Elegido**: Random Forest Classifier (100 estimadores), split estratificado 80/20 para mitigar el desbalance de clases.

**Resultados del modelo baseline:**

| Clase | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| no    | 0.92      | 0.97   | 0.94     |
| yes   | 0.60      | 0.30   | 0.40     |
| **Accuracy global** | | | **0.90** |

**Interpretación**: El Accuracy del 90% es engañoso — refleja principalmente la capacidad del modelo para predecir correctamente los "no". El Recall de 0.30 para la clase positiva ("yes") indica que el modelo falla en detectar a 7 de cada 10 clientes que realmente se suscribirían.

**Determinación del Error Crítico**: Desde la perspectiva comercial del Banco, se determinó que **el Falso Negativo (FN) es el error más perjudicial**. Un FN implica no contactar a alguien que sí se hubiera suscrito (pérdida directa de cliente y depósito), mientras que un Falso Positivo (FP) solo representa el costo operativo marginal de realizar una llamada telefónica adicional.

---

### Ejercicio 3 — Auditoría de Equidad (Fairness) (`ej3.ipynb`)

Se auditó cómo el modelo base trata a los distintos grupos demográficos utilizando 4 criterios formales, con un **umbral de tolerancia de disparidad del 10% (0.1)**.

**Definición de criterios en contexto:**

1. **Statistical Parity**: El modelo debe predecir suscripción en igual proporción para todos los grupos, independientemente de si realmente se suscriben.
2. **Equal Opportunity (TPR igual)**: Entre quienes sí iban a suscribirse, el modelo debe detectarlos con la misma eficacia en todos los grupos (mismo Recall/TPR).
3. **Equalized Odds**: Requiere igualar tanto TPR como FPR entre grupos (más restrictivo que Equal Opportunity).
4. **Predictive Parity (Precision igual)**: Cuando el modelo predice "suscripción", la confianza de esa predicción debe ser similar para todos los grupos.

> **Nota teórica**: Cuando las tasas base difieren entre grupos (como ocurre en este dataset), es **matemáticamente imposible** satisfacer simultáneamente Statistical Parity, Equal Opportunity y Predictive Parity (Chouldechova, 2017).

**Resultados para `job` (proxy género):**

| Métrica | hist_femenino | hist_masculino_otro | Disparidad | Fair? |
|---------|---------------|---------------------|-----------|-------|
| Statistical Parity | 0.065 | 0.053 | 0.012 | ✅ |
| Equal Opportunity (TPR) | 0.315 | 0.296 | 0.019 | ✅ |
| Predictive Parity (Prec.) | 0.648 | 0.583 | 0.065 | ✅ |
| FPR (Equalized Odds) | 0.026 | 0.025 | 0.002 | ✅ |

**Resultados para `marital` (estado civil):** También cumplió todos los criterios bajo el umbral del 10%.

**Criterio seleccionado**: **Equal Opportunity**. Priorizamos asegurar que el modelo detecte a los verdaderos suscriptores potenciales con la misma eficacia en todos los grupos (mismo TPR/Recall). Esto es coherente con la conclusión del Ej2: el error más costoso es el FN, y no queremos que el modelo sea bueno detectando un grupo e ineficiente con otro.

---

### Ejercicio 4 — Mitigación de Sesgos (`ej4.ipynb`)

Para mejorar proactivamente la equidad e intentar cerrar las brechas observadas en el TPR, se aplicaron y evaluaron dos estrategias de mitigación.

**Técnicas aplicadas:**

1. **Reweighting (Pre-procesamiento)**: Se reasignaron pesos a las instancias de entrenamiento de forma inversamente proporcional a la frecuencia de su combinación grupo-clase:
   $$W(g, y) = \frac{N}{n_{grupos} \times n_{clases} \times N(g, y)}$$
   Esto fuerza al algoritmo a "prestar más atención" a grupos minoritarios (ej. mujeres suscriptoras).

2. **Ajuste de Umbrales por Grupo (Post-procesamiento)**: En lugar de usar un umbral de decisión fijo del 50%, se recalibraron los umbrales de probabilidad de forma independiente para cada grupo protegido, con el objetivo matemático de igualar sus tasas de Verdaderos Positivos.

**Resultados comparativos (job — proxy género):**

| Modelo | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|-----|
| Baseline | 0.899 | 0.603 | 0.302 | 0.403 |
| Reweighting | 0.897 | 0.593 | 0.283 | 0.383 |
| Ajuste Umbral | 0.897 | 0.583 | 0.316 | 0.410 |

**Disparidad de Equal Opportunity (TPR) por grupo:**

| Modelo | Disparidad TPR (job) | Fair? |
|--------|----------------------|-------|
| Baseline | 0.0193 | ✅ |
| Reweighting | 0.0430 | ✅ |
| Ajuste Umbral | 0.0035 | ✅ |

El **Ajuste de Umbrales** logró la menor disparidad de TPR (0.35%), a costa de una ligera caída en Accuracy y Precision.

#### Versión alternativa con Holistic AI (`ej4_holisticAI.ipynb`)

Este notebook replica el mismo análisis del `ej4.ipynb`, reemplazando las implementaciones manuales por la librería **Holistic AI**:

| Técnica | En ej4.ipynb | En ej4_holisticAI.ipynb |
|---------|--------------|--------------------------|
| Reweighting | Función manual con bucle | `Reweighing().fit().transform()` |
| Post-procesamiento | Grid search de umbrales | `EqualizedOdds` (PL óptimo) |
| Métricas fairness | Cálculo manual | `classification_bias_metrics()` |

```bash
# Para instalar Holistic AI:
pip install holisticai[bias]
```

---

### Ejercicio 5 — Comparación y Reflexión Final (`ej5.ipynb`)

Se confrontaron las métricas de rendimiento y equidad del modelo base contra los modelos mitigados.

**Trade-off Equidad vs. Performance:**

Ambas técnicas de mitigación lograron disminuir aún más la disparidad de *Equal Opportunity*. Como es esperable:
- El **Ajuste de Umbrales** logró la mejor reducción de disparidad de TPR entre grupos.
- El **Reweighting** presentó resultados mixtos: en algunos grupos mejoró la equidad pero a mayor costo en performance.

**Conclusiones principales:**

1. **La leve caída en performance global es justificable**: Capturar más clientes potenciales de grupos que el modelo original ignoraba incrementa las oportunidades de suscripción totales.

2. **El banco opera más éticamente**: Garantizar que todos los grupos demográficos tengan acceso igualitario a los productos financieros protege al banco de riesgos legales y daños reputacionales por discriminación algorítmica.

3. **Equal Opportunity es el criterio más relevante en este contexto**: Dada la asimetría de costos de errores (FN >> FP), garantizar que el modelo no sea sistemáticamente peor en detectar suscriptores de un grupo demográfico específico es prioritario.

4. **Los criterios de fairness son incompatibles entre sí** cuando las tasas base difieren entre grupos, lo que nos obliga a priorizar un criterio según el contexto del problema (Chouldechova, 2017).

---

## Estructura del Repositorio

```text
TP_equidades/
├── data/
│   ├── full.csv               # Dataset Bank Marketing (UCI)
│   ├── datasheet.md           # Ficha técnica del dataset
│   └── consigna.pdf           # Consigna original del trabajo práctico
├── ej1.ipynb                  # Análisis exploratorio y marco conceptual
├── ej2.ipynb                  # Modelo base + evaluación de costo de errores
├── ej3.ipynb                  # Auditoría de equidad (Fairness metrics)
├── ej4.ipynb                  # Mitigación de sesgos (implementación manual)
├── ej4_holisticAI.ipynb       # Mitigación de sesgos con librería Holistic AI
├── ej5.ipynb                  # Comparación de métricas y reflexión final
├── presentacion.tex           # Presentación en LaTeX (Beamer)
└── README.md                  # Este documento (Overview del TP)
```

---

## Requisitos Técnicos e Instalación

Para replicar y ejecutar los notebooks se recomienda un entorno con Python 3.x y las siguientes librerías:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
# Solo para ej4_holisticAI.ipynb:
pip install holisticai[bias]
```

**Librerías principales:**
- `pandas` y `numpy` — manejo y manipulación de datos tabulares.
- `matplotlib` y `seaborn` — visualizaciones.
- `scikit-learn` — preprocesamiento, modelado y métricas de rendimiento y equidad.
- `holisticai` (opcional) — técnicas de mitigación de sesgo y métricas de fairness estandarizadas.

---

## Referencias

- Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing*. Decision Support Systems, 62, 22-31.
- Gebru, T., et al. (2018). *Datasheets for Datasets*.
- Chouldechova, A. (2017). *Fair prediction with disparate impact: A study of bias in recidivism prediction instruments*. Big Data, 5(2), 153–163.
