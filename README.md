# TP Equidad en Aprendizaje Automático

## Dataset
**Bank Marketing** (UCI) — 41.188 registros de campañas de telemarketing de un banco portugués.  
**Objetivo**: predecir si un cliente suscribirá un depósito a plazo fijo (`y`: yes/no).

## Variables protegidas y proxies

El dataset **no contiene variable de género directa**. En el Ejercicio 1 se identificaron dos proxies:

| Proxy | Variable | Justificación |
|-------|----------|---------------|
| **Género** | `job` | Trabajos como `housemaid` y `admin.` fueron históricamente femeninos |
| **Edad** | `marital` | El estado civil correlaciona con la etapa de vida (single→joven, married→mediana edad, divorced→mayor) |

> **Nota**: la variable `duration` fue excluida de todos los modelos por representar *data leakage* (solo se conoce después de la llamada).

---

## Resumen de ejercicios

### Ejercicio 1 — Análisis exploratorio y marco conceptual
**Consigna**: Explorar el dataset, identificar variables protegidas, proxies de género y posibles sesgos.

**Resolución**: Se identificaron los proxies `job` (género) y `marital` (edad), se documentaron desbalances de clases (~90% "no") y sub-representación en ciertos grupos demográficos. Se justificó la exclusión de `duration`.

---

### Ejercicio 2 — Modelo base y evaluación
**Consigna**: Entrenar un modelo clásico, evaluar con métricas estándar y determinar qué tipo de error es peor.

**Modelo**: Random Forest (100 estimadores, random_state=42), split estratificado 80/20.

**Conclusión sobre errores**: El **Falso Negativo (FN) es el error más crítico** porque representa un suscriptor potencial que el modelo ignora, causando pérdida directa de oportunidad comercial. El costo de un FP (una llamada extra) es marginal comparado con perder un depósito.

---

### Ejercicio 3 — Análisis de Fairness
**Consigna**: Describir criterios de fairness en contexto, evaluar cuantitativamente si el modelo es fair (umbral de disparidad = 0.1), y elegir el criterio más relevante.

**Criterio elegido**: **Equal Opportunity** — coherente con que el FN es el error más costoso (Ej2). Queremos que el modelo detecte suscriptores potenciales con la misma eficacia sin importar el grupo.

---

### Ejercicio 4 — Mitigación de sesgos
**Consigna**: Aplicar al menos 2 técnicas de mitigación, evaluar performance y fairness post-mitigación.

**Técnicas aplicadas**:
1. **Reweighting** (pre-processing): pesos inversamente proporcionales a frecuencia de (grupo, clase)
2. **Ajuste de umbral por grupo** (post-processing): umbrales distintos por grupo para igualar TPR

---

### Ejercicio 5 — Comparación y reflexión
**Consigna**: Comparar modelo original vs. ajustado en performance y fairness, discutir trade-offs, reflexionar sobre impacto real.

**Conclusión general**: Las técnicas de mitigación logran reducir la disparidad de TPR entre grupos. La pequeña pérdida en performance se justifica porque: (1) un FN es más costoso que un FP, (2) mejor recall captura más suscriptores, (3) el costo marginal de llamadas extra es bajo.

---

---

# Resultados por proxy

## PROXY 1: `job` como proxy de género

**Grupos**: `hist_femenino` (housemaid, admin.) vs. `hist_masculino_otro` (resto)

### Ej2 — Performance por grupo

| Grupo | Accuracy | Recall(yes) | Precision(yes) | N |
|-------|----------|-------------|-----------------|------|
| hist_femenino | 0.8854 | 0.3154 | 0.6483 | 2226 |
| hist_masculino_otro | 0.9041 | 0.2962 | 0.5831 | 5996 |

**Observación**: Diferencia de Recall pequeña (~0.02). El grupo históricamente femenino tiene recall ligeramente mayor, lo que indica que el modelo no discrimina severamente contra este grupo en detección de suscriptores.

### Ej3 — Fairness (umbral = 0.1)

| Métrica | Disparidad | ¿Fair? |
|---------|-----------|--------|
| Statistical Parity | 0.0119 | ✅ |
| Equal Opportunity (TPR) | 0.0193 | ✅ |
| Predictive Parity (Precision) | 0.0652 | ✅ |
| FPR (Equalized Odds) | 0.0017 | ✅ |

**Resultado**: El modelo original **cumple todos los criterios de fairness** con el umbral de 0.1 para el proxy de género.

### Ej4 — Mitigación

- **Reweighting**: Compensa el desbalance entre grupos de `job`, dando más peso a combinaciones sub-representadas (ej: hist_femenino + yes).
- **Ajuste de umbral**: Encuentra umbrales individuales para cada grupo que acercan los TPR al máximo observado.

### Ej5 — Comparación

- **Reweighting** mejora la equidad (menor disparidad de TPR) con costo leve en Accuracy/Precision global.
- **Ajuste de umbral** iguala el TPR de forma directa pero puede aumentar FPR para ciertos grupos.
- En ambos casos, la pérdida de performance es justificable desde la perspectiva del banco (captar más suscriptores potenciales > reducir llamadas innecesarias).

---

## PROXY 2: `marital` como proxy de edad

**Grupos**: `married`, `single`, `divorced`

### Ej2 — Performance por grupo

| Grupo | Accuracy | Recall(yes) | Precision(yes) | N |
|-------|----------|-------------|-----------------|------|
| married | 0.9086 | 0.2736 | 0.5991 | 5055 |
| single | 0.8724 | 0.3483 | 0.6042 | 2296 |
| divorced | 0.9139 | 0.2941 | 0.6250 | 871 |

**Observación**: `single` (jóvenes) tiene el mayor Recall, `married` el menor. La diferencia máxima de TPR (~0.075) sugiere un posible sesgo etario pero se mantiene bajo el umbral de 0.1.

### Ej3 — Fairness (umbral = 0.1)

| Métrica | Máx. disparidad | Par con máx. disp. | ¿Fair? |
|---------|-----------------|---------------------|--------|
| Statistical Parity | 0.0377 | married vs single | ✅ |
| Equal Opportunity (TPR) | 0.0747 | married vs single | ✅ |
| Predictive Parity (Precision) | 0.0259 | married vs divorced | ✅ |
| FPR (Equalized Odds) | 0.0196 | single vs divorced | ✅ |

**Resultado**: El modelo original también **cumple todos los criterios de fairness** para el proxy de edad, aunque las disparidades son algo mayores que con el proxy de género.

### Ej4 — Mitigación

- **Reweighting**: Compensa la sobrerrepresentación de `married` (~60%) dando más peso a `single` y especialmente a `divorced` (~11%).
- **Ajuste de umbral**: Con tres grupos, se optimizan umbrales individuales para acercar los TPR del grupo más rezagado (married) al máximo (single).

### Ej5 — Comparación

- Las técnicas de mitigación logran reducir la disparidad de TPR entre los tres grupos de estado civil.
- El trade-off equidad/performance se justifica con el mismo argumento que para el proxy de género: captar suscriptores perdidos vale más que evitar llamadas extra.

---

## Reflexión final

A lo largo de este TP se demostró un **pipeline completo de auditoría y mitigación de sesgos**:

1. **Conocer el dataset** (Ej1) → identificar variables protegidas, proxies, sesgos en los datos
2. **Construir el baseline** (Ej2) → evaluar performance y entender el tipo de error más costoso
3. **Auditar equidad** (Ej3) → medir fairness con criterios formales
4. **Mitigar** (Ej4) → aplicar técnicas de pre y post-procesamiento
5. **Comparar y reflexionar** (Ej5) → evaluar trade-offs y entender las implicaciones

### Puntos clave

- **No existe un modelo perfectamente justo Y perfectamente preciso**. La elección del criterio de fairness y del umbral son decisiones de *política*, no solo técnicas.
- **Los sistemas de ML amplifican sesgos históricos**. Sin intervención activa, estos sesgos se perpetúan y escalan.
- **Las limitaciones del proxy** (usar `job` como aproximación de género) subrayan la importancia de recolectar datos demográficos reales para auditorías más rigurosas.
- **La equidad no es opcional** — es un requisito ético y, en muchos contextos financieros, legal.

---

## Estructura del repositorio

```
TP_equidades/
├── data/
│   └── full.csv              # Dataset Bank Marketing (UCI)
├── ej1.ipynb                  # Análisis exploratorio y marco conceptual
├── ej2.ipynb                  # Modelo base + evaluación + análisis por proxy
├── ej3.ipynb                  # Análisis de fairness (ambos proxies)
├── ej4.ipynb                  # Mitigación de sesgos (ambos proxies)
├── ej5.ipynb                  # Comparación y reflexión final
└── README.md                  # Este archivo
```

## Requisitos

- Python 3.x
- pandas, numpy, matplotlib, seaborn, scikit-learn
