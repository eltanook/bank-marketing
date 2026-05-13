# TP Equidad en Aprendizaje Automático

## Contexto y Dataset
**Dataset**: Bank Marketing (UCI) — 41.188 registros de campañas de telemarketing de un banco portugués.  
**Objetivo de Negocio**: Incrementar la tasa de suscripción a depósitos a plazo fijo, prediciendo si un cliente suscribirá o no (`y`: yes/no).

## Variables Protegidas y Proxies Analizados
La consigna establece que las variables protegidas oficiales para este dominio bancario son el **estado civil** y la **edad**, sumado a un enfoque particular en el **género**. Debido a la composición del dataset, el análisis se estructuró de la siguiente forma:

| Variable / Proxy | Nombre en Dataset | Justificación y Uso |
|-------|----------|---------------|
| **Género** (Proxy) | `job` | Dado que el dataset no incluye género directo, se utilizó la ocupación como proxy, agrupando trabajos históricamente femeninos (`housemaid`, `admin.`) versus el resto. |
| **Estado civil / Edad** | `marital` | Variable protegida oficial asignada por la consigna. Se utiliza porque el estado civil correlaciona fuertemente con la etapa de la vida (solteros=jóvenes, casados=mediana edad, divorciados=mayores). |

> **Nota Crítica sobre Data Leakage**: La variable `duration` (duración de la llamada) fue **excluida de todos los modelos predictivos**. Como advierten los autores del dataset, esta variable solo se conoce *después* de que la llamada ha finalizado y determina directamente el éxito de la misma. Incluirla crearía un modelo sesgado e irrealizable en un entorno de producción.

---

## Resumen de la Implementación por Ejercicio

### Ejercicio 1 — Análisis Exploratorio y Marco Conceptual
Se realizó una investigación profunda sobre el origen y propósito del dataset siguiendo la metodología *Datasheets for Datasets*.  
* **Identificación de Sesgos**: Se detectó un fuerte desbalance de clases (casi 90% de los contactos resultan en un "no"). También se identificó sub-representación en grupos demográficos clave, como solteros y divorciados frente a casados, y una distribución muy desigual en niveles educativos.
* **Depuración Conceptual**: Aunque durante la exploración inicial se barajaron variables de tipo operativo (como `contact` - celular vs. teléfono fijo) o la edad numérica, para el desarrollo formal del TP se consolidó el uso de `job` (como proxy de género) y `marital` (como proxy de edad y estado civil). Esto permitió centrar el análisis de equidad exclusivamente en atributos demográficos éticamente relevantes y cumplir orgánicamente con la consigna.

### Ejercicio 2 — Modelo Base y Evaluación del Error
Se entrenó un modelo clásico de Machine Learning como punto de partida (baseline).
* **Modelo Elegido**: Random Forest Classifier (100 estimadores), validado con un split estratificado 80/20 para mitigar el desbalance.
* **Evaluación de Métricas**: Si bien el modelo arrojó un *Accuracy* engañosamente alto (~90%), el *Recall* para la clase positiva resultó bajo, revelando dificultades para detectar a los verdaderos suscriptores debido a la escasez de ejemplos positivos.
* **Determinación del Error Crítico**: Desde la perspectiva comercial del Banco, se determinó que **el Falso Negativo (FN) es el error más perjudicial**. Un FN implica no contactar a alguien que sí se hubiera suscrito (pérdida directa de cliente y depósito), mientras que un Falso Positivo (FP) solo representa el costo operativo marginal de realizar una llamada telefónica adicional.

### Ejercicio 3 — Auditoría de Equidad (Fairness)
Se auditó cómo el modelo base trata a los distintos grupos demográficos utilizando 4 criterios formales (Statistical Parity, Equal Opportunity, Equalized Odds, Predictive Parity), con un umbral de tolerancia de disparidad del 10% (0.1).
* **Resultados Iniciales**: El modelo original cumplió matemáticamente con todos los criterios de fairness bajo el umbral estipulado para ambos proxies, aunque evidenciando brechas menores.
* **Criterio Seleccionado (El más relevante)**: **Equal Opportunity**. Al priorizar la minimización de Falsos Negativos (conclusión del Ej2), es imperativo asegurar que el modelo detecte a los verdaderos suscriptores potenciales con la misma eficacia (es decir, el mismo *True Positive Rate* o Recall) en todos los grupos. Evitamos así que el modelo sea bueno detectando hombres y malo detectando mujeres, o eficaz con casados y deficiente con divorciados.

### Ejercicio 4 — Mitigación de Sesgos
Para mejorar proactivamente la equidad e intentar cerrar las brechas observadas en el TPR, se aplicaron y evaluaron dos estrategias de mitigación:
1. **Reweighting (Pre-procesamiento)**: Se reasignaron pesos a las instancias de entrenamiento de forma inversamente proporcional a la frecuencia de su combinación grupo-clase. Esto obliga al algoritmo a "prestar más atención" a grupos minoritarios (ej. mujeres suscriptoras).
2. **Ajuste de Umbrales por Grupo (Post-procesamiento)**: En lugar de usar un umbral de decisión fijo del 50%, se recalibraron los umbrales de probabilidad de forma independiente para cada grupo protegido, con el objetivo matemático explícito de igualar sus tasas de Verdaderos Positivos.

### Ejercicio 5 — Comparación de Resultados y Reflexión
Se confrontaron las métricas de rendimiento y equidad del modelo base contra los modelos mitigados.
* **Trade-off Equidad vs. Performance**: Ambas técnicas de mitigación lograron disminuir aún más la disparidad de *Equal Opportunity*. Sin embargo, como es habitual, esto se logró a costa de una ligera reducción en el *Accuracy* y *Precision* global.
* **Impacto en el Mundo Real**: Se concluyó que esta leve caída en el rendimiento absoluto es no solo justificable, sino deseable. Capturar a clientes de grupos que el modelo original ignoraba (mejorando su Recall específico) incrementa las oportunidades de depósito totales. Además, garantiza que el Banco opere de manera ética, ofrezca un acceso igualitario a sus productos financieros y se proteja ante riesgos legales y daños reputacionales por discriminación algorítmica.

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
├── ej4.ipynb                  # Implementación de técnicas de mitigación
├── ej5.ipynb                  # Comparación de métricas y reflexión final
└── README.md                  # Este documento (Overview del TP)
```

## Requisitos Técnicos e Instalación
Para replicar y ejecutar los notebooks se recomienda un entorno con Python 3.x y las siguientes librerías principales:
- `pandas` y `numpy` para el manejo y manipulación de datos tabulares.
- `matplotlib` y `seaborn` para la generación de visualizaciones.
- `scikit-learn` para el preprocesamiento de características, modelado predictivo y cálculo de métricas de rendimiento y equidad.
