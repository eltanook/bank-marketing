---
marp: true
theme: custom
paginate: true
style: |
  section {
    background-color: #F0F8FF;
    color: #002244;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    padding: 60px 80px 80px 80px; /* Incrementado el padding inferior */
  }
  h1, h2, h3 {
    color: #0047AB;
    margin-bottom: 0.5em;
  }
  h1 { font-size: 2.2em; font-weight: bold; }
  h2 { font-size: 1.6em; border-bottom: 2px solid #0047AB; padding-bottom: 10px; }
  p, li { font-size: 1.1em; line-height: 1.4; }
  ul { padding-left: 1.2em; }
  strong { color: #003366; }
  
  .columns { display: flex; gap: 2em; align-items: center; }
  .column { flex: 1; }
  
  img {
    max-width: 100%;
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
---

<!-- _class: center -->

# Equidad en Aprendizaje Automático
## Resultados del Trabajo Práctico: Bank Marketing

---

## 1. El Dataset: Bank Marketing

**Motivación**
Predecir si un cliente **suscribirá a un plazo fijo** para optimizar campañas de marketing.

**Naturaleza de los Datos**
- **45,211 instancias** (llamadas telefónicas).
- Variables sociodemográficas y económicas.

**Variable Protegida (Foco de Equidad)**
- **`job` (Ocupación) como proxy de género** (ej. "housemaid" vs "admin").

---

## 2. Sesgos Potenciales

Fuertes desbalances estructurales en los datos:

<div class="columns">
<div class="column">

- **Etiquetas**: ~90% de los contactos dicen "no".
- **Educación**: Nivel secundario es mayoría absoluta.
- **Demografía**: Grupos como solteros o jubilados están fuertemente subrepresentados.

</div>
<div class="column">
<div class="highlight-box">
<strong>Riesgo:</strong><br>
El modelo puede priorizar inadvertidamente a los grupos mayoritarios, siendo ineficaz para captar a las minorías.
</div>
</div>
</div>

---

## 3. El Costo del Error

Al entrenar el modelo, evaluamos el impacto del error:

**El Falso Negativo (FN) es el más crítico.**
- **Falso Negativo**: Predecir que el cliente *no* aceptará, cuando en realidad *sí* lo haría.
- **Impacto**: Pérdida directa de un cliente potencial (costo de oportunidad).
- *Un Falso Positivo solo cuesta el tiempo de una llamada.*

---

## 4. Evaluación Inicial (`job` como proxy)

Evaluación del modelo sin mitigación utilizando la ocupación como proxy de género.

<div class="columns">
<div class="column">

- Las tasas de verdaderos positivos (captar al cliente real) varían drásticamente.
- Trabajos "feminizados" reciben predicciones negativas desproporcionadas debido a correlaciones espurias del dataset.

</div>
<div class="column">
  <img src="../assets/ej2_img1.png" alt="Evaluación Inicial 1">
</div>
</div>

---

## 5. Criterios de Equidad (Fairness)

Evaluación de criterios matemáticos de sesgo.

<div class="columns">
<div class="column">

**Criterio Relevante: Igualdad de Oportunidades**
Como el Falso Negativo es el peor error, exigimos que la **Tasa de Verdaderos Positivos (TPR) sea igual para todos los grupos**. 

*Si un cliente realmente va a suscribirse, el modelo debe detectarlo sin importar su ocupación.*

</div>
<div class="column">
  <img src="../assets/ej3_img1.png" alt="Criterios Cuantitativos">
</div>
</div>

---

## 6. Mitigación: Reweighing (Pre-procesamiento)

**Técnica:** Asigna "pesos" a los datos de entrenamiento para balancear la importancia de grupos desfavorecidos.

<div class="columns">
<div class="column">

- **No altera los datos**, solo su distribución de peso.
- Evita que el modelo penalice a grupos minoritarios.
- *Resultado*: Mejora leve, pero suele ser insuficiente ante sesgos profundos.

</div>
<div class="column">
  <img src="../assets/ej4_img1.png" alt="Resultados Reweighing">
</div>
</div>

---

## 7. Mitigación: Equalized Odds (Post-procesamiento)

**Técnica:** Modifica los umbrales de decisión del modelo ya entrenado, forzando matemáticamente la igualdad de TPR y FPR.

<div class="columns">
<div class="column">

- Intervención directa sobre el resultado final.
- **Garantiza la Igualdad de Oportunidades.**
- *Costo*: Impacta negativamente en el "Accuracy" global del modelo.

</div>
<div class="column">
  <img src="../assets/ej4_img2.png" alt="Resultados Equalized Odds">
</div>
</div>

---

## 8. Fairness vs. Performance

Mitigar sesgos conlleva un costo en el rendimiento matemático general.

<div class="columns">
<div class="column">

**El Trade-off**
- Igualar oportunidades obliga al modelo a cometer errores "intencionales" para no discriminar.
- **Decisión:** ¿Cuánto *Accuracy* sacrificamos para garantizar un trato justo?

</div>
<div class="column">
  <img src="../assets/ej5_img1.png" alt="Trade-off Plot">
</div>
</div>

---

<!-- _class: center -->

## 9. Reflexión en el Mundo Real

**Más allá de los números**
Un modelo sesgado genera **ciclos de retroalimentación negativos**:
Grupos marginados no acceden al producto $\rightarrow$ no generan historial $\rightarrow$ el modelo futuro aprende que "no son exitosos".

<div class="highlight-box" style="text-align: left;">
<strong>Conclusión:</strong><br>
Los algoritmos no son neutros. Mitigar sesgos es un imperativo ético para no amplificar inequidades estructurales a gran escala.
</div>
