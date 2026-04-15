# GUIÓN DE PRESENTACIÓN: EQUIDAD EN IA - CASO BANK MARKETING

**Tiempo estimado:** 15-20 minutos.
**Integrantes:** [Tus Nombres]
**Contexto:** Avance preliminar (Ejercicios 1, 2 y 3).

---

## 1. INTRODUCCIÓN Y OBJETIVOS (2 minutos)

### Slide 1: Título y Contexto
- **Discurso:** Buen día. Hoy presentamos el avance de nuestro trabajo práctico sobre Equidad en Aprendizaje Automático. Elegimos el dataset de **Bank Marketing**, extraído del repositorio UCI, el cual contiene datos de una institución bancaria de Portugal.

### Slide 2: Objetivos del Proyecto
- **Puntos clave:** Incrementar suscripciones a depósitos a plazo fijo mediante telemarketing eficientemente.
- **Discurso:** El objetivo principal del banco es optimizar sus campañas de telemarketing. Buscan un modelo que prediga si un cliente suscribirá un depósito a plazo fijo para priorizar a quién llamar. Nuestra meta técnica es maximizar el acierto (Accuracy) pero nuestro enfoque ético es garantizar que este proceso no discrimine a grupos específicos.

---

## 2. ACCIONES, DATOS Y ANÁLISIS (3 minutos)

### Slide 3: El Dataset (Datasheet for Datasets)
- **Discurso:** En el Ejercicio 1 realizamos una auditoría del origen de los datos. El dataset representa contactos telefónicos reales entre 2008 y 2010. Es un proceso híbrido: registros administrativos del banco combinados con indicadores macroeconómicos de la Eurozona.

### Slide 4: Análisis Exploratorio
- **Discurso:** Analizamos la distribución de las variables clave. Por ejemplo, observamos cómo el tipo de trabajo o el nivel educativo influyen en la suscripción.
- **Visualización:**
![Distribución de Trabajo](presentation_assets/img_1.png)
*(Aquí explicamos cómo el trabajo puede ser un proxy de nivel socioeconómico)*

### Slide 5: Sesgos Históricos y Subrepresentación
- **Discurso:** Identificamos fallas potenciales: un fuerte desbalance de clases (90% "no"). Además, vimos grupos subrepresentados en educación y estado civil.
- **Visualización:**
![Distribución de Educación](presentation_assets/img_5.png)
*(Explicamos el sesgo potencial haca personas con educación terciaria)*

---

## 3. COMPORTAMIENTOS DISCRIMINATORIOS Y ESCENARIOS DE DAÑO (4 minutos)

### Slide 6: Definición del Escenario de Daño
- **Discurso:** Definimos nuestro escenario de daño basándonos en el **Estado Civil** como variable protegida (provisionalmente usada como proxy de género ante la falta de esa variable).
- **Visualización:**
![Distribución Marital](presentation_assets/img_7.png)

### Slide 7: Gravedad de los Errores (FP vs. FN)
- **Discurso:** Concluimos que el **Falso Negativo (FN)** es el más grave. Un FN es un cliente que quería el producto pero el banco no lo llamó, perdiendo una oportunidad comercial. El daño es la **denegación injusta de servicios**.

---

## 4. EL MODELO Y MÉTRICAS DE DESEMPEÑO (3 minutos)

### Slide 8: Implementación del Modelo (Random Forest)
- **Discurso:** Entrenamos un Random Forest. Obtuvimos un Accuracy del 91%.
- **Visualización:**
![Matriz de Confusión](presentation_assets/img_13.png)
- **Discurso:** La matriz muestra que, aunque el acierto total es alto, fallamos mucho en detectar el "sí" (Recall bajo).

### Slide 9: Desempeño por Grupos
- **Visualización:**
![Accuracy por Edad](presentation_assets/img_14.png)
- **Discurso:** Al abrir el desempeño por grupos de edad, vemos variaciones que podrían indicar que el modelo funciona mejor para ciertos rangos etarios.

---

## 5. AUDITORÍA DE EQUIDAD (2 minutos)

### Slide 10: Criterios de Fairness (Ejercicio 3)
- **Discurso:** Comparamos los grupos de "Casados" vs "Otros". Evaluamos Igualdad de Oportunidades y Paridad Estadística.
- **Visualización:**
![Métricas de Equidad](presentation_assets/img_15.png)

### Slide 11: Resultados y Elección del Criterio
- **Discurso:** Elegimos **Equal Opportunity**. Queremos que el Recall sea parejo. En nuestras pruebas, las diferencias son menores al 0.1, por lo que el modelo es inicialmente "fair", pero buscaremos mejorar el Recall general sin romper esta equidad.

---

## 6. CONCLUSIÓN (1 minuto)

### Slide 12: Reflexión Final
- **Discurso:** Confirmamos que una buena métrica global no garantiza un modelo justo. El monitoreo de la equidad es nuestro compromiso. En la siguiente fase, aplicaremos técnicas de mitigación de sesgo. ¡Muchas gracias!
