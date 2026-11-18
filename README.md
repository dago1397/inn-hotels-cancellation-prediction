# INN Hotels: predicción de cancelaciones de reservas

Construí un árbol de decisión con post-poda que predice cancelaciones de reservas con un recall del 86,03 % y un F1 de 80,74 en 10.883 reservas de prueba, para orientar políticas de reembolso, depósitos y overbooking.

## Problema de negocio

INN Hotels Group, una cadena de hoteles en Portugal, enfrenta un alto número de cancelaciones y *no-shows*. Eso implica pérdida de ingresos si no se revende la habitación, más comisiones o publicidad, bajadas de precio de último minuto y carga operativa. El encargo es identificar los factores que más influyen en la cancelación, anticipar qué reserva se va a cancelar y formular políticas rentables de cancelación y reembolso.

## Datos

Archivo: [`INNHotelsGroup.csv`](INNHotelsGroup.csv). **36.275 reservas** y **19 columnas**, sin valores nulos.

La variable objetivo es `booking_status`: **32,76 %** canceladas y 67,24 % no canceladas. La clase positiva es cancelar (`Canceled = 1`).

Predictores: número de adultos y de niños, noches de fin de semana y entre semana, plan de comidas, estacionamiento, tipo de habitación, *lead time*, mes de llegada, segmento de mercado, huésped repetido, cancelaciones y reservas previas no canceladas, precio promedio por habitación (euros) y número de peticiones especiales. Se excluyeron `Booking_ID`, `arrival_year` y `arrival_date` antes de modelar.

Partición 70/30 (`random_state=42`): **25.392** reservas de entrenamiento y **10.883** de prueba.

## Enfoque

Comparé cuatro modelos, priorizando el **recall**: el error más costoso es no anticipar una cancelación (falso negativo).

1. **Regresión logística** con `statsmodels` (umbral 0,5). Se revisó la multicolinealidad (VIF) y se eliminó `market_segment_type_Online`.
2. **Árbol de decisión sin podar** (`DecisionTreeClassifier`, `random_state=1`). Sobreajuste claro en entrenamiento (recall 98,42 %).
3. **Pre-poda** con `GridSearchCV` (5 folds, score = recall): el mejor estimador quedó con `max_depth=2` y `class_weight='balanced'`.
4. **Post-poda** por complejidad de costos, con `class_weight='balanced'`. Se eligió el `ccp_alpha` que maximiza el recall en test: **7,94×10⁻⁵**.

El modelo final es el **árbol con post-poda**.

## Resultados clave

Métricas en el conjunto de prueba (10.883 reservas):

| Modelo | Accuracy | Precision | Recall | Specificity | F1 |
|--------|----------|-----------|--------|-------------|-----|
| Regresión logística | 80,46 % | 74,85 % | 61,8 % | 89,71 % | 67,7 % |
| Árbol sin podar | 86,56 % | 79,37 % | 80,32 % | 89,65 % | 79,84 % |
| Árbol con pre-poda | 62,11 % | 46,14 % | 85,75 % | 50,38 % | 60,0 % |
| **Árbol con post-poda (final)** | **86,4 %** | **76,07 %** | **86,03 %** | **86,59 %** | **80,74 %** |

El árbol post-podado sube el recall de 61,8 % (logística) a 86,03 % y el F1 de 67,7 a 80,74, con un mejor equilibrio que la pre-poda (recall 85,75 %, pero accuracy 62,11 % y precision 46,14 %).

Variables con más peso en el modelo final:

| Variable | Importancia |
|----------|-------------|
| `lead_time` | 0,39 |
| `avg_price_per_room` | 0,16 |
| `market_segment_type` | 0,13 |
| `no_of_special_requests` | 0,11 |
| `arrival_month` | 0,07 |

## Recomendaciones de negocio

- Aplicar reembolsos más estrictos o un depósito en reservas de alto riesgo (lead time largo, precio alto, segmento OTA).
- Mantener flexibilidad para huéspedes repetidos y para el canal directo.
- Guiar el *overbooking* con las probabilidades del modelo.
- Lanzar campañas de reconfirmación anticipada.
- Incentivar las reservas directas.

## Herramientas

Python, pandas, NumPy, scikit-learn (`DecisionTreeClassifier`, `GridSearchCV`, métricas de clasificación), statsmodels (`Logit`, VIF), Matplotlib y Seaborn.

## Estructura del repositorio

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── INNHotelsGroup.csv
└── Project_SLC_DSBA_INNHotels_FullCode.ipynb
```

## Cómo ejecutarlo

```bash
python -m venv .venv
source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Abre `Project_SLC_DSBA_INNHotels_FullCode.ipynb` en Jupyter, VS Code o Google Colab. El notebook ya incluye las salidas ejecutadas (tablas y gráficas), así que se puede revisar el análisis sin volver a correrlo.

**Ruta de los datos.** El notebook se escribió en Google Colab y lee el CSV con una ruta fija de Drive; no la modifiqué, para no alterar el código entregado:

```python
data = pd.read_csv('/content/drive/MyDrive/GREAT LEARNING/4. SUPERVISED LEARNING CLASSIFICATION/Proyect/INNHotelsGroup.csv')
```

Para ejecutarlo en local, cambia esa llamada a:

```python
data = pd.read_csv('./INNHotelsGroup.csv')
```

Si lo corres en Colab, sube `INNHotelsGroup.csv` y actualiza la ruta, o coloca el archivo en la ruta original de Drive después de montar el disco.

## Crédito

Project completed as part of the Post-Graduate Program in Data Science and Business Analytics, The University of Texas at Austin / Great Learning (2025).

---

# INN Hotels: booking cancellation prediction

I built a post-pruned decision tree that predicts booking cancellations with 86.03% recall and an F1 of 80.74 on 10,883 test reservations, to guide refund, deposit, and overbooking policy.

## Business problem

INN Hotels Group, a hotel chain in Portugal, faces a high number of cancellations and no-shows. That means lost revenue when a room is not resold, extra commissions or advertising, last-minute price cuts, and operational strain. The brief is to identify the main drivers of cancellation, predict which bookings will be canceled, and design profitable cancellation and refund policies.

## Data

File: [`INNHotelsGroup.csv`](INNHotelsGroup.csv). **36,275 bookings** and **19 columns**, with no missing values.

The target is `booking_status`: **32.76%** canceled and 67.24% not canceled. The positive class is a cancellation (`Canceled = 1`).

Predictors include party size, weekend and weekday nights, meal plan, parking, room type, lead time, arrival month, market segment, repeat-guest status, prior cancellations and prior non-canceled bookings, average room price (euros), and special requests. `Booking_ID`, `arrival_year`, and `arrival_date` were dropped before modeling.

A 70/30 split (`random_state=42`) yields **25,392** training bookings and **10,883** test bookings.

## Approach

I compared four models and prioritized **recall**: the costliest error is failing to anticipate a cancellation (a false negative).

1. **Logistic regression** with `statsmodels` (0.5 threshold). Multicollinearity was checked with VIF, and `market_segment_type_Online` was removed.
2. **Unpruned decision tree** (`DecisionTreeClassifier`, `random_state=1`). Clear overfitting on the training set (98.42% recall).
3. **Pre-pruning** with `GridSearchCV` (5-fold, scoring = recall): the best estimator used `max_depth=2` and `class_weight='balanced'`.
4. **Cost-complexity post-pruning**, with `class_weight='balanced'`. The `ccp_alpha` that maximizes test recall was selected: **7.94×10⁻⁵**.

The final model is the **post-pruned tree**.

## Key results

Test-set metrics (10,883 bookings):

| Model | Accuracy | Precision | Recall | Specificity | F1 |
|-------|----------|-----------|--------|-------------|-----|
| Logistic regression | 80.46% | 74.85% | 61.8% | 89.71% | 67.7% |
| Unpruned tree | 86.56% | 79.37% | 80.32% | 89.65% | 79.84% |
| Pre-pruned tree | 62.11% | 46.14% | 85.75% | 50.38% | 60.0% |
| **Post-pruned tree (final)** | **86.4%** | **76.07%** | **86.03%** | **86.59%** | **80.74%** |

The post-pruned tree lifts recall from 61.8% (logistic regression) to 86.03% and F1 from 67.7 to 80.74, with a better balance than pre-pruning (85.75% recall, but 62.11% accuracy and 46.14% precision).

Top features in the final model:

| Feature | Importance |
|---------|------------|
| `lead_time` | 0.39 |
| `avg_price_per_room` | 0.16 |
| `market_segment_type` | 0.13 |
| `no_of_special_requests` | 0.11 |
| `arrival_month` | 0.07 |

## Business recommendations

- Apply stricter refunds or a deposit on high-risk bookings (long lead time, high price, OTA segment).
- Keep flexibility for repeat guests and the direct channel.
- Guide overbooking with the model's predicted probabilities.
- Run early reconfirmation campaigns.
- Incentivize direct bookings.

## Tools

Python, pandas, NumPy, scikit-learn (`DecisionTreeClassifier`, `GridSearchCV`, classification metrics), statsmodels (`Logit`, VIF), Matplotlib, and Seaborn.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── INNHotelsGroup.csv
└── Project_SLC_DSBA_INNHotels_FullCode.ipynb
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open `Project_SLC_DSBA_INNHotels_FullCode.ipynb` in Jupyter, VS Code, or Google Colab. The notebook already contains executed outputs (tables and plots), so the analysis can be reviewed without re-running it.

**Data path.** The notebook was written in Google Colab and reads the CSV from a hardcoded Google Drive path; I left that path unchanged so the delivered code stays intact:

```python
data = pd.read_csv('/content/drive/MyDrive/GREAT LEARNING/4. SUPERVISED LEARNING CLASSIFICATION/Proyect/INNHotelsGroup.csv')
```

To run it locally, change that call to:

```python
data = pd.read_csv('./INNHotelsGroup.csv')
```

If you run it in Colab, upload `INNHotelsGroup.csv` and update the path, or place the file at the original Drive path after mounting Drive.

## Credit

Project completed as part of the Post-Graduate Program in Data Science and Business Analytics, The University of Texas at Austin / Great Learning (2025).
