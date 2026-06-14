# TODO: Próximas exploraciones del proyecto

Lista priorizada de extensiones para el proyecto **Shallow Water + Wavebreaker** (PINN).

---

## 1. Diagrama espacio-tiempo (Hovmöller) 🎨

**Esfuerzo:** ~30 min · **Valor:** Visualización muy informativa

- Plotear `h(x, t)` y `u(x, t)` como **imágenes 2D** (eje x = espacio, eje y = tiempo, color = valor).
- Permite ver de un vistazo:
  - Frentes de onda y su velocidad de propagación
  - Reflexiones contra el wavebreaker
  - Atrapamiento de energía sobre el obstáculo
  - Periodicidad / estacionariedad

**Implementación esperada:**

- `utils/plot.py` → función `plot_hovmoller(field, x, t, outdir, name)`
- Llamar desde `debug.py` o desde el script principal de evaluación.

---

## 2. Conservación de masa y energía 📐

**Esfuerzo:** ~1 h · **Valor:** Diagnóstico crítico para PINN

Calcular en función del tiempo:

$$M(t) = \int_0^L h(x,t)\, dx$$

$$E(t) = \int_0^L \tfrac{1}{2}\left(h u^2 + g h^2\right) dx$$

- En SW sin fricción, `M(t)` debe ser **exactamente constante** y `E(t)` casi constante (salvo shocks).
- Si el PINN no conserva masa → señal de underfitting de la ecuación de continuidad.
- Plotear `M(t)/M(0)` y `E(t)/E(0)` como métrica de calidad.

**Implementación esperada:**

- `utils/diagnostics.py` (nuevo) → funciones `mass(h, dx)`, `energy(h, u, dx, g)`
- Integración con `torch.trapz` o regla del trapecio en numpy.

---

## 3. Coeficiente de transmisión y reflexión 🌊

**Esfuerzo:** ~2-3 h · **Valor:** Métrica física estándar en ingeniería costera

Para una onda incidente sobre el wavebreaker:

$$R = \frac{E_{reflejada}}{E_{incidente}}, \quad T = \frac{E_{transmitida}}{E_{incidente}}$$

- Descomponer el campo en aguas arriba en componentes incidente + reflejada (ventana temporal).
- Medir energía aguas abajo (transmitida).
- Verificar `R + T + D = 1` (donde `D` = disipación).
- Hacer **barrido paramétrico** sobre:
  - Altura del wavebreaker `H`
  - Ancho `(b - a)`
  - Número de wavebreakers (1, 2, N) → buscar **bandas de Bragg**

**Implementación esperada:**

- `experiments/transmission_reflection.py`
- Output: gráfica `R, T vs. H` y `R, T vs. (b-a)`.

---

## 4. Curriculum learning sobre `k` (suavidad del wavebreaker) 🧠

**Esfuerzo:** ~3-4 h · **Valor:** Necesario para topografía cuadrada

Entrenar con topografía progresivamente más rectangular:

| Fase | `k`  | Forma                  |
| ---- | ---- | ---------------------- |
| 1    | 10   | Lomo suave (gaussiano) |
| 2    | 25   | Trapezoidal redondeado |
| 3    | 50   | Casi rectangular       |
| 4    | 100+ | Escalón                |

- Aumentar `k` cada N épocas o cuando el loss se estabiliza.
- Compararlo con entrenamiento directo en `k=100` (debería fallar o ser mucho más lento).
- Documentar curvas de loss por fase.

**Implementación esperada:**

- Modificar `wavebreaker_topography` para aceptar `k` dinámico.
- Scheduler de `k` en el loop de entrenamiento.

---

## 5. Análisis "river meets shoal" (aceleración por conservación de masa) 🚀

**Esfuerzo:** ~1-2 h · **Valor:** Validación física directa del PINN

Cuantificar el efecto observado en el caso 7 (`Ztwowavebreakers_Hgauss_Vsine`), donde
`u` se amplifica fuertemente al pasar de aguas profundas a aguas someras sobre el wavebreaker.

Predicción teórica (continuidad, caudal `q = h·u` conservado localmente):

$$u_{shoal} = u_{deep} \cdot \frac{h_{deep}}{h_{shoal}}$$

- Extraer `u(x, t)` y `h(x, t)` del PINN sobre líneas verticales (un `x` específico) o regiones.
- Comparar la razón empírica `u_shoal / u_deep` con la razón teórica `h_deep / h_shoal`.
- Plotear el ratio como función del tiempo → debería tender al valor teórico en régimen cuasi-estacionario.
- **Métrica de validación física:** error relativo entre la predicción del PINN y la ley de continuidad.

**Casos de barrido sugeridos:**

- Variar `H` del wavebreaker (controla `h_shoal`).
- Variar amplitud inicial de `u` (controla `u_deep`).

**Implementación esperada:**

- `utils/diagnostics.py` → función `continuity_ratio(h, u, x, x_deep, x_shoal)`.
- Plot tipo `ratio_vs_t.png` con la línea teórica como referencia.

---

## 6. Búsqueda de resaltos hidráulicos (hydraulic jumps) ⚡

**Esfuerzo:** ~2-3 h · **Valor:** Fenómeno no lineal espectacular, alto impacto en informe

Forzar al sistema a entrar en **régimen supercrítico** sobre el wavebreaker (`Fr > 1`)
y observar la formación de un resalto hidráulico.

Número de Froude local:

$$Fr(x, t) = \frac{|u(x, t)|}{\sqrt{g\, h(x, t)}}$$

- Para el caso 7 actual, `Fr_max ≈ 0.5` → subcrítico pero **cerca del crítico**.
- Aumentar la amplitud inicial de `u` (p. ej. `u₀(x) = 0.8·sin(πx)` o más) o bajar `h₀`
  para empujar el sistema a `Fr > 1` sobre el wavebreaker.
- Esperar: salto abrupto de `h` (pequeña → grande) y caída de `u` (grande → pequeña),
  acompañado de disipación de energía.

**Diagnósticos:**

- Mapa de `Fr(x, t)` (Hovmöller del Froude) → marcar contorno `Fr = 1`.
- Detección automática de discontinuidades en `h` (gradiente espacial alto).
- Verificar relaciones de Rankine-Hugoniot en torno al salto.

**⚠️ Consideración numérica:**
Los PINNs vanilla **suelen tener problemas con shocks** (suavizan discontinuidades).
Esto puede convertirse en un sub-experimento sobre limitaciones del método y motivar
variantes (weighted residuals, viscosidad artificial, dominio descomposición).

**Implementación esperada:**

- `utils/diagnostics.py` → función `froude(h, u, g)`.
- Nuevo caso en `utils/cases.py`: `Ztwowavebreakers_Hgauss_VsineHigh` (mayor amplitud).
- Plot dedicado: panel con `h`, `u`, `Fr` en el mismo `x` para distintos `t`.

---

## Orden recomendado

1 → 2 → 5 → 6 → 3 → 4

(rápido/visual → diagnósticos → física no lineal → métricas formales → metodología)
