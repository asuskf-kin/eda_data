#!/usr/bin/env python3
"""Genera el informe HTML de EDA a partir de las salidas del universo completo.

Las tablas se construyen desde el JSON, no a mano: 190 pares de correlacion, 39 cadenas
y 60 zonas metropolitanas transcritos manualmente serian una fuente segura de errores.

    uv run python scripts/render_informe_eda.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
REP = RAIZ / "outputs/report"
R = json.loads((REP / "eda_universo_completo.json").read_text(encoding="utf-8"))
MAPA = pd.read_csv(REP / "universo_mapa.csv")

SEM_COLOR = {"VERDE": "#2c7a52", "AMARILLO": "#b8860b", "ROJO": "#b5382b"}


def tabla(cols, filas, num=(), clase=""):
    """cols: [(clave, etiqueta)]; num: claves alineadas a la derecha."""
    th = "".join(f'<th{" class=\"n\"" if k in num else ""}>{e}</th>' for k, e in cols)
    tr = []
    for f in filas:
        td = "".join(f'<td{" class=\"n\"" if k in num else ""}>{f.get(k, "")}</td>' for k, _ in cols)
        tr.append(f"<tr>{td}</tr>")
    return (f'<div class="tablebox {clase}"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(tr)}</tbody></table></div>')


# ===========================================================================
# Mapas
# ===========================================================================
def proyectar(lat, lon, w=680, h=440, x0=10, y0=20):
    """Equirectangular con correccion por coseno de la latitud media: sin ella,
    Mexico sale estirado ~9% en horizontal."""
    la0, la1 = MAPA.Latitud.min(), MAPA.Latitud.max()
    lo0, lo1 = MAPA.Longitud.min(), MAPA.Longitud.max()
    k = np.cos(np.radians((la0 + la1) / 2))
    ancho_deg = (lo1 - lo0) * k
    esc = min(w / ancho_deg, h / (la1 - la0))
    cx = x0 + (w - ancho_deg * esc) / 2 + (np.asarray(lon) - lo0) * k * esc
    cy = y0 + (h + (la1 - la0) * esc) / 2 - (np.asarray(lat) - la0) * esc
    return cx, cy


def mapa_pdv():
    d = MAPA.copy()
    # Se dibuja verde, luego amarillo, luego rojo: en zonas densas el rojo es lo que
    # interesa ver y quedaria tapado si se pintara primero.
    orden = {"VERDE": 0, "AMARILLO": 1, "ROJO": 2}
    d = d.assign(_o=d.sem_sueros.map(orden)).sort_values("_o")
    cx, cy = proyectar(d.Latitud.values, d.Longitud.values)
    pts = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.7" fill="{SEM_COLOR[s]}"/>'
        for x, y, s in zip(cx, cy, d.sem_sueros))
    leyenda = "".join(
        f'<rect x="{18 + i*128}" y="452" width="11" height="11" fill="{c}"/>'
        f'<text x="{34 + i*128}" y="462" font-size="11.5" fill="#333f47" '
        f'font-family="sans-serif">{s.capitalize()} · {int((MAPA.sem_sueros==s).sum()):,}</text>'
        for i, (s, c) in enumerate(SEM_COLOR.items()))
    return (f'<svg viewBox="0 0 700 472" role="img" aria-label="Distribución geográfica de los '
            f'8,864 puntos de venta en México, coloreados por el semáforo de Sueros">'
            f'<g opacity="0.62">{pts}</g>{leyenda}</svg>')


def mapa_zm():
    g = (MAPA.groupby("Zonas Metropolitanas")
         .agg(pdv=("Nielsen ID", "count"), lat=("Latitud", "median"), lon=("Longitud", "median"),
              rojo=("sem_sueros", lambda s: (s == "ROJO").mean()))
         .sort_values("pdv", ascending=False))
    g = g[g.index != "NON-METRO"]
    cx, cy = proyectar(g.lat.values, g.lon.values)
    rmax = np.sqrt(g.pdv.max())
    burb, etiq = [], []
    for i, ((zm, r), x, y) in enumerate(zip(g.iterrows(), cx, cy)):
        rad = 3 + np.sqrt(r.pdv) / rmax * 21
        col = "#b5382b" if r.rojo >= .33 else "#b8860b" if r.rojo >= .27 else "#2c7a52"
        burb.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="{col}" '
                    f'fill-opacity="0.45" stroke="{col}" stroke-width="1.1"/>')
        if i < 8:
            # Tijuana cae en x~14 y Cancun en x~686: sin acotar, la etiqueta centrada se sale
            xe = min(max(x, 44), 656)
            etiq.append(f'<text x="{xe:.1f}" y="{y - rad - 4:.1f}" text-anchor="middle" '
                        f'font-size="10" font-weight="600" fill="#10171c" '
                        f'font-family="sans-serif">{zm.title()}</text>')
    ley = ('<text x="18" y="458" font-size="11.5" fill="#333f47" font-family="sans-serif">'
           'Tamaño = PDV en la zona · Color = share de tiendas en rojo:</text>'
           '<circle cx="412" cy="454" r="5" fill="#2c7a52" fill-opacity="0.45" stroke="#2c7a52"/>'
           '<text x="422" y="458" font-size="11.5" fill="#333f47" font-family="sans-serif">&lt;27%</text>'
           '<circle cx="470" cy="454" r="5" fill="#b8860b" fill-opacity="0.45" stroke="#b8860b"/>'
           '<text x="480" y="458" font-size="11.5" fill="#333f47" font-family="sans-serif">27-33%</text>'
           '<circle cx="543" cy="454" r="5" fill="#b5382b" fill-opacity="0.45" stroke="#b5382b"/>'
           '<text x="553" y="458" font-size="11.5" fill="#333f47" font-family="sans-serif">&gt;33%</text>')
    return (f'<svg viewBox="0 0 700 472" role="img" aria-label="Zonas metropolitanas por número de '
            f'puntos de venta y proporción de tiendas con Sueros en rojo">'
            f'{"".join(burb)}{"".join(etiq)}{ley}</svg>'), g


# ===========================================================================
svg_pdv = mapa_pdv()
svg_zm, gzm = mapa_zm()
I, CS, PG, AF, SC, PN, SE = (R["integridad"], R["cluster_semaforo"], R["perfil_golden"],
                             R["afinidad"], R["scantrack"], R["potencial_nacional"], R["sensibilidad"])
N = R["meta"]["universo_pdv"]

# --- barras del semaforo por cluster
barras = []
for i, t in enumerate(CS["tabla"]):
    y, x = 18 + i * 42, 150
    barras.append(f'<text x="140" y="{y+16}" text-anchor="end" fill="#10171c" font-size="12.5" '
                  f'font-weight="700" font-family="sans-serif">{t["cluster"]}</text>'
                  f'<text x="140" y="{y+30}" text-anchor="end" fill="#667680" font-size="10" '
                  f'font-family="sans-serif">{t["pdv"]:,}</text>')
    for s in ["VERDE", "AMARILLO", "ROJO"]:
        w = t[s] / 100 * 522
        barras.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="28" fill="{SEM_COLOR[s]}"/>'
                      f'<text x="{x + w/2:.1f}" y="{y+19}" text-anchor="middle" fill="#fff" '
                      f'font-size="11" font-weight="700" font-family="sans-serif">{t[s]}</text>')
        x += w
svg_sem = (f'<svg viewBox="0 0 700 215" role="img" aria-label="Semáforo de Sueros por clúster: las '
           f'cuatro barras son casi idénticas">{"".join(barras)}'
           f'<line x1="150" y1="188" x2="672" y2="188" stroke="#c5ced3"/>'
           f'<text x="150" y="203" fill="#667680" font-size="10.5" font-family="sans-serif">0%</text>'
           f'<text x="411" y="203" text-anchor="middle" fill="#667680" font-size="10.5" '
           f'font-family="sans-serif">50%</text>'
           f'<text x="672" y="203" text-anchor="end" fill="#667680" font-size="10.5" '
           f'font-family="sans-serif">100%</text></svg>')

# --- Cohen's d, LOS 20 indices
pf = PG["detalle"]
mx = max(abs(p["d"]) for p in pf)
bd = []
for i, p in enumerate(pf):
    y = 16 + i * 20
    w = abs(p["d"]) / mx * 250
    if p["d"] >= 0:
        bd.append(f'<rect x="350" y="{y}" width="{w:.1f}" height="14" fill="#1f6f8b"/>'
                  f'<text x="344" y="{y+11}" text-anchor="end" fill="#10171c" font-size="11" '
                  f'font-family="sans-serif">{p["variable"]}</text>'
                  f'<text x="{356+w:.1f}" y="{y+11}" fill="#1f6f8b" font-size="10.5" '
                  f'font-weight="700" font-family="sans-serif">+{p["d"]:.2f}</text>')
    else:
        bd.append(f'<rect x="{350-w:.1f}" y="{y}" width="{w:.1f}" height="14" fill="#8d9aa2"/>'
                  f'<text x="356" y="{y+11}" fill="#10171c" font-size="11" '
                  f'font-family="sans-serif">{p["variable"]}</text>'
                  f'<text x="{344-w:.1f}" y="{y+11}" text-anchor="end" fill="#667680" font-size="10.5" '
                  f'font-weight="700" font-family="sans-serif">{p["d"]:.2f}</text>')
svg_cohen = (f'<svg viewBox="0 0 700 {16+len(pf)*20+22}" role="img" aria-label="Tamaño de efecto de '
             f'los 20 índices entre Golden Stores y el resto del universo">'
             f'<line x1="350" y1="10" x2="350" y2="{10+len(pf)*20}" stroke="#c5ced3"/>'
             f'{"".join(bd)}</svg>')

# --- importancia de variables, LAS 29
imp = AF["importancia"]
mi = max(d["caida_auc"] for d in imp)
bi = "".join(
    f'<rect x="250" y="{14+i*19}" width="{max(d["caida_auc"],0)/mi*380:.1f}" height="13" fill="#1f6f8b"/>'
    f'<text x="244" y="{25+i*19}" text-anchor="end" fill="#10171c" font-size="10.5" '
    f'font-family="sans-serif">{d["variable"]}</text>'
    f'<text x="{256+max(d["caida_auc"],0)/mi*380:.1f}" y="{25+i*19}" fill="#667680" font-size="9.5" '
    f'font-family="sans-serif">{d["caida_auc"]:.4f}</text>'
    for i, d in enumerate(imp))
svg_imp = (f'<svg viewBox="0 0 700 {14+len(imp)*19+8}" role="img" aria-label="Importancia por '
           f'permutación de las variables del modelo de afinidad">{bi}</svg>')

FECHA = datetime.now()

CSS = """
:root{--ink:#10171c;--body:#333f47;--muted:#667680;--line:#dfe5e8;--line2:#c5ced3;
--surface:#f7f9fa;--accent:#1f6f8b;--accent-soft:#e4f0f4;--ambar:#b8860b;--rojo:#b5382b}
*{box-sizing:border-box}
body{margin:0;padding:34px 20px 64px;background:#fff;color:var(--body);
font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;font-size:15.5px;line-height:1.6}
.doc{max-width:860px;margin:0 auto}
.sans{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
h1{margin:0 0 6px;font-size:30px;line-height:1.2;color:var(--ink);letter-spacing:-.012em;font-weight:600}
.sub{font-size:16.5px;color:var(--muted);margin-bottom:18px}
h2{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:36px 0 10px;font-size:12px;
font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);
padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{margin:22px 0 6px;font-size:16px;color:var(--ink);font-weight:600}
p{margin:0 0 11px}p:last-child{margin-bottom:0}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.88em;
background:var(--surface);padding:1px 5px;border-radius:3px}
strong{color:var(--ink)}
.eyebrow{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:11.5px;
font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin-bottom:8px}
.stamp{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:12px;
color:var(--muted);padding-top:14px;margin-top:4px;border-top:1px solid var(--line)}
.scope{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:16px 20px;
margin:18px 0 4px;font-size:14.5px}
.obs{border-left:3px solid var(--accent);background:var(--accent-soft);border-radius:0 5px 5px 0;
padding:13px 17px;font-size:14.5px;color:#0f4657;margin:14px 0}
.obs strong{color:#0c3a48}
.caution{border-left:3px solid var(--ambar);background:#fdf7e8;border-radius:0 5px 5px 0;
padding:13px 17px;font-size:14.5px;color:#6b4d05;margin:14px 0}
.caution strong{color:#7a5806}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(146px,1fr));gap:1px;
background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden;margin:14px 0}
.kpi{background:#fff;padding:13px 15px}
.kpi .v{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:21px;
font-weight:700;color:var(--ink);font-variant-numeric:tabular-nums;line-height:1.1}
.kpi .k{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:11.5px;
color:var(--muted);margin-top:3px;line-height:1.35}
.tablebox{overflow-x:auto;margin:12px 0;border:1px solid var(--line);border-radius:6px}
.tablebox.larga{max-height:420px;overflow-y:auto}
table{width:100%;border-collapse:collapse;font-size:13.5px;
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--line);vertical-align:top}
thead th{background:var(--surface);color:var(--ink);font-size:11px;letter-spacing:.05em;
text-transform:uppercase;font-weight:700;position:sticky;top:0}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
figure{margin:16px 0 0}
.chartbox{border:1px solid var(--line);border-radius:6px;padding:14px 12px 10px;background:#fff;overflow-x:auto}
.chartbox svg{display:block;width:100%;min-width:430px;height:auto}
figcaption{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:12.5px;
color:var(--muted);margin-top:8px;line-height:1.5}
ul.q{margin:0;padding-left:20px}ul.q li{margin-bottom:10px}
.src{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin-top:26px;
padding-top:14px;border-top:1px solid var(--line);font-size:11.5px;color:var(--muted);line-height:1.6}
@media(max-width:640px){body{padding:22px 14px 40px;font-size:15px}h1{font-size:23px}}
@media print{@page{size:letter portrait;margin:15mm}
body{padding:0;font-size:10pt}.doc{max-width:none}
.obs,.caution,.scope,table,figure,.kpis{page-break-inside:avoid}
h1,h2,h3{page-break-after:avoid}
.tablebox.larga{max-height:none;overflow:visible}
thead th{position:static}}
"""

HTML = f"""<title>EDA Universo Sueros</title>
<style>{CSS}</style>
<div class="doc">

<div class="eyebrow">Análisis exploratorio de datos</div>
<h1>Universo de puntos de venta para la categoría de Sueros</h1>
<p class="sub">Caracterización completa de los {N:,} PDV en México</p>
<div class="stamp sans">Fuentes Golden Stores FY&rsquo;23 (NielsenIQ · Spectra) y top ciudades
Scantrack · {FECHA:%d de %B de %Y}</div>

<div class="scope">
<p><strong>Qué es este documento.</strong> Un informe de exploración: describe qué contienen los
datos, qué estructura tienen y qué patrones se observan. Registra hallazgos y mediciones, no
recomendaciones.</p>
<p><strong>Sin muestras ni recortes.</strong> Todos los cálculos usan los {N:,} PDV. Las tablas
reportan el universo completo de cada corte &mdash; las {len(R['lift_cadena'])} cadenas, las
{len(R['lift_zm'])} zonas metropolitanas, los {R['meta']['indices']} índices, los
{R['dependencias']['n_pares']} pares de correlación y las {len(AF['importancia'])} variables del
modelo &mdash; no los primeros N.</p>
<p style="margin-bottom:0"><strong>Qué no es.</strong> No propone un plan de activación, no prioriza
inversión ni fija fechas. Las observaciones apuntan a decisiones posibles, pero tomarlas exige
validaciones que aún no se han hecho; quedan listadas al final como preguntas abiertas.</p>
</div>

<section>
<h2>1 · Datos, grano e integridad</h2>
<p>Se consolidaron tres libros Golden Stores FY&rsquo;23 con el mismo layout: encabezado de dos
niveles, {R['meta']['indices']} bloques de perfil del área transaccional (<code>HHs</code>,
<code>%</code>, <code>Índice</code>) y atributos de ubicación y cadena por tienda.</p>

{tabla([("fuente","Fuente"),("canal","Canal"),("pdv","PDV")],
       I["por_fuente"] + [{"fuente":"<strong>Consolidado</strong>","canal":"2 canales",
                           "pdv":f"<strong>{N:,}</strong>"}], num=("pdv",))}

<div class="kpis">
<div class="kpi"><div class="v">{I['nielsen_id_unicos']:,}</div><div class="k">Nielsen ID únicos sobre {I['filas']:,} filas · {I['duplicados']} duplicados</div></div>
<div class="kpi"><div class="v">{I['geo_nulos']}</div><div class="k">coordenadas nulas · {I['geo_fuera_bbox']} fuera del recuadro de México</div></div>
<div class="kpi"><div class="v">{I['indices_nulos']}</div><div class="k">valores faltantes en los {R['meta']['indices']} índices</div></div>
<div class="kpi"><div class="v">{I['cadenas']} · {I['zonas_metropolitanas']}</div><div class="k">cadenas y zonas metropolitanas</div></div>
</div>

<p>El grano declarado es <strong>una fila = un punto de venta</strong> y se comprobó en lugar de
asumirse. Los faltantes que sí aparecen son <strong>estructurales, no errores</strong>: la línea
<em>Electrolit + Suerox</em> está vacía en el libro de Hidratación R. Norte porque ese libro reporta
Agua como target, y <em>Saturación de farmacias independientes</em> sólo existe en el libro de
farmacias. Las colonias en blanco corresponden a plazas y centros comerciales sin colonia catastral.</p>
</section>

<section>
<h2>2 · Distribución geográfica</h2>
<p>Los {N:,} PDV tienen coordenadas completas y cubren {MAPA.Estado.nunique()} estados. El mapa
muestra cada tienda coloreada por su semáforo de Sueros.</p>

<figure><div class="chartbox">{svg_pdv}</div>
<figcaption>Los {N:,} puntos de venta del universo, por semáforo de la línea T. Sueros. Proyección
equirectangular con corrección por coseno de la latitud media. En zonas densas los puntos se
superponen: el rojo se dibuja encima para que no quede oculto, así que la mancha roja de las grandes
metrópolis se lee mejor de lo que su proporción real sugiere &mdash; para la proporción, ver el
mapa siguiente.</figcaption></figure>

<figure><div class="chartbox">{svg_zm}</div>
<figcaption>Las {len(gzm)} zonas metropolitanas con PDV en el universo. El tamaño de la burbuja es el
número de tiendas; el color, la proporción de ellas con Sueros en rojo. Se excluye la categoría
NON-METRO, que agrupa {int((MAPA['Zonas Metropolitanas']=='NON-METRO').sum()):,} PDV dispersos y no
tiene un centroide con significado.</figcaption></figure>

<div class="obs">
<strong>Lo que muestra el mapa.</strong> La cobertura del universo es nacional, no regional: hay PDV
en los {MAPA.Estado.nunique()} estados. La densidad sigue el eje urbano habitual &mdash; Valle de
México, Bajío, la franja fronteriza norte y la península de Yucatán &mdash; y el semáforo rojo no se
concentra en ninguna región en particular, sino que aparece en proporciones parecidas en plazas de
tamaños muy distintos. Esa dispersión geográfica es coherente con la observación de la sección 6.
</div>

{tabla([("zm","Zona metropolitana"),("pdv","PDV"),("rojo","% en rojo")],
       [{"zm": zm.title(), "pdv": f"{int(r.pdv):,}", "rojo": f"{r.rojo*100:.1f}%"}
        for zm, r in gzm.iterrows()], num=("pdv","rojo"), clase="larga")}
<figcaption class="sans">Las {len(gzm)} zonas metropolitanas, ordenadas por número de PDV. Tabla completa.</figcaption>
</section>

<section>
<h2>3 · Perfil del área transaccional</h2>
<p>Los índices Spectra se construyen con base 100 = promedio nacional: un índice de 130 en
<code>A/B</code> indica 30% más hogares de ese nivel socioeconómico que el promedio del país. Ese
anclaje los hace comparables entre plazas.</p>

<div class="kpis">
<div class="kpi"><div class="v">{R['meta']['indices']}</div><div class="k">índices, sin valores faltantes</div></div>
<div class="kpi"><div class="v">{R['univariado_resumen']['cola_pesada']} / {R['meta']['indices']}</div><div class="k">variables con cola pesada (|sesgo| &gt; 1 o curtosis &gt; 3)</div></div>
<div class="kpi"><div class="v">&alpha; &ge; {R['univariado_resumen']['alpha_min']}</div><div class="k">índice de Hill mínimo: ninguna cola de varianza infinita</div></div>
</div>

{tabla([("variable","Índice"),("media","Media"),("mediana","Mediana"),("desv","Desv."),
        ("p5","P5"),("p95","P95"),("sesgo","Sesgo"),("curtosis","Curtosis"),("hill_alpha","α Hill")],
       R["univariado"], num=("media","mediana","desv","p5","p95","sesgo","curtosis","hill_alpha"))}
<figcaption class="sans">Los {R['meta']['indices']} índices, tabla completa. n = {N:,} en todos.</figcaption>

<p style="margin-top:12px">Las distribuciones no son gaussianas, pero tampoco patológicas: sólo
{R['univariado_resumen']['cola_pesada']} de {R['meta']['indices']} muestran cola pesada, y el
estimador de Hill sobre el decil superior deja a todas por encima de <strong>&alpha; = 2</strong>, de
modo que las medias siguen siendo resúmenes utilizables. Aun así, los cortes de segmentación se
hicieron sobre percentiles: el sesgo basta para desplazar un promedio sin desplazar la mediana.</p>
</section>

<section>
<h2>4 · Estructura de dependencias</h2>
<p>Con {R['meta']['indices']} índices se evalúan <strong>{R['dependencias']['n_pares']} pares</strong>.
A ese volumen, los p-values crudos producen una decena de falsos positivos por azar, así que se aplicó
control de tasa de descubrimientos falsos por Benjamini-Hochberg:
<strong>{R['dependencias']['significativos']} de {R['dependencias']['n_pares']}</strong> resultan
significativos con q &lt; 0.05.</p>

{tabla([("a","Variable A"),("b","Variable B"),("pearson","Pearson"),("spearman","Spearman"),("q","q BH")],
       [{**p, "q": f'{p["q_BH"]:.2e}' if p["q_BH"] < 0.001 else f'{p["q_BH"]:.3f}'}
        for p in R["dependencias"]["pares"]],
       num=("pearson","spearman","q"), clase="larga")}
<figcaption class="sans">Los {R['dependencias']['n_pares']} pares, ordenados por |Spearman|. Tabla completa.</figcaption>

<h3>Dimensionalidad real</h3>
<p>El análisis de componentes principales muestra que <strong>{R['pca']['k90']} componentes explican
el 90% de la varianza</strong>: los {R['meta']['indices']} índices describen unos
{R['pca']['k90']} ejes independientes de entorno, no veinte.</p>

{tabla([("pc","Componente"),("var","Varianza"),("acum","Acumulada"),("topx","Variables de mayor peso")],
       [{**c, "var": f'{c["var"]}%', "acum": f'{c["acum"]}%', "topx": ", ".join(c["top"])}
        for c in R["pca"]["componentes"]], num=("var","acum"), clase="larga")}
<figcaption class="sans">Los {len(R['pca']['componentes'])} componentes, tabla completa.</figcaption>

<h3>Entornos atípicos</h3>
<p>La distancia de Mahalanobis robusta contra el corte &chi;²(gl = {R['meta']['indices']}, 0.999) =
{R['atipicos']['corte_chi2']} identifica <strong>{R['atipicos']['n']:,} PDV, el
{R['atipicos']['pct']}% del universo</strong>, con perfil de entorno atípico.</p>
<div class="obs"><strong>No se eliminaron.</strong> En retail un atípico multivariado suele ser una
tienda real y valiosa &mdash; aeropuertos, centrales de abasto, plazas turísticas &mdash; y no un
error de captura. Se marcaron para que cualquier ranking posterior sea auditable, y se conservaron.
Descartarlos sesgaría los resultados justo en las tiendas de mayor tráfico.
<em>{R['atipicos']['nota']}</em></div>
</section>

<section>
<h2>5 · La segmentación vigente</h2>
<p>El clúster viene calculado en la fuente y cruza dos ejes: <strong>demanda potencial</strong>
(perfil sociodemográfico del área transaccional, compras del hogar, competidores, tiempo de manejo
&mdash; Spectra) contra <strong>ventas valor</strong> de la categoría (Scantrack). La primera letra
es demanda; la segunda, venta.</p>

{tabla([("cluster","Clúster"),("desc","Definición"),("pdv","PDV"),("pct","% del universo")],
       [{**c, "pdv": f'{c["pdv"]:,}', "pct": f'{c["pct"]}%',
         "desc": {"HH":"Demanda alta · venta alta","HL":"Demanda alta · venta baja",
                  "LH":"Demanda baja · venta alta","LL":"Demanda baja · venta baja"}[c["cluster"]]}
        for c in sorted(R["clusters"], key=lambda d: d["cluster"])], num=("pdv","pct"))}

<h3>Validación del eje de demanda</h3>
<p>Antes de apoyarse en el clúster convenía comprobar que su eje de demanda es coherente con los datos
socio-demográficos de la propia tabla. Contrastando los {R['meta']['indices']} índices entre demanda
alta y baja con Mann-Whitney y control BH, <strong>los
{R['validacion_demanda']['significativos']} de {R['validacion_demanda']['total']} separan
significativamente</strong> (q &lt; 0.05). El eje es internamente consistente.</p>
</section>

<section>
<h2>6 · Observación central: el clúster y el semáforo de Sueros</h2>
<p>Al cruzar el clúster contra el semáforo específico de la línea T. Sueros aparece el patrón más
llamativo del ejercicio: <strong>la distribución del semáforo es prácticamente la misma en los cuatro
cuadrantes</strong>. Si el clúster capturara el desempeño de la categoría, la franja roja debería
encogerse al subir de LL hacia HH. No lo hace.</p>

<figure><div class="chartbox">{svg_sem}</div>
<figcaption>Distribución del semáforo T. Sueros dentro de cada clúster, en porcentaje de fila.
n = {N:,} PDV.</figcaption></figure>

<div class="kpis">
<div class="kpi"><div class="v">{CS['cramers_v']}</div><div class="k">Cramér&rsquo;s V entre clúster y semáforo</div></div>
<div class="kpi"><div class="v">{CS['ic95'][0]}&ndash;{CS['ic95'][1]}</div><div class="k">IC 95% por bootstrap ({CS['n_bootstrap']:,} remuestreos)</div></div>
<div class="kpi"><div class="v">{CS['tabla'][0]['ROJO']}% vs {CS['tabla'][3]['ROJO']}%</div><div class="k">tiendas en rojo: clúster HH contra LL</div></div>
<div class="kpi"><div class="v">{CS['hh_rojo']:,}</div><div class="k">Golden Stores con Sueros en rojo · {CS['hh_rojo']/PG['n_golden']*100:.0f}% de ellas</div></div>
</div>

<div class="obs"><strong>Cómo leer este número.</strong> Cramér&rsquo;s V = {CS['cramers_v']} es
estadísticamente distinto de cero, pero eso sólo refleja el tamaño de muestra: con {N:,}
observaciones casi cualquier asociación cruza el umbral. Lo informativo es la magnitud, y
{CS['cramers_v']} es despreciable. El intervalo por bootstrap
({CS['ic95'][0]}&ndash;{CS['ic95'][1]}) descarta que se trate de falta de potencia.
<strong>En los datos observados, el clúster Golden Stores no informa sobre el desempeño de
Sueros.</strong></div>

<div class="caution"><strong>Qué queda sin responder.</strong> Es una asociación transversal en un
solo corte temporal. No dice por qué la relación es débil: podría ser que la categoría responda a
factores distintos de la canasta, que el semáforo mida algo no comparable con el eje de venta del
clúster, o que el corte FY&rsquo;23 no sea representativo. Separar esas explicaciones requiere venta
valor por tienda y al menos dos periodos, que no están en las fuentes.</div>
</section>

<section>
<h2>7 · Qué caracteriza a las Golden Stores</h2>
<p>Sobre las {PG['n_golden']:,} tiendas del cuadrante HH ({PG['base_pct']}% del universo) se contrastó
cada índice contra el resto, midiendo el tamaño de efecto con la d de Cohen.</p>

<figure><div class="chartbox">{svg_cohen}</div>
<figcaption>d de Cohen de los {len(pf)} índices, Golden Stores contra el resto. Positivo = sobre-indexa
en Golden Stores. Gráfica completa, sin recorte.</figcaption></figure>

{tabla([("variable","Índice"),("golden","Golden"),("resto","Resto"),("d","Cohen d"),("q","q BH")],
       [{**p, "q": f'{p["q_BH"]:.2e}' if p["q_BH"] < 0.001 else f'{p["q_BH"]:.3f}'} for p in pf],
       num=("golden","resto","d","q"))}

<p style="margin-top:12px">El perfil es nítido y coherente: nivel socioeconómico medio-alto, ama de
casa madura, hogar de cuatro integrantes, y ausencia marcada de hogares D/E y de amas de casa jóvenes
con niños pequeños.</p>

<h3>Concentración por cadena</h3>
{tabla([("nombre","Cadena"),("pdv","PDV"),("golden","Golden"),("pct","% Golden"),("lift","Lift")],
       [{**c, "pdv": f'{c["pdv"]:,}', "golden": f'{c["golden"]:,}', "pct": f'{c["pct"]}%',
         "lift": f'{c["lift"]}×'} for c in R["lift_cadena"]],
       num=("pdv","golden","pct","lift"), clase="larga")}
<figcaption class="sans">Las {len(R['lift_cadena'])} cadenas del universo, tabla completa. Lift contra
la base nacional de {PG['base_pct']}%.</figcaption>

<h3>Concentración por zona metropolitana</h3>
{tabla([("nombre","Zona metropolitana"),("pdv","PDV"),("golden","Golden"),("pct","% Golden"),("lift","Lift")],
       [{**c, "pdv": f'{c["pdv"]:,}', "golden": f'{c["golden"]:,}', "pct": f'{c["pct"]}%',
         "lift": f'{c["lift"]}×'} for c in R["lift_zm"]],
       num=("pdv","golden","pct","lift"), clase="larga")}
<figcaption class="sans">Las {len(R['lift_zm'])} zonas metropolitanas, tabla completa.</figcaption>
</section>

<section>
<h2>8 · Extrapolación del perfil al universo</h2>
<p>Para medir cuánto del perfil es extrapolable se ajustó un modelo de propensión que aprende el
entorno y el perfil comercial de las Golden Stores y puntúa los {N:,} PDV. Las variables de clúster y
semáforo se excluyeron de las predictoras: el objetivo se deriva del clúster, así que incluirlas
produciría un ajuste perfecto y sin significado.</p>

{tabla([("modelo","Especificación"),("auc_roc","AUC-ROC"),("auc_pr","AUC-PR")],
       AF["comparativa"], num=("auc_roc","auc_pr"))}
<figcaption class="sans">Validación cruzada estratificada de 5 pliegues sobre los {N:,} PDV.
Base del universo: {AF['base_pct']}% de Golden Stores. Modelo elegido: {AF['elegido']}.</figcaption>

<p style="margin-top:12px">El decil superior del score concentra
<strong>{AF['decil_superior_pct']}% de Golden Stores contra {AF['base_pct']}% en el universo</strong>,
un lift de {AF['lift_decil']}×.</p>

<figure><div class="chartbox">{svg_imp}</div>
<figcaption>Caída en AUC al permutar cada variable. Medida <strong>por pliegue de validación cruzada
y promediada</strong>, de modo que los {AF['cobertura_importancia']['pdv_evaluados']:,} PDV
({AF['cobertura_importancia']['pct']}%) entran exactamente una vez como dato retenido. Las
{len(imp)} variables, sin recorte.</figcaption></figure>

<div class="obs"><strong>El formato comercial pesa más que la demografía.</strong> Las dos variables
con mayor caída de AUC son <strong>cadena</strong> ({imp[0]['caida_auc']:.4f}) y
<strong>canal</strong> ({imp[1]['caida_auc']:.4f}), por encima de cualquier índice sociodemográfico.
Dicho de otro modo: saber en qué cadena y formato opera una tienda informa más sobre su condición de
Golden Store que el perfil de hogares de su área transaccional. La demografía discrimina
&mdash;la sección 7 lo muestra&mdash; pero el formato discrimina más.</div>

<div class="caution"><strong>Qué mide y qué no mide este AUC.</strong> El modelo predice la condición
de Golden Store, que se define por demanda <em>y venta de la canasta</em>. No predice desempeño de
Sueros &mdash; la sección 6 muestra que ambas cosas están poco relacionadas. Un score alto identifica
tiendas cuyo entorno y formato se parecen a los de una Golden Store; verificar si además venden
Sueros exige medición posterior.</div>
</section>

<section>
<h2>9 · Capa geográfica Scantrack</h2>
<p>El archivo de top ciudades aporta el mix de Autoservicios por plaza y el embotellador responsable.
Se enlazaron <strong>{SC['enlazadas']} de {SC['ciudades_archivo']} ciudades</strong> a zonas
metropolitanas del universo Nielsen, cubriendo el <strong>{SC['pct_mix_mapeado']}% del mix de
Autoservicios</strong> y {SC['pdv_cubiertos']:,} PDV. El puente entre nomenclaturas se hizo explícito,
ciudad por ciudad: un emparejamiento aproximado automático es el tipo de error que después nadie audita.</p>

{tabla([("ciudad","Ciudad Scantrack"),("embotellador","Embotellador"),("zm","Zona metropolitana"),
        ("mix_as","Mix AS"),("peso","Peso relativo")],
       [{**c, "mix_as": f'{c["mix_as"]}%', "zm": c["zm"].title()} for c in SC["detalle"]],
       num=("mix_as","peso"), clase="larga")}
<figcaption class="sans">Las {SC['enlazadas']} ciudades enlazadas, tabla completa. Sin enlace quedaron
{len(SC['sin_enlace'])}: {", ".join(SC['sin_enlace'])}.</figcaption>

<div class="caution"><strong>Supuesto sin verificar.</strong> El archivo reporta &ldquo;Mix AS&rdquo;
sin especificar si corresponde a la categoría de Sueros o a la canasta RRS completa. La diferencia es
material: el peso relativo de plaza es lo que más separa a las ciudades pequeñas de las grandes en
cualquier comparación, y si el mix es de RRS total entonces mide relevancia de bebidas en general.
Conviene confirmarlo con el equipo de Scantrack antes de usar esta capa para comparar plazas.</div>
</section>

<section>
<h2>10 · Score de potencial a escala nacional</h2>
<p>Combinando afinidad ({R['meta']['pesos']['afinidad']:.0%}), brecha de desempeño ({R['meta']['pesos']['brecha']:.0%}) y peso de plaza
({R['meta']['pesos']['ciudad']:.0%}) en una media geométrica ponderada se obtiene un score que ordena <strong>los {N:,} PDV</strong>,
no un subconjunto. Los tramos son percentiles del universo completo.</p>

{tabla([("tramo","Tramo"),("pdv","PDV"),("ias_medio","Afinidad media"),("brecha_media","Brecha media"),
        ("pct_golden","% hoy Golden")],
       [{**t, "pdv": f'{t["pdv"]:,}', "pct_golden": f'{t["pct_golden"]}%'} for t in PN["tramos"]],
       num=("pdv","ias_medio","brecha_media","pct_golden"))}

<h3>Sensibilidad a la ponderación</h3>
<p>Los pesos son una elección, no un resultado estadístico. Para saber cuánto dependen de ella los
resultados se recalculó el score bajo cinco esquemas y se comparó <strong>el ranking completo de los
{SE['n_evaluados']:,} PDV</strong>, no un top-N.</p>

{tabla([("escenario","Esquema de pesos"),("spearman_ranking_completo","Spearman"),
        ("kendall_ranking_completo","Kendall τ"),("traslape_p1_pct","Traslape tramo P1")],
       [{**e, "traslape_p1_pct": f'{e["traslape_p1_pct"]}%'} for e in SE["escenarios"]],
       num=("spearman_ranking_completo","kendall_ranking_completo","traslape_p1_pct"))}
<figcaption class="sans">{SE['nota']} Tramo P1 = {SE['tramo_p1']:,} PDV.</figcaption>

<div class="caution"><strong>El ranking es menos estable de lo que sugería el corte anterior.</strong>
Al comparar el universo completo en vez del top-500, el traslape mínimo del tramo P1 baja a
<strong>{min(e['traslape_p1_pct'] for e in SE['escenarios'][1:])}%</strong> y la correlación de
Kendall a {min(e['kendall_ranking_completo'] for e in SE['escenarios'][1:])}. Es decir: cambiar la
ponderación hacia afinidad pura reordena alrededor de una cuarta parte del tramo prioritario. Los
pesos no son un detalle de implementación &mdash; son una decisión de negocio con efecto medible.</div>

<h3>Corte por las tres plazas del encargo</h3>
{tabla([("nombre","Operación"),("pdv","PDV"),("pct_universo","% universo"),("farmacias_pct","% farmacias"),
        ("golden_pct","% Golden"),("hh_rojo","Golden en rojo"),("peso_ciudad","Peso plaza"),
        ("p1_nacional","P1 nacional")],
       [{**p, "pdv": f'{p["pdv"]:,}', "pct_universo": f'{p["pct_universo"]}%',
         "farmacias_pct": f'{p["farmacias_pct"]}%', "golden_pct": f'{p["golden_pct"]}%'}
        for p in R["plazas"]],
       num=("pdv","pct_universo","farmacias_pct","golden_pct","hh_rojo","peso_ciudad","p1_nacional"))}
<figcaption class="sans">Las tres plazas suman {R['plazas_total']['pdv']:,} PDV,
{R['plazas_total']['pct_universo']}% del universo. La columna P1 nacional cuenta cuántas de sus
tiendas caen en el decil superior del ranking de los {N:,}.</figcaption>

<div class="obs"><strong>El peso de plaza domina el corte nacional.</strong> Mérida tiene
{[p for p in R['plazas'] if p['zm']=='MERIDA'][0]['hh_rojo']} Golden Stores en rojo, pero sólo
{[p for p in R['plazas'] if p['zm']=='MERIDA'][0]['p1_nacional']} de sus tiendas entran al decil
superior nacional, porque su peso de plaza es {[p for p in R['plazas'] if p['zm']=='MERIDA'][0]['peso_ciudad']}
contra 1.00 del Valle de México. La comparación entre plazas está gobernada por el ponderador
geográfico, que es justamente el componente cuyo origen queda sin verificar (sección 9). Dentro de
cada plaza el ponderador es constante y no altera el orden.</div>
</section>

<section>
<h2>11 · Límites del dato disponible</h2>
<p>Las fuentes son un <strong>corte transversal de tienda por entorno</strong>, no un registro
transaccional. Eso acota de forma dura qué análisis son posibles.</p>

{tabla([("a","Análisis"),("b","Estado"),("c","Qué haría falta")],
 [{"a":"Perfilado socio-demográfico, segmentación, propensión","b":"Realizado","c":"—"},
  {"a":"Distribución geográfica y densidad","b":"Realizado","c":"—"},
  {"a":"Elasticidad-precio","b":"No posible","c":"Precio y cantidad por tienda y periodo"},
  {"a":"Canasta de compra y reglas de asociación","b":"No posible","c":"Tickets con líneas de detalle"},
  {"a":"Estacionalidad y descomposición temporal","b":"No posible","c":"Series de venta de al menos dos años"},
  {"a":"Efecto causal (diferencia-en-diferencias)","b":"No posible","c":"Ventana pre/post y grupo de control"},
  {"a":"Brecha de venta como escala continua","b":"Sólo ordinal","c":"Venta valor por tienda; hoy son tres niveles de semáforo"}])}

<p>El semáforo es la única señal disponible del desempeño de la categoría a nivel tienda, y tiene tres
escalones. Toda distinción de desempeño en este informe hereda esa resolución.</p>
</section>

<section>
<h2>12 · Preguntas abiertas</h2>
<p>El análisis deja estas preguntas sin resolver. Ninguna se contesta con los datos actuales, y todas
condicionan cómo debería interpretarse lo anterior.</p>
<ul class="q">
<li><strong>¿El &ldquo;Mix AS&rdquo; de Scantrack es de Sueros o de la canasta RRS?</strong> Es la
verificación más barata y la de mayor consecuencia: la sección 10 muestra que ese peso gobierna la
comparación entre plazas.</li>
<li><strong>¿Por qué el clúster no informa sobre el desempeño de Sueros?</strong> Las explicaciones
candidatas &mdash; drivers distintos de los de la canasta, incomparabilidad entre semáforo y eje de
venta, o un corte no representativo &mdash; llevan a lecturas muy distintas y requieren al menos dos
periodos para separarse.</li>
<li><strong>¿Por qué cadena y canal pesan más que la demografía?</strong> La sección 8 lo mide pero no
lo explica. Puede reflejar decisiones de surtido y negociación por cadena, o que el formato ya recoge
la demografía de donde cada cadena decide abrir. Distinguirlo cambiaría dónde buscar la palanca.</li>
<li><strong>¿Sigue siendo válida la definición de Golden Store para esta categoría?</strong> El
cuadrante HH se define por venta de la canasta. Si el objetivo es Sueros, habría que evaluar un
criterio propio en lugar de heredar el general.</li>
<li><strong>¿Qué tan vigente es el área transaccional Spectra?</strong> Los polígonos son de
FY&rsquo;23. Aperturas, cierres y remodelaciones de competencia los alteran, y todo el perfil
demográfico depende de ellos.</li>
<li><strong>¿Existe sell-out por tienda?</strong> Es la pieza que convertiría la brecha de tres
escalones en una escala continua y habilitaría casi todos los análisis marcados como no posibles.</li>
</ul>
</section>

<p class="src">
Base analítica: los {N:,} PDV de los libros Golden Stores Hidratación R. Norte, Sueros Pharma Nacional
y Sueros R. Sur, FY&rsquo;23 (NielsenIQ · Spectra), más el archivo de top ciudades industria RRS MX
(Scantrack). Sin muestreo en ninguna etapa: la importancia de variables se mide por pliegue de
validación cruzada cubriendo el 100% del universo, y la sensibilidad compara rankings completos.
Reproducible con <code>scripts/eda_universo_completo.py</code> y
<code>scripts/render_informe_eda.py</code>, semilla fija {R['meta']['semilla']}.
</p>

</div>
"""

out = REP / f"informe_eda_universo_{FECHA:%Y%m%d_%H%M}.html"
frag = REP / "_informe_fragmento.html"
frag.write_text(HTML, encoding="utf-8")
i = HTML.index('<div class="doc">')
out.write_text('<!DOCTYPE html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
               '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
               f'{HTML[:i].strip()}\n</head>\n<body>\n{HTML[i:].strip()}\n</body>\n</html>\n',
               encoding="utf-8")
print(f"{out}  ({out.stat().st_size/1024:.0f} KB)")
print(f"  mapa PDV      : {len(MAPA):,} puntos")
print(f"  mapa ZM       : {len(gzm)} burbujas")
print(f"  tablas completas: {len(R['lift_cadena'])} cadenas · {len(R['lift_zm'])} ZM · "
      f"{R['dependencias']['n_pares']} pares · {len(imp)} variables")
