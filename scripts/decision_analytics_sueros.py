#!/usr/bin/env python3
"""Motor analitico para el brief ejecutivo de Sueros.

Implementa executive-decision-analytics sobre la salida de 01_potencial_pdv_sueros.ipynb:

  1. Descomposicion de varianza del score (analogo exacto de PVM, log-aditivo).
  2. Contrafactual: priorizacion vigente por cluster vs. modelo especifico de Sueros.
  3. Monte Carlo / EMV sobre las tres opciones de decision.
  4. Tornado de sensibilidad (OAT, p10-p90) y umbral de conmutacion.
  5. Costo de demora semanal.

BASE DE COMPARACION: el contrafactual es la priorizacion vigente, que ordena por cluster
Golden Stores. Todo "uplift" de este script es contra ese metodo, no contra cero.

CONVENCION: no existe venta valor por tienda en las fuentes, asi que el analisis no se
expresa en moneda. La unidad del EMV es PDV-semana convertidos durante el ano 1: tiendas
que pasan de semaforo ROJO a AMARILLO o VERDE, por las semanas que sostienen ese estado.
Es una unidad observable y verificable despues de ejecutar.

    uv run python scripts/decision_analytics_sueros.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parent.parent
XLSX = RAIZ / "outputs/priorizacion_sueros_pdv.xlsx"
SALIDA = RAIZ / "outputs/report/decision_analytics_sueros.json"

SEMILLA = 42
N_SIM = 10_000
HORIZONTE_SEM = 52

# Pesos del score, identicos a los del notebook (media geometrica ponderada)
W = {"afinidad": 0.50, "brecha": 0.30, "ciudad": 0.20}

# ---------------------------------------------------------------------------
# SUPUESTOS. Beta-PERT (min, mas probable, max). NO salen de los datos: son
# rangos de juicio operativo, editables. El tornado dice cual de ellos importa.
# ---------------------------------------------------------------------------
SUPUESTOS = {
    "tasa_conversion": {
        "rango": (0.10, 0.25, 0.45),
        "desc": "Fraccion de PDV activados que pasa de semaforo ROJO a AMARILLO o VERDE",
    },
    "cobertura_ejecucion": {
        "rango": (0.75, 0.90, 0.98),
        "desc": "Fraccion de PDV del plan efectivamente visitados y ejecutados",
    },
    "sostenibilidad": {
        "rango": (0.55, 0.80, 0.95),
        "desc": "Fraccion de la conversion que se sostiene sin re-visita en el horizonte",
    },
    "penalizacion_escala": {
        "rango": (0.80, 0.92, 1.00),
        "desc": "Castigo a la calidad de ejecucion por desplegar 162 PDV en vez de 50",
    },
}

OPCIONES = {
    "opcion_1": {"nombre": "Activar los 162 P1", "pdv": 162, "arranque_sem": 8, "escala": True},
    "opcion_2": {"nombre": "Piloto GDL + Merida", "pdv": 50, "arranque_sem": 3, "escala": False,
                 "segunda_ola": {"pdv": 112, "arranque_sem": 21}},
    "status_quo": {"nombre": "Priorizar por cluster", "pdv": 0, "arranque_sem": 0, "escala": False},
}


# ===========================================================================
# Utilidades estocasticas
# ===========================================================================
def beta_pert(rng, minimo, moda, maximo, n, lamb=4.0):
    """Beta-PERT: Beta reescalada al rango, con la masa concentrada en la moda."""
    if maximo <= minimo:
        return np.full(n, moda, float)
    media = (minimo + lamb * moda + maximo) / (lamb + 2)
    if np.isclose(media, moda):
        a = b = 3.0
    else:
        a = ((media - minimo) * (2 * moda - minimo - maximo)) / ((moda - media) * (maximo - minimo))
        b = a * (maximo - media) / (media - minimo)
    a, b = max(a, 1e-3), max(b, 1e-3)
    return minimo + rng.beta(a, b, n) * (maximo - minimo)


def percentiles(x):
    return {
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "media_emv": float(np.mean(x)),
        "var95": float(np.percentile(x, 5)),                    # cola inferior: escenario adverso
        "cvar95": float(np.mean(x[x <= np.percentile(x, 5)])),  # perdida esperada en esa cola
    }


# ===========================================================================
# 1 · Descomposicion de varianza del score (analogo PVM)
# ===========================================================================
def descomposicion(piloto):
    """El score es geometrico ponderado, asi que su log es exactamente aditivo.

        log(potencial) = w_a*log(afinidad) + w_b*log(brecha) + w_c*log(ciudad)

    Eso permite repartir el score entre sus tres drivers sin residuo, igual que
    un PVM reparte una varianza entre precio, volumen y mezcla.
    """
    d = piloto.copy()
    comp = {"afinidad": d.IAS / 100, "brecha": d.brecha, "ciudad": d.peso_ciudad}
    logs = {k: W[k] * np.log(np.clip(v, 1e-3, None)) for k, v in comp.items()}
    total = sum(logs.values())

    # El log es negativo (componentes < 1); se reparte sobre la magnitud.
    aporte = {k: (v / total).replace([np.inf, -np.inf], np.nan) for k, v in logs.items()}
    res = {}
    for plaza, idx in d.groupby("operacion").groups.items():
        sub = d.loc[idx]
        p1 = sub[sub.prioridad_local == "P1"]
        res[plaza] = {
            "pdv": int(len(sub)),
            "p1": int(len(p1)),
            "aporte_p1": {k: round(float(aporte[k].loc[p1.index].mean() * 100), 1) for k in W},
            "reconcilia": round(float(sum(aporte[k].loc[p1.index].mean() for k in W)), 6),
        }
    glob = {k: round(float(aporte[k][d.prioridad_local == "P1"].mean() * 100), 1) for k in W}
    return {"por_plaza": res, "global_p1": glob,
            "reconcilia_global": round(sum(glob.values()), 2)}


# ===========================================================================
# 2 · Contrafactual: priorizacion vigente vs. modelo de Sueros
# ===========================================================================
def contrafactual(univ, piloto, rng):
    """Cuantas de las Golden-en-rojo encuentra cada metodo con el mismo presupuesto de visitas."""
    objetivo = univ.Cluster.eq("HH") & univ.sem_sueros.eq("ROJO")
    n_obj = int(objetivo.sum())

    # Asociacion cluster x semaforo: si el cluster predijera Sueros, V seria alto.
    ct = pd.crosstab(univ.Cluster, univ.sem_sueros)
    chi2 = stats.chi2_contingency(ct).statistic
    V = float(np.sqrt(chi2 / len(univ) / (min(ct.shape) - 1)))

    # Bootstrap de V: una V baja sin intervalo no descarta falta de potencia.
    boot = []
    n = len(univ)
    cl, sm = univ.Cluster.to_numpy(), univ.sem_sueros.to_numpy()
    for _ in range(1000):
        i = rng.integers(0, n, n)
        c = pd.crosstab(pd.Series(cl[i]), pd.Series(sm[i]))
        if min(c.shape) < 2:
            continue
        boot.append(np.sqrt(stats.chi2_contingency(c).statistic / n / (min(c.shape) - 1)))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # Con presupuesto de N visitas en el piloto: cuantas Golden-en-rojo captura cada orden.
    p = piloto.copy()
    p["es_objetivo"] = p.Cluster.eq("HH") & p.sem_sueros.eq("ROJO")
    n_visitas = int((p.prioridad_local == "P1").sum())

    # Metodo vigente: ordenar por cluster (HH primero) y, dentro, sin criterio de Sueros.
    orden_cluster = {"HH": 0, "HL": 1, "LH": 2, "LL": 3}
    vigente = (p.assign(_k=p.Cluster.map(orden_cluster))
                 .sort_values(["_k", "Nielsen ID"]).head(n_visitas))
    modelo = p.nlargest(n_visitas, "potencial_local")

    cap_v, cap_m = int(vigente.es_objetivo.sum()), int(modelo.es_objetivo.sum())
    return {
        "golden_en_rojo_nacional": n_obj,
        "golden_en_rojo_piloto": int(p.es_objetivo.sum()),
        "cramers_v": round(V, 4),
        "cramers_v_ic95": [round(float(lo), 4), round(float(hi), 4)],
        "n_visitas": n_visitas,
        "captura_metodo_vigente": cap_v,
        "captura_modelo_sueros": cap_m,
        "uplift_pp": round((cap_m - cap_v) / n_visitas * 100, 1),
        "uplift_x": round(cap_m / cap_v, 2) if cap_v else None,
    }


# ===========================================================================
# 3 · Monte Carlo / EMV
# ===========================================================================
def simular(rng, n=N_SIM):
    par = {k: beta_pert(rng, *v["rango"], n) for k, v in SUPUESTOS.items()}
    return par, {k: evaluar(op, par) for k, op in OPCIONES.items()}


def evaluar(op, par):
    """PDV-semana convertidos en el ano 1. Unidad observable; el analisis no se monetiza."""
    if op["pdv"] == 0:
        return np.zeros(len(par["tasa_conversion"]))
    esc = par["penalizacion_escala"] if op["escala"] else 1.0
    base = par["tasa_conversion"] * par["cobertura_ejecucion"] * par["sostenibilidad"] * esc
    out = op["pdv"] * base * (HORIZONTE_SEM - op["arranque_sem"])
    if "segunda_ola" in op:
        o2 = op["segunda_ola"]
        out = out + o2["pdv"] * base * (HORIZONTE_SEM - o2["arranque_sem"])
    return out


# ===========================================================================
# 4 · Tornado (OAT) y umbral de conmutacion
# ===========================================================================
def tornado(rng, clave="opcion_1"):
    """Cada parametro se mueve de su p10 a su p90 con los demas fijos en la mediana."""
    med = {k: beta_pert(rng, *v["rango"], 20000) for k, v in SUPUESTOS.items()}
    medianas = {k: float(np.median(v)) for k, v in med.items()}
    op = OPCIONES[clave]

    def punto(over=None):
        p = {k: np.array([v]) for k, v in medianas.items()}
        if over:
            p[over[0]] = np.array([over[1]])
        return float(evaluar(op, p)[0])

    base = punto()
    filas = []
    for k, v in med.items():
        lo, hi = float(np.percentile(v, 10)), float(np.percentile(v, 90))
        a, b = punto((k, lo)), punto((k, hi))
        filas.append({"parametro": k, "desc": SUPUESTOS[k]["desc"],
                      "p10": round(lo, 3), "p90": round(hi, 3),
                      "salida_p10": round(a), "salida_p90": round(b),
                      "amplitud": round(abs(b - a))})
    tot = sum(f["amplitud"] for f in filas) or 1
    for f in filas:
        f["pct_varianza"] = round(f["amplitud"] / tot * 100, 1)
    filas.sort(key=lambda f: -f["amplitud"])
    return {"base_p50": round(base), "ranking": filas, "medianas": medianas}


def umbral_conmutacion(medianas):
    """Con que penalizacion de escala la Opcion 2 supera a la Opcion 1."""
    p = {k: np.array([v]) for k, v in medianas.items()}
    v2 = float(evaluar(OPCIONES["opcion_2"], p)[0])
    for esc in np.arange(1.0, 0.0, -0.001):
        p["penalizacion_escala"] = np.array([esc])
        if float(evaluar(OPCIONES["opcion_1"], p)[0]) < v2:
            return {"penalizacion_critica": round(float(esc), 3),
                    "rango_supuesto": SUPUESTOS["penalizacion_escala"]["rango"],
                    "dentro_del_rango": bool(esc >= SUPUESTOS["penalizacion_escala"]["rango"][0])}
    return {"penalizacion_critica": None, "dentro_del_rango": False}


# ===========================================================================
# 5 · Costo de demora
# ===========================================================================
def costo_demora(medianas):
    p = {k: np.array([v]) for k, v in medianas.items()}
    base = medianas["tasa_conversion"] * medianas["cobertura_ejecucion"] \
        * medianas["sostenibilidad"] * medianas["penalizacion_escala"]
    return {
        "pdv_semana_por_semana_de_demora": round(float(162 * base)),
        "por_mes": round(float(162 * base * 4.33)),
        "nota": "Cada semana de demora elimina una semana de horizonte para los 162 PDV.",
    }


# ===========================================================================
def main():
    rng = np.random.default_rng(SEMILLA)
    univ = pd.read_excel(XLSX, sheet_name="Universo nacional")
    piloto = pd.concat([
        pd.read_excel(XLSX, sheet_name=h).assign(operacion=n)
        for h, n in [("VALLE DE MEXICO", "KOF - Ciudad de Mexico"),
                     ("GUADALAJARA", "Arca - Guadalajara"),
                     ("MERIDA", "Bepensa - Merida")]
    ], ignore_index=True)

    par, sims = simular(rng)
    tor = tornado(rng)

    # Comparacion pareada: ambas opciones se evaluan sobre los MISMOS sorteos, asi que
    # la diferencia se lee por iteracion y no como resta de dos medias independientes.
    delta = sims["opcion_1"] - sims["opcion_2"]
    pareado = {
        "delta_p50": round(float(np.percentile(delta, 50))),
        "delta_p10": round(float(np.percentile(delta, 10))),
        "delta_p90": round(float(np.percentile(delta, 90))),
        "prob_opcion2_gana": round(float(np.mean(delta < 0) * 100), 1),
        "ventaja_relativa_pct": round(float(np.median(sims["opcion_1"]) /
                                            np.median(sims["opcion_2"]) - 1) * 100, 1),
    }

    res = {
        "meta": {
            "base_comparacion": "Priorizacion vigente por cluster Golden Stores",
            "unidad_emv": "PDV-semana convertidos en el ano 1",
            "n_simulaciones": N_SIM, "horizonte_semanas": HORIZONTE_SEM, "semilla": SEMILLA,
            "universo_pdv": int(len(univ)), "piloto_pdv": int(len(piloto)),
        },
        "supuestos": {k: {"min": v["rango"][0], "moda": v["rango"][1], "max": v["rango"][2],
                          "desc": v["desc"]} for k, v in SUPUESTOS.items()},
        "descomposicion": descomposicion(piloto),
        "contrafactual": contrafactual(univ, piloto, rng),
        "emv": {k: {"nombre": OPCIONES[k]["nombre"], **percentiles(v)} for k, v in sims.items()},
        "comparacion_pareada": pareado,
        "tornado": tor,
        "umbral": umbral_conmutacion(tor["medianas"]),
        "costo_demora": costo_demora(tor["medianas"]),
    }

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n-> {SALIDA}")


if __name__ == "__main__":
    main()
