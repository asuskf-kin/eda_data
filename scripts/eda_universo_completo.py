#!/usr/bin/env python3
"""EDA de la categoria Sueros sobre el UNIVERSO COMPLETO, sin muestras ni top-N.

Corrige tres recortes que tenia la version anterior:

  1. Importancia de variables: antes se medía sobre un holdout del 25%. Ahora se calcula
     por pliegue de la validación cruzada y se promedia, de modo que cada uno de los
     8,864 PDV entra exactamente una vez como dato retenido. Cobertura 100%.
  2. Sensibilidad a los pesos: antes comparaba el traslape del top-500. Ahora usa el
     ranking completo (correlacion de Spearman sobre 8,864) y el traslape del tramo de
     decision real (decil superior nacional).
  3. Cortes de reporte: todas las cadenas, todas las zonas metropolitanas, todos los
     indices y los 190 pares de correlacion. Ningun top-N.

La priorizacion se calcula a escala nacional sobre los 8,864 PDV; las tres plazas del
encargo se reportan como corte, no como universo.

    uv run python scripts/eda_universo_completo.py
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
SALIDA = RAIZ / "outputs/report/eda_universo_completo.json"

SEMILLA = 42
W = {"afinidad": 0.50, "brecha": 0.30, "ciudad": 0.20}
ESCALA_SEMAFORO = {"VERDE": 1.0, "AMARILLO": 0.5, "ROJO": 0.0}
PLAZAS = {"VALLE DE MEXICO": "KOF — Ciudad de México",
          "GUADALAJARA": "Arca Continental — Guadalajara",
          "MERIDA": "Bepensa — Mérida"}

FUENTES = [
    ("GS Hidratación R. Norte/Golden Stores Hidratación R.Norte - KO FY'23.xlsx",
     0, "GCA R.Norte", "Autoservicios"),
    ("GS Sueros Farmacias Nacional/Golden Stores Sueros Pharma Nacional - KO FY'23.xlsx",
     0, "T.Nacional Pharma", "Farmacias"),
    ("GS Sueros GCA R.Sur/Golden Stores Sueros R.Sur - KO FY'23.xlsx",
     0, "GCA R.Sur", "Autoservicios"),
]


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()


def leer_gs(ruta, hoja, fuente, canal, fila_grupo=8, fila_campo=9):
    wb = openpyxl.load_workbook(ruta, read_only=True, data_only=True)
    ws = wb.worksheets[hoja]
    g, h = [list(r) for r in ws.iter_rows(min_row=fila_grupo, max_row=fila_campo, values_only=True)]
    wb.close()
    ATRIB, cols, grupo = {"HHs", "%", "Indice"}, [], ""
    for a, b in zip(g, h):
        if a is not None:
            grupo = _norm(a)
        campo = _norm(b) if b is not None else ""
        cols.append(f"{grupo}|{campo}" if campo in ATRIB else campo)
    df = pd.read_excel(ruta, sheet_name=hoja, header=None, skiprows=fila_campo, names=cols)
    df = df.loc[:, [c for c in df.columns if c != ""]]
    df = df.rename(columns={c: "Cluster" for c in df.columns if c.startswith("Cluster")})
    df = df.rename(columns={c: "Cadena" for c in df.columns if c.startswith("Cadena de")})
    return df.dropna(subset=["Nielsen ID"]).assign(fuente=fuente, canal=canal)


def cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp else 0.0


def cramers_v(ct, n):
    return float(np.sqrt(stats.chi2_contingency(ct).statistic / n / (min(ct.shape) - 1)))


# ===========================================================================
def main():
    rng = np.random.default_rng(SEMILLA)
    univ = pd.concat([leer_gs(DATA / r, h, f, c) for r, h, f, c in FUENTES], ignore_index=True)

    univ["sem_sueros"] = univ["T. Sueros"].fillna(univ["Sueros"])
    univ["Cluster"] = univ["Cluster"].astype(str).str.strip().str.upper()
    univ["demanda_potencial"] = univ["Cluster"].str[0].map({"H": "Alta", "L": "Baja"})
    univ["venta_valor"] = univ["Cluster"].str[1].map({"H": "Alta", "L": "Baja"})
    univ["Colonia"] = univ["Colonia"].fillna("SIN COLONIA").replace(r"^\s*$", "SIN COLONIA", regex=True)
    univ["es_golden"] = univ.Cluster.eq("HH").astype(int)

    IDX = [c for c in univ.columns if c.endswith("|Indice")]
    HHS = [c for c in univ.columns if c.endswith("|HHs")]
    N = len(univ)
    R = {"meta": {"universo_pdv": N, "indices": len(IDX), "semilla": SEMILLA,
                  "pesos": W,
                  "nota": "Sin muestreo: todos los cálculos usan los 8,864 PDV."}}

    # ---------------- 1 · integridad ----------------
    R["integridad"] = {
        "filas": N,
        "nielsen_id_unicos": int(univ["Nielsen ID"].nunique()),
        "duplicados": int(univ.duplicated("Nielsen ID").sum()),
        "geo_nulos": int(univ[["Latitud", "Longitud"]].isna().sum().sum()),
        "geo_fuera_bbox": int((~(univ.Latitud.between(14, 33) &
                                 univ.Longitud.between(-118, -86))).sum()),
        "indices_nulos": int(univ[IDX].isna().sum().sum()),
        "por_fuente": [{"fuente": f, "canal": c, "pdv": int(n)} for (f, c), n in
                       univ.groupby(["fuente", "canal"]).size().items()],
        "cadenas": int(univ.Cadena.nunique()),
        "zonas_metropolitanas": int(univ["Zonas Metropolitanas"].nunique()),
    }

    # ---------------- 2 · univariado, LOS 20 ----------------
    uni = []
    for c in IDX:
        x = pd.to_numeric(univ[c], errors="coerce").dropna().to_numpy(float)
        xs = np.sort(x[x > 0]); k = max(10, int(len(xs) * .10)); cola = xs[-k:]
        uni.append({"variable": c.replace("|Indice", ""), "n": int(len(x)),
                    "media": round(float(x.mean()), 1), "mediana": round(float(np.median(x)), 1),
                    "desv": round(float(x.std(ddof=1)), 1),
                    "p5": round(float(np.percentile(x, 5)), 1),
                    "p95": round(float(np.percentile(x, 95)), 1),
                    "sesgo": round(float(stats.skew(x)), 2),
                    "curtosis": round(float(stats.kurtosis(x)), 2),
                    "hill_alpha": round(float(k / np.sum(np.log(cola / cola[0]))), 2)})
    R["univariado"] = uni
    R["univariado_resumen"] = {
        "cola_pesada": int(sum(abs(u["sesgo"]) > 1 or u["curtosis"] > 3 for u in uni)),
        "alpha_min": round(min(u["hill_alpha"] for u in uni), 2)}

    # ---------------- 3 · dependencias, LOS 190 PARES ----------------
    X = univ[IDX].apply(pd.to_numeric, errors="coerce")
    pares = []
    for i, a in enumerate(IDX):
        for b in IDX[i + 1:]:
            rp, _ = stats.pearsonr(X[a], X[b])
            rs, ps = stats.spearmanr(X[a], X[b])
            pares.append({"a": a.replace("|Indice", ""), "b": b.replace("|Indice", ""),
                          "pearson": round(float(rp), 3), "spearman": round(float(rs), 3), "p": ps})
    q = stats.false_discovery_control(np.clip([p["p"] for p in pares], 1e-300, 1), method="bh")
    for p_, qq in zip(pares, q):
        p_["q_BH"] = float(qq); p_.pop("p")
    pares.sort(key=lambda d: -abs(d["spearman"]))
    R["dependencias"] = {"n_pares": len(pares),
                         "significativos": int(sum(p["q_BH"] < .05 for p in pares)),
                         "pares": pares}

    Xs = StandardScaler().fit_transform(X.fillna(100))
    pca = PCA(random_state=SEMILLA).fit(Xs)
    ev = pca.explained_variance_ratio_
    carga = pd.DataFrame(pca.components_.T, index=[c.replace("|Indice", "") for c in IDX])
    R["pca"] = {"k90": int(np.argmax(np.cumsum(ev) >= .90) + 1),
                "componentes": [{"pc": f"PC{i+1}", "var": round(float(v) * 100, 1),
                                 "acum": round(float(np.cumsum(ev)[i]) * 100, 1),
                                 "top": list(carga[i].abs().nlargest(4).index)}
                                for i, v in enumerate(ev)]}

    mcd = MinCovDet(random_state=SEMILLA, support_fraction=.85).fit(X.fillna(100))
    d2 = mcd.mahalanobis(X.fillna(100))
    corte = float(stats.chi2.ppf(.999, df=len(IDX)))
    univ["entorno_atipico"] = d2 > corte
    R["atipicos"] = {"corte_chi2": round(corte, 1), "n": int(univ.entorno_atipico.sum()),
                     "pct": round(float(univ.entorno_atipico.mean() * 100), 1),
                     "nota": "MinCovDet usa 85% de soporte por diseño del estimador robusto; "
                             "la distancia se evalúa sobre los 8,864 PDV."}

    # ---------------- 4 · segmentacion ----------------
    seg = univ.Cluster.value_counts()
    R["clusters"] = [{"cluster": k, "pdv": int(v), "pct": round(v / N * 100, 1)}
                     for k, v in seg.items()]
    val = []
    alta = univ.demanda_potencial.eq("Alta")
    for c in IDX:
        x = pd.to_numeric(univ[c], errors="coerce")
        a, b = x[alta].dropna(), x[~alta].dropna()
        val.append({"variable": c.replace("|Indice", ""), "d": round(cohen_d(a, b), 3),
                    "p": stats.mannwhitneyu(a, b).pvalue})
    qv = stats.false_discovery_control(np.clip([v["p"] for v in val], 1e-300, 1), method="bh")
    for v_, qq in zip(val, qv):
        v_["q_BH"] = float(qq); v_.pop("p")
    R["validacion_demanda"] = {"significativos": int(sum(v["q_BH"] < .05 for v in val)),
                               "total": len(val), "detalle": val}

    # ---------------- 5 · cluster x semaforo ----------------
    ct = pd.crosstab(univ.Cluster, univ.sem_sueros)
    boot = []
    cl, sm = univ.Cluster.to_numpy(), univ.sem_sueros.to_numpy()
    for _ in range(2000):
        i = rng.integers(0, N, N)
        c = pd.crosstab(pd.Series(cl[i]), pd.Series(sm[i]))
        if min(c.shape) > 1:
            boot.append(cramers_v(c, N))
    pct = pd.crosstab(univ.Cluster, univ.sem_sueros, normalize="index").mul(100).round(1)
    R["cluster_semaforo"] = {
        "cramers_v": round(cramers_v(ct, N), 4),
        "ic95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "n_bootstrap": len(boot),
        "tabla": [{"cluster": k, **{c: float(pct.loc[k, c]) for c in ["VERDE", "AMARILLO", "ROJO"]},
                   "pdv": int(seg[k])} for k in ["HH", "HL", "LH", "LL"]],
        "hh_rojo": int((univ.Cluster.eq("HH") & univ.sem_sueros.eq("ROJO")).sum()),
    }

    # ---------------- 6 · perfil golden: LOS 20 INDICES ----------------
    g, ng = univ.es_golden.eq(1), univ.es_golden.eq(0)
    perfil = []
    for c in IDX:
        x = pd.to_numeric(univ[c], errors="coerce")
        a, b = x[g].dropna(), x[ng].dropna()
        perfil.append({"variable": c.replace("|Indice", ""),
                       "golden": round(float(a.mean()), 1), "resto": round(float(b.mean()), 1),
                       "d": round(cohen_d(a, b), 3), "p": stats.mannwhitneyu(a, b).pvalue})
    qp = stats.false_discovery_control(np.clip([p["p"] for p in perfil], 1e-300, 1), method="bh")
    for p_, qq in zip(perfil, qp):
        p_["q_BH"] = float(qq); p_.pop("p")
    perfil.sort(key=lambda d: -d["d"])
    R["perfil_golden"] = {"base_pct": round(float(univ.es_golden.mean() * 100), 1),
                          "n_golden": int(univ.es_golden.sum()), "detalle": perfil}

    base = univ.es_golden.mean()

    def lift(col):
        t = univ.groupby(col).es_golden.agg(pdv="count", golden="sum")
        t["pct"] = (t.golden / t.pdv * 100).round(1)
        t["lift"] = (t.golden / t.pdv / base).round(2)
        return [{"nombre": str(k), "pdv": int(r.pdv), "golden": int(r.golden),
                 "pct": float(r.pct), "lift": float(r.lift)}
                for k, r in t.sort_values("lift", ascending=False).iterrows()]

    R["lift_cadena"] = lift("Cadena")          # LAS 39
    R["lift_zm"] = lift("Zonas Metropolitanas")  # LAS 60

    # ---------------- 7 · modelo de afinidad ----------------
    NUM = IDX + [c for c in HHS if c.split("|")[0] in ("A/B", "C+", "C", "Sin Ninos")]
    CAT = [c for c in ["Cadena", "canal", "Region Nielsen", "Zonas Metropolitanas", "Estado"]
           if c in univ.columns]
    FUGA = {"Cluster", "sem_sueros", "venta_valor", "demanda_potencial", "es_golden"}
    assert not (set(NUM) | set(CAT)) & FUGA

    Xm = univ[NUM + CAT].copy()
    for c in NUM:
        Xm[c] = pd.to_numeric(Xm[c], errors="coerce")
    Xm[NUM] = Xm[NUM].fillna(Xm[NUM].median())
    y = univ.es_golden.to_numpy()

    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=False), CAT)])
    modelos = {
        "Regresión logística (L2)": Pipeline([("prep", prep), ("clf", LogisticRegression(
            max_iter=2000, C=.5, class_weight="balanced", random_state=SEMILLA))]),
        "Gradient boosting": Pipeline([("prep", prep), ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=.06, max_leaf_nodes=24, random_state=SEMILLA))]),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)
    scores, comparativa = {}, []
    for nombre, mod in modelos.items():
        p = cross_val_predict(mod, Xm, y, cv=cv, method="predict_proba")[:, 1]
        scores[nombre] = p
        comparativa.append({"modelo": nombre, "auc_roc": round(roc_auc_score(y, p), 3),
                            "auc_pr": round(average_precision_score(y, p), 3)})
    mejor = max(scores, key=lambda k: roc_auc_score(y, scores[k]))
    univ["p_golden"] = scores[mejor]
    univ["IAS"] = (univ.p_golden.rank(pct=True) * 100).round(1)

    d = pd.DataFrame({"p": univ.p_golden, "y": y})
    d["decil"] = pd.qcut(d.p, 10, labels=False, duplicates="drop") + 1
    cal = d.groupby("decil").agg(obs=("y", "mean"), pred=("p", "mean"), n=("y", "size"))

    # CORRECCION 1: importancia por pliegue de CV. Cada PDV entra exactamente una vez
    # como dato retenido, asi que la medicion cubre el universo completo y no un 25%.
    imps, cubiertos = [], 0
    for tr, te in cv.split(Xm, y):
        m = modelos[mejor].fit(Xm.iloc[tr], y[tr])
        r = permutation_importance(m, Xm.iloc[te], y[te], n_repeats=5,
                                   random_state=SEMILLA, scoring="roc_auc")
        imps.append(r.importances_mean); cubiertos += len(te)
    imp = pd.Series(np.mean(imps, axis=0), index=NUM + CAT).sort_values(ascending=False)
    R["afinidad"] = {
        "comparativa": comparativa, "elegido": mejor,
        "base_pct": round(float(y.mean() * 100), 1),
        "decil_superior_pct": round(float(cal.obs.iloc[-1] * 100), 1),
        "lift_decil": round(float(cal.obs.iloc[-1] / y.mean()), 1),
        "calibracion": [{"decil": int(i), "observado": round(float(r.obs * 100), 1),
                         "predicho": round(float(r.pred * 100), 1), "n": int(r.n)}
                        for i, r in cal.iterrows()],
        "importancia": [{"variable": k.replace("|Indice", "").replace("|HHs", " (hogares)"),
                         "caida_auc": round(float(v), 5)} for k, v in imp.items()],
        "cobertura_importancia": {"pdv_evaluados": int(cubiertos), "pct": round(cubiertos / N * 100, 1),
                                  "metodo": "permutación por pliegue de CV, promediada sobre 5 pliegues"},
    }

    # ---------------- 8 · scantrack ----------------
    ciu = pd.read_excel(DATA / "Info Top Ciudades Industria RRS MX.xlsx",
                        sheet_name="Segmentación Ciudades", header=1)
    ciu.columns = ["embotellador", "ciudad", "mix_as", "tiendas_oxxo", "nota"][:len(ciu.columns)]
    ciu = ciu.dropna(subset=["ciudad"])
    ciu["ciudad"] = ciu.ciudad.astype(str).str.strip()
    ciu = ciu[~ciu.ciudad.str.lower().eq("otras")]
    ciu["embotellador"] = ciu.embotellador.ffill()

    PUENTE = {"VDM": "VALLE DE MEXICO", "Guadalajara": "GUADALAJARA", "Monterrey": "MONTERREY",
              "Mérida": "MERIDA", "Cancún": "CANCUN", "Tijuana": "TIJUANA",
              "Puebla": "PUEBLA TLAXCALA", "Querétaro": "QUERETARO", "Culiacán": "CULIACAN",
              "Chihuahua": "CHIHUAHUA", "Veracruz": "VERACRUZ", "León ": "LEON",
              "SLP ": "SAN LUIS POTOSI", "Tampico": "TAMPICO", "Playa del Carmen": "PLAYA DEL CARMEN",
              "Torreón ": "LAGUNA", "Saltillo": "SALTILLO", "Mexicali": "MEXICALI",
              "Acapulco": "ACAPULCO", "Cuernavaca": "CUERNAVACA", "aguascalientes": "AGUASCALIENTES",
              "Hermosillo": "HERMOSILLO", "Puerto Vallarta": "PUERTO VALLARTA",
              "Villahermosa": "VILLAHERMOSA", "Mazatlán ": "MAZATLAN", "Durango": "DURANGO",
              "Tuxtla": "TUXTLA GUTIERREZ", "La Paz": "LA PAZ", "Xalapa": "XALAPA",
              "Oaxaca": "OAXACA", "Ensenada": "ENSENADA", "Reynosa": "REYNOSA",
              "Matamoros": "MATAMOROS", "Nogales": "NOGALES", "Tepic": "TEPIC",
              "Coatzacoalcos": "COATZACOALCOS", "Irapuato": "IRAPUATO", "Nuevo Laredo": "NUEVO LAREDO",
              "Orizaba": "ORIZABA", "Poza Rica": "POZA RICA", "Cozumel": "COZUMEL",
              "Campeche": "CAMPECHE", "Monclova": "MONCLOVA", "Los Cabos": "LOS CABOS",
              "Morelina": "MORELIA"}
    ciu["zm"] = ciu.ciudad.map(PUENTE)
    zm_univ = set(univ["Zonas Metropolitanas"].dropna().unique())
    enl = ciu[ciu.zm.isin(zm_univ)]
    peso = enl.groupby("zm").mix_as.max()
    peso = (peso / peso.max()).clip(lower=.05)
    univ["peso_ciudad"] = univ["Zonas Metropolitanas"].map(peso).fillna(peso.min())

    R["scantrack"] = {
        "ciudades_archivo": int(len(ciu)), "enlazadas": int(len(enl)),
        "pct_mix_mapeado": round(float(enl.mix_as.sum() / ciu.mix_as.sum() * 100)),
        "sin_enlace": sorted(set(ciu.ciudad) - set(enl.ciudad)),
        "pdv_cubiertos": int(univ["Zonas Metropolitanas"].isin(peso.index).sum()),
        "detalle": [{"ciudad": r.ciudad, "embotellador": str(r.embotellador), "zm": r.zm,
                     "mix_as": float(r.mix_as), "peso": round(float(peso[r.zm]), 3)}
                    for _, r in enl.sort_values("mix_as", ascending=False).iterrows()],
    }

    # ---------------- 9 · potencial NACIONAL ----------------
    univ["desempeno_actual"] = univ.sem_sueros.map(ESCALA_SEMAFORO)
    univ["brecha"] = 1 - univ.desempeno_actual
    univ["ias_n"] = univ.IAS / 100

    def geom(df, pesos):
        v = np.ones(len(df))
        for c, w in zip(["ias_n", "brecha", "peso_ciudad"], pesos):
            v *= np.clip(df[c].to_numpy(float), 1e-3, None) ** w
        return v

    base_w = [W["afinidad"], W["brecha"], W["ciudad"]]
    univ["potencial"] = geom(univ, base_w)
    univ["potencial"] = (univ.potencial / univ.potencial.max() * 100).round(1)
    univ["prioridad"] = pd.cut(univ.potencial.rank(pct=True), [0, .70, .90, 1.0],
                               labels=["P3", "P2", "P1"], include_lowest=True)

    R["potencial_nacional"] = {
        "tramos": [{"tramo": str(k), "pdv": int(v.pdv), "ias_medio": round(float(v.ias), 1),
                    "brecha_media": round(float(v.br), 2), "pct_golden": round(float(v.gold * 100), 1)}
                   for k, v in univ.groupby("prioridad", observed=True).agg(
                       pdv=("Nielsen ID", "count"), ias=("IAS", "mean"),
                       br=("brecha", "mean"), gold=("es_golden", "mean")).iterrows()],
        "p1_nacional": int((univ.prioridad == "P1").sum()),
    }

    # CORRECCION 2: sensibilidad sobre el ranking completo, no sobre el top-500.
    ESC = {"Base (.50/.30/.20)": (.50, .30, .20), "Afinidad (.80/.10/.10)": (.80, .10, .10),
           "Brecha (.30/.50/.20)": (.30, .50, .20), "Geografía (.40/.20/.40)": (.40, .20, .40),
           "Equitativo (1/3)": (1 / 3, 1 / 3, 1 / 3)}
    n_p1 = int((univ.prioridad == "P1").sum())
    base_p1 = set(univ.nlargest(n_p1, "potencial")["Nielsen ID"])
    sens = []
    for nombre, w in ESC.items():
        s = geom(univ, w)
        sens.append({"escenario": nombre,
                     "spearman_ranking_completo": round(float(stats.spearmanr(s, univ.potencial).statistic), 4),
                     "kendall_ranking_completo": round(float(stats.kendalltau(s, univ.potencial).statistic), 4),
                     "traslape_p1_pct": round(len(set(univ.assign(_s=s).nlargest(n_p1, "_s")["Nielsen ID"])
                                                  & base_p1) / n_p1 * 100, 1)})
    R["sensibilidad"] = {"n_evaluados": N, "tramo_p1": n_p1, "escenarios": sens,
                         "nota": "Correlación sobre los 8,864 PDV completos; el traslape usa el "
                                 "tramo de decisión real (decil superior nacional), no un top-N arbitrario."}

    # ---------------- 10 · corte por plaza ----------------
    R["plazas"] = []
    for zm, nombre in PLAZAS.items():
        d = univ[univ["Zonas Metropolitanas"] == zm]
        R["plazas"].append({
            "zm": zm, "nombre": nombre, "pdv": int(len(d)),
            "pct_universo": round(len(d) / N * 100, 1),
            "farmacias_pct": round(float(d.canal.eq("Farmacias").mean() * 100), 1),
            "golden": int(d.es_golden.sum()), "golden_pct": round(float(d.es_golden.mean() * 100), 1),
            "hh_rojo": int((d.Cluster.eq("HH") & d.sem_sueros.eq("ROJO")).sum()),
            "peso_ciudad": round(float(d.peso_ciudad.iloc[0]), 3),
            "p1_nacional": int((d.prioridad == "P1").sum()),
            "ias_medio": round(float(d.IAS.mean()), 1),
        })
    R["plazas_total"] = {"pdv": int(sum(p["pdv"] for p in R["plazas"])),
                         "pct_universo": round(sum(p["pdv"] for p in R["plazas"]) / N * 100, 1)}

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
    # Solo las columnas del mapa: el frame completo trae columnas de tipo mixto
    # (Retailer ID mezcla texto y numero) que ningun formato columnar acepta.
    MAPA = ["Nielsen ID", "Latitud", "Longitud", "sem_sueros", "Cluster", "canal",
            "Cadena", "Zonas Metropolitanas", "Estado", "IAS", "potencial", "prioridad"]
    univ[MAPA].to_csv(RAIZ / "outputs/report/universo_mapa.csv", index=False,
                      encoding="utf-8")

    print(f"universo         : {N:,} PDV, sin muestreo")
    print(f"pares evaluados  : {R['dependencias']['n_pares']} ({R['dependencias']['significativos']} signif.)")
    print(f"cadenas / ZM     : {len(R['lift_cadena'])} / {len(R['lift_zm'])} (completas)")
    print(f"importancia sobre: {R['afinidad']['cobertura_importancia']['pdv_evaluados']:,} PDV "
          f"({R['afinidad']['cobertura_importancia']['pct']}%)")
    print(f"sensibilidad     : Spearman sobre {N:,}, traslape en tramo P1 = {n_p1:,}")
    print(f"-> {SALIDA}")


if __name__ == "__main__":
    main()
