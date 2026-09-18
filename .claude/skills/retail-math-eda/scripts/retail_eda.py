"""
============================================================================
retail_eda — toolkit de EDA retail con rigor matematico
============================================================================
Polars + NumPy + SciPy. Sin pandas. scikit-learn y statsmodels son opcionales:
si faltan, las funciones que los usan caen a una alternativa documentada.

Cubre los pasos computacionales 1-4 del SKILL.md:

  1. integridad, densidad_temporal            -> integridad y faltantes
  2. univariado, hill, mejor_ajuste           -> distribuciones y colas pesadas
  3. dependencias, mahalanobis                -> dependencia bi/multivariada
  4. market_basket, rfm, elasticidad, abc_xyz -> dimensiones de negocio
     kpis                                     -> sintesis ejecutiva

Uso:
    import sys; sys.path.insert(0, "<skill_dir>/scripts")
    import retail_eda as rx
    rx.integridad(df, clave=["txn_id", "linea"], cols_monetarias=["importe"])

Verificar el entorno:
    python retail_eda.py --self-test
"""
from __future__ import annotations

import datetime as _dt
from typing import Iterable, Sequence

import numpy as np
import polars as pl
from scipy import stats

try:  # opcional: estimador KSG de informacion mutua y covarianza robusta
    from sklearn.feature_selection import mutual_info_regression as _mi_ksg
    from sklearn.covariance import MinCovDet as _MinCovDet
    _TIENE_SKLEARN = True
except ImportError:  # pragma: no cover - depende del entorno
    _TIENE_SKLEARN = False


# ════════════════════════════════════════════════════════════════════════════
#  Utilidades internas
# ════════════════════════════════════════════════════════════════════════════

def _a_numpy(x) -> np.ndarray:
    """Serie de polars, lista o array -> ndarray float64 sin nulos ni NaN."""
    if isinstance(x, pl.Series):
        x = x.to_numpy()
    x = np.asarray(x, dtype="float64").ravel()
    return x[np.isfinite(x)]


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg. Usa scipy si esta disponible, si no lo implementa."""
    p = np.asarray(p, dtype="float64")
    try:
        return stats.false_discovery_control(p, method="bh")
    except AttributeError:  # scipy < 1.11
        m = p.size
        orden = np.argsort(p)
        q = np.empty(m)
        prev = 1.0
        for pos in range(m - 1, -1, -1):
            i = orden[pos]
            prev = min(prev, p[i] * m / (pos + 1))
            q[i] = prev
        return np.clip(q, 0.0, 1.0)


def _ols_hc1(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS con errores estandar Huber-White HC1 y prueba Breusch-Pagan."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)

    # HC1: (n/(n-k)) * (X'X)^-1 (Σ e_i^2 x_i x_i') (X'X)^-1
    meat = (X * (resid ** 2)[:, None]).T @ X
    cov_hc1 = (n / max(n - k, 1)) * xtx_inv @ meat @ xtx_inv
    se_hc1 = np.sqrt(np.clip(np.diag(cov_hc1), 0, None))

    # errores clasicos, para contrastar
    s2 = float(resid @ resid) / max(n - k, 1)
    se_ols = np.sqrt(np.clip(np.diag(s2 * xtx_inv), 0, None))

    # Breusch-Pagan: LM = n * R^2 de e^2 sobre X
    e2 = resid ** 2
    b_aux, *_ = np.linalg.lstsq(X, e2, rcond=None)
    ss_res = float(np.sum((e2 - X @ b_aux) ** 2))
    ss_tot = float(np.sum((e2 - e2.mean()) ** 2))
    r2_aux = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lm = n * r2_aux
    gl = max(k - 1, 1)

    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se_hc1 > 0, beta / se_hc1, np.nan)
    return {
        "beta": beta,
        "se_hc1": se_hc1,
        "se_ols": se_ols,
        "t_hc1": t,
        "p_hc1": 2 * stats.norm.sf(np.abs(t)),
        "r2": r2,
        "n": n,
        "bp_lm": float(lm),
        "bp_p": float(stats.chi2.sf(lm, gl)),
        "heterocedastico": bool(stats.chi2.sf(lm, gl) < 0.05),
    }


# ════════════════════════════════════════════════════════════════════════════
#  1. Integridad y faltantes
# ════════════════════════════════════════════════════════════════════════════

def integridad(
    df: pl.DataFrame,
    *,
    clave: Sequence[str] | None = None,
    cols_monetarias: Sequence[str] = (),
    cols_cantidad: Sequence[str] = (),
) -> dict:
    """Censo de integridad: nulos, duplicados de clave, signos anomalos.

    `clave` es el grano declarado (p.ej. ["txn_id", "linea", "sku"]). Si la clave
    no es unica, ningun agregado posterior es confiable.
    """
    n = df.height
    nulos = (
        df.null_count()
        .transpose(include_header=True, header_name="columna", column_names=["nulos"])
        .with_columns((pl.col("nulos") / max(n, 1)).alias("pct_nulos"))
        .filter(pl.col("nulos") > 0)
        .sort("nulos", descending=True)
    )

    out: dict = {"filas": n, "columnas": df.width, "nulos": nulos}

    if clave:
        faltan = [c for c in clave if c not in df.columns]
        if faltan:
            raise KeyError(f"columnas de clave ausentes: {faltan}")
        n_unicas = df.select(clave).n_unique()
        out["clave"] = list(clave)
        out["clave_unica"] = bool(n_unicas == n)
        out["filas_duplicadas_clave"] = int(n - n_unicas)

    out["filas_duplicadas_totales"] = int(df.is_duplicated().sum())

    signos = []
    for col, tipo in ((c, "monetaria") for c in cols_monetarias):
        signos.append((col, tipo))
    for col in cols_cantidad:
        signos.append((col, "cantidad"))

    detalle = []
    for col, tipo in signos:
        if col not in df.columns:
            continue
        s = df[col]
        detalle.append(
            {
                "columna": col,
                "tipo": tipo,
                "negativos": int((s < 0).sum() or 0),
                "ceros": int((s == 0).sum() or 0),
                "nulos": int(s.null_count()),
                "min": float(s.min()) if s.len() and s.min() is not None else None,
                "max": float(s.max()) if s.len() and s.max() is not None else None,
            }
        )
    out["signos"] = pl.DataFrame(detalle) if detalle else pl.DataFrame()

    # devoluciones = cantidad negativa; ajustes = signos discordantes
    if cols_cantidad and cols_monetarias:
        q, m = cols_cantidad[0], cols_monetarias[0]
        if q in df.columns and m in df.columns:
            out["devoluciones"] = int(df.filter(pl.col(q) < 0).height)
            out["ajustes_signo_discordante"] = int(
                df.filter(
                    ((pl.col(q) > 0) & (pl.col(m) < 0))
                    | ((pl.col(q) < 0) & (pl.col(m) > 0))
                ).height
            )
    return out


def densidad_temporal(
    df: pl.DataFrame,
    fecha: str,
    *,
    grupo: str | None = None,
    every: str = "1d",
) -> pl.DataFrame:
    """Conteo de filas por periodo (y por grupo) para detectar caidas de captura.

    Marca `hueco=True` en los periodos con cero filas frente a la rejilla completa
    del rango observado. Un cero aqui casi nunca es demanda cero.
    """
    d = df.select([fecha] + ([grupo] if grupo else [])).drop_nulls(fecha)
    d = d.with_columns(pl.col(fecha).cast(pl.Date)).sort(fecha)
    if d.is_empty():
        return pl.DataFrame()

    por = [grupo] if grupo else None
    conteo = (
        d.group_by_dynamic(fecha, every=every, group_by=por)
        .agg(pl.len().alias("filas"))
    )

    rejilla = pl.date_range(
        d[fecha].min(), d[fecha].max(), interval=every, eager=True
    ).alias(fecha).to_frame()

    if grupo:
        grupos = d.select(grupo).unique()
        rejilla = rejilla.join(grupos, how="cross")
        llaves = [fecha, grupo]
    else:
        llaves = [fecha]

    return (
        rejilla.join(conteo, on=llaves, how="left")
        .with_columns(pl.col("filas").fill_null(0))
        .with_columns((pl.col("filas") == 0).alias("hueco"))
        .sort(llaves)
    )


# ════════════════════════════════════════════════════════════════════════════
#  2. Univariado y colas pesadas
# ════════════════════════════════════════════════════════════════════════════

def hill(x, k: int | None = None) -> dict:
    """Estimador de Hill del indice de cola sobre los k mayores estadisticos.

    alpha < 1 -> media infinita; alpha < 2 -> varianza infinita (no reportar sd).
    """
    x = _a_numpy(x)
    x = x[x > 0]
    n = x.size
    if n < 50:
        return {"alpha": float("nan"), "k": 0, "n_positivos": int(n)}
    if k is None:
        k = max(int(round(n ** 0.5)), 10)     # regla practica k ~ sqrt(n)
    k = min(k, n - 1)
    orden = np.sort(x)[::-1]
    corte = orden[k]
    suma = float(np.sum(np.log(orden[:k]) - np.log(corte)))
    alpha = k / suma if suma > 0 else float("inf")
    return {
        "alpha": float(alpha),
        "k": int(k),
        "n_positivos": int(n),
        "media_finita": bool(alpha > 1),
        "varianza_finita": bool(alpha > 2),
    }


def mejor_ajuste(x, familias: Sequence[str] = ("lognorm", "pareto", "weibull_min"),
                 n_max: int = 5000, seed: int = 2000) -> pl.DataFrame:
    """Rankea familias por log-verosimilitud y estadistico KS.

    Submuestrea a `n_max`: con n grande toda prueba de bondad de ajuste rechaza,
    asi que el ranking se lee por estadistico/AIC, no por el p-valor.
    """
    x = _a_numpy(x)
    x = x[x > 0]
    if x.size < 50:
        return pl.DataFrame()
    rng = np.random.default_rng(seed)
    m = x if x.size <= n_max else rng.choice(x, n_max, replace=False)

    filas = []
    for fam in familias:
        dist = getattr(stats, fam)
        try:
            par = dist.fit(m, floc=0)
            ks = stats.kstest(m, fam, args=par)
            ll = float(np.sum(dist.logpdf(m, *par)))
            k = len(par)
            filas.append(
                {
                    "familia": fam,
                    "ks_stat": float(ks.statistic),
                    "ks_p": float(ks.pvalue),
                    "loglik": ll,
                    "aic": 2 * k - 2 * ll,
                    "params": str(tuple(round(float(p), 4) for p in par)),
                }
            )
        except Exception as exc:  # ajuste fallido: se reporta, no se oculta
            filas.append({"familia": fam, "ks_stat": None, "ks_p": None,
                          "loglik": None, "aic": None, "params": f"error: {exc}"})
    return pl.DataFrame(filas).sort("aic", nulls_last=True)


def univariado(x, nombre: str = "x", *, ajustes: bool = True) -> dict:
    """Momentos, estadisticos robustos, inflacion de ceros e indice de cola."""
    if isinstance(x, pl.Series):
        nombre = x.name or nombre
    bruto = _a_numpy(x)
    n = bruto.size
    if n == 0:
        return {"columna": nombre, "n": 0}

    neg = int(np.sum(bruto < 0))
    ceros = int(np.sum(bruto == 0))
    pos = bruto[bruto > 0]

    med = float(np.median(bruto))
    mad = float(np.median(np.abs(bruto - med)))
    q1, q3 = np.percentile(bruto, [25, 75])

    out = {
        "columna": nombre,
        "n": n,
        "negativos": neg,
        "pct_negativos": neg / n,
        "ceros": ceros,
        "pct_ceros": ceros / n,
        "media": float(bruto.mean()),
        "sd": float(bruto.std(ddof=1)) if n > 1 else float("nan"),
        "cv": float(bruto.std(ddof=1) / bruto.mean()) if n > 1 and bruto.mean() != 0 else float("nan"),
        "mediana": med,
        "iqr": float(q3 - q1),
        "mad": mad,
        "mad_n": 1.4826 * mad,
        "p01": float(np.percentile(bruto, 1)),
        "p99": float(np.percentile(bruto, 99)),
        "min": float(bruto.min()),
        "max": float(bruto.max()),
        "asimetria": float(stats.skew(bruto)),
        "curtosis_exceso": float(stats.kurtosis(bruto)),
    }

    # parte condicional: distribucion dado que hay compra
    if pos.size > 1:
        out["media_condicional"] = float(pos.mean())
        out["mediana_condicional"] = float(np.median(pos))
        out["p_positivo"] = pos.size / n
        out.update({f"hill_{k}": v for k, v in hill(pos).items()})
        # concentracion: cuota del 1% y 20% superior (Pareto empirico)
        orden = np.sort(pos)[::-1]
        total = orden.sum()
        if total > 0:
            out["cuota_top1pct"] = float(orden[: max(1, pos.size // 100)].sum() / total)
            out["cuota_top20pct"] = float(orden[: max(1, pos.size // 5)].sum() / total)

    if neg > 0:
        out["_aviso"] = ("hay valores negativos: separar devoluciones antes de "
                         "transformar o calcular varianza (ver gotchas.md)")

    if ajustes and pos.size >= 50:
        out["ajustes"] = mejor_ajuste(pos)
        try:
            _, lam = stats.boxcox(pos)
            out["boxcox_lambda"] = float(lam)
        except Exception:
            out["boxcox_lambda"] = None
    return out


# ════════════════════════════════════════════════════════════════════════════
#  3. Dependencia bivariada y multivariada
# ════════════════════════════════════════════════════════════════════════════

def _mi_binned(a: np.ndarray, b: np.ndarray, bins: int = 12) -> float:
    """MI por binning en cuantiles, con correccion de sesgo Miller-Madow."""
    def _bordes(v):
        e = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)))
        return e if e.size > 2 else np.array([v.min(), v.max()])
    ca = np.clip(np.digitize(a, _bordes(a)[1:-1]), 0, None)
    cb = np.clip(np.digitize(b, _bordes(b)[1:-1]), 0, None)
    tabla = np.histogram2d(ca, cb, bins=(ca.max() + 1, cb.max() + 1))[0]
    n = tabla.sum()
    if n == 0:
        return 0.0
    p = tabla / n
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p * np.log(p / (px @ py))
    mi = float(np.nansum(term))
    bx, by = (px > 0).sum(), (py > 0).sum()
    mi -= (bx * by - bx - by + 1) / (2 * n)     # Miller-Madow
    return max(mi, 0.0)


def dependencias(
    df: pl.DataFrame,
    cols: Sequence[str],
    *,
    n_max: int = 50_000,
    con_mi: bool = True,
    q: float = 0.05,
    seed: int = 2000,
) -> pl.DataFrame:
    """Panel de dependencia por pares: Pearson, Spearman, Kendall, MI y q de BH.

    Kendall solo se calcula con n <= 5000 (es O(n log n) pero con constante alta).
    MI usa el estimador KSG de scikit-learn si esta instalado; si no, binning.
    """
    cols = [c for c in cols if c in df.columns]
    d = df.select(cols).drop_nulls()
    if d.height > n_max:
        d = d.sample(n_max, seed=seed)
    M = d.to_numpy().astype("float64")
    if M.shape[0] < 30 or len(cols) < 2:
        return pl.DataFrame()

    filas = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = M[:, i], M[:, j]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            r, pr = stats.pearsonr(a, b)
            rho, ps = stats.spearmanr(a, b)
            if a.size <= 5000:
                tau, pt = stats.kendalltau(a, b)
            else:
                tau, pt = float("nan"), float("nan")
            fila = {
                "x": cols[i], "y": cols[j],
                "pearson": float(r), "p_pearson": float(pr),
                "spearman": float(rho), "p_spearman": float(ps),
                "kendall": float(tau), "p_kendall": float(pt),
            }
            if con_mi:
                if _TIENE_SKLEARN:
                    mi = float(_mi_ksg(a.reshape(-1, 1), b, n_neighbors=3,
                                       random_state=seed)[0])
                else:
                    mi = _mi_binned(a, b)
                fila["mi"] = mi
                # no-monotonia: MI alta con rho baja es el caso interesante
                fila["no_monotonica"] = bool(mi > 0.05 and abs(rho) < 0.15)
            filas.append(fila)

    out = pl.DataFrame(filas)
    if out.is_empty():
        return out
    out = out.with_columns(pl.Series("q_spearman", _bh(out["p_spearman"].to_numpy())))
    return out.with_columns(
        (pl.col("q_spearman") < q).alias("significativa")
    ).sort(pl.col("spearman").abs(), descending=True)


def mahalanobis(
    df: pl.DataFrame,
    cols: Sequence[str],
    *,
    robusto: bool = True,
    alpha: float = 0.975,
    log1p: bool = False,
) -> pl.DataFrame:
    """Distancia de Mahalanobis y corte chi2. `log1p=True` para columnas monetarias.

    Con `robusto=True` usa MinCovDet (sklearn) para evitar el enmascaramiento: la
    media y covarianza clasicas ya estan contaminadas por los atipicos buscados.
    Usa pseudo-inversa, asi que tolera columnas colineales (revenue = units*price).
    """
    cols = [c for c in cols if c in df.columns]
    d = df.select(cols).drop_nulls()
    M = d.to_numpy().astype("float64")
    if M.shape[0] <= M.shape[1] + 1:
        return pl.DataFrame()
    if log1p:
        M = np.log1p(np.clip(M, 0, None))

    if robusto and _TIENE_SKLEARN:
        est = _MinCovDet(random_state=0).fit(M)
        mu, S = est.location_, est.covariance_
    else:
        mu, S = M.mean(0), np.cov(M, rowvar=False)

    dif = M - mu
    d2 = np.einsum("ij,jk,ik->i", dif, np.linalg.pinv(S), dif)
    corte = float(stats.chi2.ppf(alpha, len(cols)))
    return d.with_columns(
        pl.Series("d2", d2),
        pl.Series("atipico", d2 > corte),
        pl.lit(corte).alias("corte_chi2"),
    ).sort("d2", descending=True)


# ════════════════════════════════════════════════════════════════════════════
#  4A. Market basket
# ════════════════════════════════════════════════════════════════════════════

def market_basket(
    df: pl.DataFrame,
    txn: str,
    sku: str,
    *,
    top_n: int = 60,
    min_conteo: int = 30,
    min_lift: float = 1.0,
    q: float = 0.05,
) -> pl.DataFrame:
    """Reglas de asociacion con soporte, confianza, lift, leverage y conviccion.

    Restringe a los `top_n` SKU por soporte antes de probar: con miles de SKU la
    correccion por multiplicidad aniquila todo lo real. Filtra por conteo absoluto
    (`min_conteo`) antes que por lift: un lift de 8 sobre 12 tickets es ruido.
    Fisher exacto + q de Benjamini-Hochberg sobre todos los pares probados.
    """
    d = df.select([txn, sku]).drop_nulls().unique()
    N = d.select(pl.col(txn).n_unique()).item()
    if N < 2:
        return pl.DataFrame()

    conteo_sku = d.group_by(sku).agg(pl.len().alias("n")).sort("n", descending=True)
    vivos = conteo_sku.head(top_n)[sku]
    d = d.filter(pl.col(sku).is_in(vivos.to_list()))
    n_de = dict(zip(conteo_sku[sku].to_list(), conteo_sku["n"].to_list()))

    pares = (
        d.join(d, on=txn, suffix="_b")
        .filter(pl.col(sku) < pl.col(f"{sku}_b"))
        .group_by([sku, f"{sku}_b"])
        .agg(pl.len().alias("n11"))
        .filter(pl.col("n11") >= min_conteo)
    )
    if pares.is_empty():
        return pl.DataFrame()

    a = pares[sku].to_list()
    b = pares[f"{sku}_b"].to_list()
    n11 = pares["n11"].to_numpy().astype("float64")
    na = np.array([n_de[s] for s in a], dtype="float64")
    nb = np.array([n_de[s] for s in b], dtype="float64")

    sop_ab, sop_a, sop_b = n11 / N, na / N, nb / N
    conf_ab = n11 / na
    conf_ba = n11 / nb
    lift = sop_ab / (sop_a * sop_b)
    leverage = sop_ab - sop_a * sop_b
    with np.errstate(divide="ignore", invalid="ignore"):
        conviccion = np.where(conf_ab < 1, (1 - sop_b) / (1 - conf_ab), np.inf)

    p = np.empty(n11.size)
    for i in range(n11.size):
        n10 = na[i] - n11[i]
        n01 = nb[i] - n11[i]
        n00 = N - n11[i] - n10 - n01
        p[i] = stats.fisher_exact([[n11[i], n10], [n01, n00]], alternative="greater")[1]

    out = pl.DataFrame(
        {
            "a": a, "b": b,
            "n_ab": n11.astype("int64"),
            "soporte_ab": sop_ab, "soporte_a": sop_a, "soporte_b": sop_b,
            "confianza_a_b": conf_ab, "confianza_b_a": conf_ba,
            "lift": lift, "leverage": leverage, "conviccion": conviccion,
            "p_fisher": p, "q_fisher": _bh(p),
        }
    ).with_columns(
        (pl.col("q_fisher") < q).alias("significativa"),
        pl.when((pl.col("soporte_ab") >= pl.col("soporte_ab").median()) & (pl.col("lift") < 1.2))
        .then(pl.lit("ancla"))
        .when(pl.col("lift") >= 1.2)
        .then(pl.lit("afinidad"))
        .otherwise(pl.lit("ruido"))
        .alias("rol"),
    )
    return out.filter(pl.col("lift") >= min_lift).sort("leverage", descending=True)


# ════════════════════════════════════════════════════════════════════════════
#  4B. RFM y cohortes
# ════════════════════════════════════════════════════════════════════════════

def rfm(
    df: pl.DataFrame,
    cliente: str,
    fecha: str,
    monto: str,
    t_split: _dt.date | str | None = None,
    *,
    orden: str | None = None,
    min_dias_observacion: int = 0,
) -> pl.DataFrame:
    """RFM con corte estricto en `t_split`. Sin corte no hay analisis, hay fuga.

    R = dias desde la ultima compra hasta t_split
    F = ocasiones de compra distintas (por `orden` si se da, si no por fecha)
    M = gasto total en la ventana

    Devuelve tambien quintiles (R invertida: 5 = mas reciente) y el ticket medio,
    que es el insumo de la verificacion Gamma-Gamma (F vs M/F deben ser independientes).
    """
    d = df.select([c for c in {cliente, fecha, monto, orden} if c]).drop_nulls([cliente, fecha])
    d = d.with_columns(pl.col(fecha).cast(pl.Date))

    if t_split is None:
        t_split = d.select(pl.col(fecha).max()).item()
    elif isinstance(t_split, str):
        t_split = _dt.date.fromisoformat(t_split)

    d = d.filter(pl.col(fecha) <= pl.lit(t_split))
    if d.is_empty():
        return pl.DataFrame()

    freq = (pl.col(orden).n_unique() if orden else pl.col(fecha).n_unique()).alias("F")
    g = d.group_by(cliente).agg(
        (pl.lit(t_split) - pl.col(fecha).max()).dt.total_days().alias("R"),
        freq,
        pl.col(monto).sum().alias("M"),
        pl.col(fecha).min().alias("primera_compra"),
        pl.col(fecha).max().alias("ultima_compra"),
    ).with_columns(
        (pl.lit(t_split) - pl.col("primera_compra")).dt.total_days().alias("antiguedad_dias"),
        (pl.col("M") / pl.col("F")).alias("ticket_medio"),
    )

    if min_dias_observacion > 0:
        g = g.filter(pl.col("antiguedad_dias") >= min_dias_observacion)

    # intervalo entre compras: base de la definicion operativa de churn
    g = g.with_columns(
        pl.when(pl.col("F") > 1)
        .then((pl.col("ultima_compra") - pl.col("primera_compra")).dt.total_days()
              / (pl.col("F") - 1))
        .otherwise(None)
        .alias("ipi_medio")
    )

    def _quintil(col: str, invertir: bool = False) -> pl.Expr:
        r = pl.col(col).rank("average") / pl.len()
        r = (1 - r) if invertir else r
        return (r * 5).ceil().clip(1, 5).cast(pl.Int8)

    return (
        g.with_columns(
            _quintil("R", invertir=True).alias("R_score"),
            _quintil("F").alias("F_score"),
            _quintil("M").alias("M_score"),
        )
        .with_columns(
            (pl.col("R_score").cast(pl.Utf8) + pl.col("F_score").cast(pl.Utf8)
             + pl.col("M_score").cast(pl.Utf8)).alias("rfm"),
            pl.lit(str(t_split)).alias("t_split"),
        )
        .sort("M", descending=True)
    )


def supuestos_clv(rfm_df: pl.DataFrame) -> dict:
    """Verifica los supuestos de Pareto/NBD y Gamma-Gamma antes de ajustar nada.

    - sobredispersion de F (heterogeneidad Gamma esperada: razon > 1)
    - bimodalidad de F (indica dos poblaciones: retail y B2B -> segmentar antes)
    - independencia F vs ticket medio (requisito de Gamma-Gamma; |rho| > 0.1 la invalida)
    """
    f = _a_numpy(rfm_df["F"])
    out = {
        "n_clientes": int(f.size),
        "media_F": float(f.mean()) if f.size else float("nan"),
        "var_F": float(f.var(ddof=1)) if f.size > 1 else float("nan"),
    }
    out["razon_dispersion"] = out["var_F"] / out["media_F"] if out["media_F"] else float("nan")
    out["sobredispersa"] = bool(out["razon_dispersion"] > 1.2)

    rec = rfm_df.filter(pl.col("F") > 1)
    if rec.height > 30:
        a = _a_numpy(rec["F"])
        b = _a_numpy(rec["ticket_medio"])
        n = min(a.size, b.size)
        rho, p = stats.spearmanr(a[:n], b[:n])
        out["rho_F_ticket"] = float(rho)
        out["p_F_ticket"] = float(p)
        out["gamma_gamma_valido"] = bool(abs(rho) <= 0.1)
    out["_nota"] = ("razon_dispersion <= 1 o F bimodal -> no aplicar BG/NBD sin segmentar; "
                    "ver statistical-methods.md §5")
    return out


def cohortes(
    df: pl.DataFrame,
    cliente: str,
    fecha: str,
    monto: str | None = None,
    *,
    granularidad: str = "1mo",
) -> pl.DataFrame:
    """Retencion por cohorte de adquisicion. Se calcula sobre cohorte, nunca sobre la base."""
    d = (
        df.select([c for c in {cliente, fecha, monto} if c])
        .drop_nulls([cliente, fecha])
        .with_columns(pl.col(fecha).cast(pl.Date))
    )
    d = d.with_columns(
        pl.col(fecha).min().over(cliente).dt.truncate(granularidad).alias("cohorte"),
        pl.col(fecha).dt.truncate(granularidad).alias("periodo"),
    )
    tam = d.group_by("cohorte").agg(pl.col(cliente).n_unique().alias("tam_cohorte"))
    act = d.group_by(["cohorte", "periodo"]).agg(pl.col(cliente).n_unique().alias("activos"))
    return (
        act.join(tam, on="cohorte")
        .with_columns(
            (pl.col("activos") / pl.col("tam_cohorte")).alias("retencion"),
            ((pl.col("periodo").dt.year() - pl.col("cohorte").dt.year()) * 12
             + (pl.col("periodo").dt.month() - pl.col("cohorte").dt.month())).alias("periodo_n"),
        )
        .sort(["cohorte", "periodo_n"])
    )


# ════════════════════════════════════════════════════════════════════════════
#  4C. Elasticidad precio
# ════════════════════════════════════════════════════════════════════════════

def elasticidad(
    df: pl.DataFrame,
    precio: str,
    cantidad: str,
    *,
    controles: Sequence[str] = (),
    efectos_fijos: Sequence[str] = (),
    max_niveles_fe: int = 200,
) -> dict:
    """Elasticidad precio en forma log-log con errores HC1 y prueba Breusch-Pagan.

        ln(Q) = a + b*ln(P) + g'X + FE + e        ->  b es la elasticidad

    `efectos_fijos` acepta columnas categoricas (sku, semana, tienda): se expanden
    a dummies. La especificacion minima creible incluye FE de SKU y de tiempo.

    Signo esperado NEGATIVO. Un b positivo es fallo de especificacion (endogeneidad,
    ver gotchas.md §2), no un hallazgo. La funcion lo marca pero no lo corrige.
    """
    cols = [precio, cantidad, *controles, *efectos_fijos]
    d = df.select([c for c in cols if c in df.columns]).drop_nulls()
    d = d.filter((pl.col(precio) > 0) & (pl.col(cantidad) > 0))
    if d.height < 30:
        return {"error": "menos de 30 observaciones con precio y cantidad positivos"}

    y = np.log(d[cantidad].to_numpy().astype("float64"))
    partes = [np.ones((d.height, 1)), np.log(d[precio].to_numpy().astype("float64")).reshape(-1, 1)]
    nombres = ["const", f"ln_{precio}"]

    for c in controles:
        v = d[c]
        col = v.to_numpy().astype("float64").reshape(-1, 1)
        if (col > 0).all():
            col = np.log(col)
            nombres.append(f"ln_{c}")
        else:
            nombres.append(c)
        partes.append(col)

    for c in efectos_fijos:
        niveles = d[c].unique().sort().to_list()
        if len(niveles) > max_niveles_fe:
            raise ValueError(f"'{c}' tiene {len(niveles)} niveles (> {max_niveles_fe}); "
                             "agrupar antes o usar within-transformation")
        v = d[c].to_list()
        for niv in niveles[1:]:                      # una categoria omitida
            partes.append(np.array([1.0 if x == niv else 0.0 for x in v]).reshape(-1, 1))
            nombres.append(f"{c}={niv}")

    X = np.hstack(partes)
    res = _ols_hc1(y, X)
    b = float(res["beta"][1])
    se = float(res["se_hc1"][1])

    return {
        "elasticidad": b,
        "se_hc1": se,
        "ic95": (b - 1.96 * se, b + 1.96 * se),
        "p_hc1": float(res["p_hc1"][1]),
        "r2": res["r2"],
        "n": res["n"],
        "bp_lm": res["bp_lm"],
        "bp_p": res["bp_p"],
        "heterocedastico": res["heterocedastico"],
        "signo_sospechoso": bool(b > 0),
        "coeficientes": pl.DataFrame(
            {"termino": nombres, "beta": res["beta"],
             "se_hc1": res["se_hc1"], "p": res["p_hc1"]}
        ),
        "_nota": ("b > 0 o |b| muy pequeno -> el precio se esta moviendo con la demanda "
                  "(liquidacion, temporada). Anadir FE de tiempo o instrumentar."),
    }


def uplift(
    df: pl.DataFrame,
    fecha: str,
    cantidad: str,
    promo: str,
    *,
    grupo: str | None = None,
) -> pl.DataFrame:
    """Baseline y uplift incremental. El baseline se ajusta SOLO fuera de promocion.

    Ajustar sobre la serie completa deja que la promocion contamine el baseline y
    subestima el uplift. Aqui el baseline es la media de periodos no promocionales
    por grupo (sustituir por un modelo con estacionalidad cuando haya suficiente
    historia; el punto metodologico es el mismo: nunca ver el flag de promo).
    """
    llaves = [grupo] if grupo else []
    d = df.select([fecha, cantidad, promo, *llaves]).drop_nulls()

    base = (
        d.filter(~pl.col(promo).cast(pl.Boolean))
        .group_by(llaves if llaves else pl.lit(1).alias("_todo"))
        .agg(pl.col(cantidad).mean().alias("baseline"))
    )
    if not llaves:
        base = base.drop("_todo")
        d = d.with_columns(pl.lit(base["baseline"][0]).alias("baseline"))
    else:
        d = d.join(base, on=llaves, how="left")

    return (
        d.filter(pl.col(promo).cast(pl.Boolean))
        .with_columns(
            (pl.col(cantidad) - pl.col("baseline")).alias("uplift"),
            ((pl.col(cantidad) - pl.col("baseline")) / pl.col("baseline")).alias("lift_pct"),
        )
        .sort(fecha)
    )


# ════════════════════════════════════════════════════════════════════════════
#  4D. ABC-XYZ
# ════════════════════════════════════════════════════════════════════════════

def abc_xyz(
    df: pl.DataFrame,
    sku: str,
    ingreso: str,
    periodo: str,
    *,
    cantidad: str | None = None,
    rellenar_ceros: bool = True,
) -> pl.DataFrame:
    """Matriz ABC-XYZ de 9 celdas.

    ABC por contribucion acumulada al ingreso (80 / 95 / 100).
    XYZ por CV de la demanda por periodo (0.5 / 1.0).

    `rellenar_ceros=True` completa la rejilla SKU x periodo con ceros: omitir los
    periodos sin demanda hace que todo SKU intermitente parezca estable, que es
    justo al reves. Anade ADI y CV^2 para la clasificacion Syntetos-Boylan, mas
    informativa que el CV cuando la demanda es intermitente.
    """
    demanda = cantidad or ingreso
    d = df.select([sku, periodo, ingreso] + ([cantidad] if cantidad else [])).drop_nulls()
    d = d.group_by([sku, periodo]).agg(
        pl.col(ingreso).sum().alias("_ing"),
        *( [pl.col(cantidad).sum().alias("_cant")] if cantidad else [] ),
    )
    col_dem = "_cant" if cantidad else "_ing"

    if rellenar_ceros:
        rejilla = d.select(sku).unique().join(d.select(periodo).unique(), how="cross")
        d = rejilla.join(d, on=[sku, periodo], how="left").with_columns(
            pl.col("_ing").fill_null(0),
            *( [pl.col("_cant").fill_null(0)] if cantidad else [] ),
        )

    g = d.group_by(sku).agg(
        pl.col("_ing").sum().alias("ingreso_total"),
        pl.col(col_dem).mean().alias("demanda_media"),
        pl.col(col_dem).std(ddof=1).alias("demanda_sd"),
        pl.len().alias("periodos"),
        (pl.col(col_dem) > 0).sum().alias("periodos_con_demanda"),
    ).with_columns(
        pl.when(pl.col("demanda_media") > 0)
        .then(pl.col("demanda_sd") / pl.col("demanda_media"))
        .otherwise(None)
        .alias("cv"),
        pl.when(pl.col("periodos_con_demanda") > 0)
        .then(pl.col("periodos") / pl.col("periodos_con_demanda"))
        .otherwise(None)
        .alias("adi"),
    ).with_columns((pl.col("cv") ** 2).alias("cv2"))

    g = g.sort("ingreso_total", descending=True).with_columns(
        (pl.col("ingreso_total").cum_sum() / pl.col("ingreso_total").sum()).alias("acum_ingreso")
    ).with_columns(
        pl.when(pl.col("acum_ingreso") <= 0.80).then(pl.lit("A"))
        .when(pl.col("acum_ingreso") <= 0.95).then(pl.lit("B"))
        .otherwise(pl.lit("C")).alias("abc"),
        pl.when(pl.col("cv").is_null()).then(pl.lit("?"))
        .when(pl.col("cv") < 0.5).then(pl.lit("X"))
        .when(pl.col("cv") <= 1.0).then(pl.lit("Y"))
        .otherwise(pl.lit("Z")).alias("xyz"),
        # Syntetos-Boylan: ADI 1.32 / CV^2 0.49
        pl.when(pl.col("adi").is_null()).then(pl.lit("?"))
        .when((pl.col("adi") > 1.32) & (pl.col("cv2") > 0.49)).then(pl.lit("grumosa"))
        .when(pl.col("adi") > 1.32).then(pl.lit("intermitente"))
        .when(pl.col("cv2") > 0.49).then(pl.lit("erratica"))
        .otherwise(pl.lit("suave")).alias("patron"),
    )
    return g.with_columns((pl.col("abc") + pl.col("xyz")).alias("celda"))


# ════════════════════════════════════════════════════════════════════════════
#  6. Sintesis ejecutiva
# ════════════════════════════════════════════════════════════════════════════

def kpis(
    df: pl.DataFrame,
    *,
    txn: str,
    ingreso: str,
    cantidad: str | None = None,
    costo: str | None = None,
    inventario_costo: float | None = None,
    unidades_recibidas: float | None = None,
    unidades_en_mano: float | None = None,
) -> dict:
    """AOV, UPT, AUR, margen, GMROI y sell-through. Todo como razon de sumas.

    Nunca promedio de razones: ver gotchas.md §4 (paradoja de Simpson).
    """
    n_txn = df.select(pl.col(txn).n_unique()).item()
    rev = float(df[ingreso].sum())
    out = {"transacciones": int(n_txn), "ingreso_neto": rev}

    if n_txn:
        out["aov"] = rev / n_txn
        aov_txn = df.group_by(txn).agg(pl.col(ingreso).sum().alias("v"))["v"]
        out["aov_mediano"] = float(aov_txn.median())

    if cantidad:
        u = float(df[cantidad].sum())
        out["unidades"] = u
        if n_txn:
            out["upt"] = u / n_txn
        if u:
            out["aur"] = rev / u

    if costo:
        cogs = float(df[costo].sum())
        out["cogs"] = cogs
        out["margen_bruto"] = rev - cogs
        if rev:
            out["margen_pct"] = (rev - cogs) / rev
        if inventario_costo:
            out["gmroi"] = (rev - cogs) / inventario_costo
            out["rotacion"] = cogs / inventario_costo
            if cogs:
                out["dias_de_inventario"] = 365 * inventario_costo / cogs

    if cantidad and unidades_recibidas:
        out["sell_through_compra"] = float(df[cantidad].sum()) / unidades_recibidas
    if cantidad and unidades_en_mano is not None:
        u = float(df[cantidad].sum())
        if u + unidades_en_mano:
            out["sell_through_posicion"] = u / (u + unidades_en_mano)
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Self-test
# ════════════════════════════════════════════════════════════════════════════

def _datos_sinteticos(seed: int = 7) -> pl.DataFrame:
    """Panel retail sintetico con cola pesada, devoluciones, huecos y afinidad real."""
    rng = np.random.default_rng(seed)
    n_txn, n_sku, n_cli = 4000, 40, 800
    inicio = _dt.date(2025, 1, 1)

    filas = []
    for t in range(n_txn):
        dia = int(rng.integers(0, 330))
        if 150 <= dia < 157:                       # hueco de captura: sistema caido
            continue
        fecha = inicio + _dt.timedelta(days=dia)
        cli = int(rng.integers(0, n_cli))
        n_lineas = 1 + int(rng.poisson(1.6))
        skus = list(rng.choice(n_sku, size=min(n_lineas, n_sku), replace=False))
        if 0 in skus and rng.random() < 0.55:      # afinidad real sku_00 -> sku_01
            skus.append(1)
        for s in set(skus):
            precio = float(np.exp(rng.normal(2.2, 0.35))) * (1 + 0.03 * (s % 5))
            promo = rng.random() < 0.15
            if promo:
                precio *= 0.75
            # demanda con elasticidad verdadera ~ -1.6. El nivel se elige para que
            # la media de Poisson quede lejos de 0: censurar en 1 destruye la senal.
            mu = np.exp(5.5 - 1.6 * np.log(precio) + rng.normal(0, 0.3))
            cant = max(1, int(rng.poisson(mu)))
            filas.append(
                {
                    "txn_id": f"T{t:06d}", "linea": int(s), "sku": f"SKU_{s:02d}",
                    "cliente": f"C{cli:04d}", "fecha": fecha,
                    "cantidad": cant, "precio": round(precio, 2),
                    "importe": round(cant * precio, 2),
                    "costo": round(cant * precio * rng.uniform(0.45, 0.75), 2),
                    "promo": bool(promo),
                }
            )
    df = pl.DataFrame(filas)

    # devoluciones: ~4% de las lineas, con signo negativo en ambas columnas
    idx = rng.choice(df.height, size=int(0.04 * df.height), replace=False)
    dev = df[idx].with_columns(
        (-pl.col("cantidad")).alias("cantidad"),
        (-pl.col("importe")).alias("importe"),
        (-pl.col("costo")).alias("costo"),
    )
    return pl.concat([df, dev]).with_columns(
        pl.col("fecha").dt.truncate("1w").alias("semana")
    )


def _self_test() -> int:
    print("retail_eda self-test")
    print(f"  polars {pl.__version__} | numpy {np.__version__} | sklearn: {_TIENE_SKLEARN}")

    df = _datos_sinteticos()
    print(f"  datos sinteticos: {df.height:,} lineas x {df.width} columnas")

    ok = True

    def check(nombre, cond, detalle=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok ' if cond else 'FAIL'}] {nombre}{(' — ' + detalle) if detalle else ''}")

    # 1 integridad
    ig = integridad(df, clave=["txn_id", "sku"], cols_monetarias=["importe", "costo"],
                    cols_cantidad=["cantidad"])
    check("integridad", ig["devoluciones"] > 0, f"{ig['devoluciones']} devoluciones detectadas")

    dens = densidad_temporal(df, "fecha")
    check("densidad_temporal", dens.filter(pl.col("hueco")).height >= 5,
          f"{dens.filter(pl.col('hueco')).height} dias sin captura")

    # 2 univariado (solo ventas: las devoluciones van aparte)
    ventas = df.filter(pl.col("cantidad") > 0)
    uni = univariado(ventas["importe"])
    check("univariado", uni["n"] > 0 and "hill_alpha" in uni,
          f"asimetria {uni['asimetria']:.2f}, hill alpha {uni['hill_alpha']:.2f}")
    check("mejor_ajuste", not uni["ajustes"].is_empty(),
          f"mejor familia: {uni['ajustes']['familia'][0]}")

    # 3 dependencias
    dep = dependencias(ventas.select(["cantidad", "precio", "importe", "costo"]),
                       ["cantidad", "precio", "importe", "costo"], n_max=8000)
    check("dependencias", not dep.is_empty() and "q_spearman" in dep.columns,
          f"{dep.height} pares, {dep['significativa'].sum()} significativos")

    mah = mahalanobis(ventas.select(["cantidad", "precio", "importe"]),
                      ["cantidad", "precio", "importe"], robusto=False, log1p=True)
    check("mahalanobis", not mah.is_empty(), f"{int(mah['atipico'].sum())} atipicos")

    # 4A market basket
    mb = market_basket(ventas, "txn_id", "sku", top_n=40, min_conteo=20)
    par = mb.filter((pl.col("a") == "SKU_00") & (pl.col("b") == "SKU_01"))
    check("market_basket", not mb.is_empty(), f"{mb.height} reglas")
    check("market_basket detecta la afinidad sembrada",
          par.height == 1 and par["lift"][0] > 1.2,
          f"lift {par['lift'][0]:.2f}" if par.height else "par no encontrado")

    # 4B RFM
    r = rfm(ventas, "cliente", "fecha", "importe", t_split="2025-10-01", orden="txn_id")
    check("rfm", r.height > 0 and int(r["R"].min()) >= 0,
          f"{r.height} clientes, R mediana {int(r['R'].median())}d")
    check("rfm respeta el corte", r["ultima_compra"].max() <= _dt.date(2025, 10, 1))
    sup = supuestos_clv(r)
    check("supuestos_clv", "razon_dispersion" in sup,
          f"dispersion {sup['razon_dispersion']:.2f}, gamma-gamma "
          f"{sup.get('gamma_gamma_valido')}")
    coh = cohortes(ventas, "cliente", "fecha")
    check("cohortes", not coh.is_empty(), f"{coh['cohorte'].n_unique()} cohortes")

    # 4C elasticidad
    el = elasticidad(ventas, "precio", "cantidad", efectos_fijos=["sku"])
    check("elasticidad recupera el signo y la magnitud",
          -2.4 < el["elasticidad"] < -1.0,
          f"b = {el['elasticidad']:.3f} (verdadero -1.6), HC1 {el['se_hc1']:.3f}")
    up = uplift(ventas, "fecha", "cantidad", "promo", grupo="sku")
    check("uplift", not up.is_empty(), f"{up.height} filas en promocion")

    # 4D ABC-XYZ
    ax = abc_xyz(ventas, "sku", "importe", "semana", cantidad="cantidad")
    check("abc_xyz", ax.height > 0 and ax["celda"].n_unique() >= 2,
          f"celdas: {sorted(ax['celda'].unique().to_list())}")

    # 6 KPIs
    k = kpis(ventas, txn="txn_id", ingreso="importe", cantidad="cantidad",
             costo="costo", inventario_costo=50_000, unidades_en_mano=20_000)
    check("kpis", "aov" in k and "gmroi" in k,
          f"AOV {k['aov']:.2f}, UPT {k['upt']:.2f}, GMROI {k['gmroi']:.2f}")

    print("\n  RESULTADO:", "todo ok" if ok else "hay fallos")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
