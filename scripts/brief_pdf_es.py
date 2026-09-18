#!/usr/bin/env python3
"""Genera el brief ejecutivo en PDF con los rotulos de seccion en espanol.

Reusa integramente el layout de la skill executive-brief-hook; lo unico que cambia son
las 7 etiquetas que el script original trae fijas en ingles. Se sustituyen sobre la
fuente en memoria para no modificar la skill instalada, que es global al usuario.

    uv run python scripts/brief_pdf_es.py payload.json salida.pdf
"""
import pathlib
import sys
import types

SKILL = pathlib.Path.home() / ".claude/skills/executive-brief-hook/scripts/generate_brief_pdf.py"

ROTULOS = {
    "THE HOOK:": "EL HOOK:",
    "BOTTOM LINE UP FRONT (BLUF)": "CONCLUSION PRINCIPAL",
    "CORE DRIVERS &amp; EVIDENCE": "DRIVERS Y EVIDENCIA",
    "DECISION MATRIX": "MATRIZ DE DECISION",
    "ACTION REQUIRED:": "DECISION REQUERIDA:",
    "DEADLINE:": "FECHA LIMITE:",
    "OWNER:": "RESPONSABLE:",
}


def cargar_generador():
    """Importa el modulo de la skill con los rotulos ya traducidos."""
    src = SKILL.read_text(encoding="utf-8")
    for en, es in ROTULOS.items():
        if en not in src:
            raise SystemExit(f"La skill cambio: ya no contiene el rotulo {en!r}. Revisa {SKILL}")
        src = src.replace(en, es)
    mod = types.ModuleType("brief_es")
    mod.__file__ = str(SKILL)
    exec(compile(src, str(SKILL), "exec"), mod.__dict__)
    return mod


def main(payload, salida):
    import json
    mod = cargar_generador()
    d = json.loads(pathlib.Path(payload).read_text(encoding="utf-8"))
    mod.generate_pdf(salida, d["title"], d["hook"], d["bluf"], d["drivers"],
                     d["table"], d["action"], subtitle=d.get("subtitle"),
                     tone=d.get("tone", "risk"))
    print(f"Escrito {salida}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
