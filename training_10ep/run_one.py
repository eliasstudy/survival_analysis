"""Trainiert EIN Modell auf EINER Kohorte mit abweichender Epochenzahl.

Nimmt den Code direkt aus dem Trainings-Notebook bzw. train_<KOHORTE>.py, ersetzt in der Config nur
"cohort" und "epochs", schreibt das Ergebnis nach generated/ und fuehrt es im Modell-Ordner aus
(-> Ergebnisse landen wie gewohnt in models/<MODELL>/runs/). Die Originaldateien bleiben unveraendert.

Vor dem Training wird die Config mit dem neuesten 20-Epochen-Run derselben Kohorte verglichen:
außer "epochs" muss alles gleich sein, sonst waeren die Ergebnisse nicht vergleichbar.

Aufruf (mit dem Python des passenden Conda-Environments):
    python run_one.py <MODELL-ORDNER> <KOHORTE> <EPOCHEN> [--check]
"""
import ast
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"
NOTEBOOKS = {
    "ABMIL": "abmil.ipynb", "TRANSMIL": "transmil.ipynb", "MCAT": "MCAT.ipynb", "MOTCAT": "MOTCAT.ipynb",
    "SURVPATH": "SURVPATH.ipynb", "healnet": "healnet.ipynb", "SNN": "SNN.ipynb", "MLP": "MLP.ipynb",
    "SNN_20k": None, "MLP_20k": None,  # None -> train_<KOHORTE>.py
}
REFERENCE_EPOCHS = 20


def load_source(model: str, cohort: str) -> str:
    if NOTEBOOKS[model] is None:
        return (MODELS / model / f"train_{cohort.removeprefix('TCGA-')}.py").read_text()
    nb = json.loads((MODELS / model / NOTEBOOKS[model]).read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return "\n\n".join(f"# ===== cell {i}\n{s}" for i, s in enumerate(cells))


def patch(source: str, key: str, value: str) -> str:
    source, n = re.subn(rf'^(\s*"{key}":\s*)("[^"]*"|\d+)', rf"\g<1>{value}", source, flags=re.M)
    if n != 1:
        sys.exit(f"FEHLER: '{key}' {n}x in der Config gefunden (erwartet: 1x)")
    return source


def config_literal(source: str) -> dict:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "config":
            return ast.literal_eval(node.value)
    sys.exit("FEHLER: keine config gefunden")


def check_against_reference(model: str, cohort: str, config: dict):
    runs = []
    for path in (MODELS / model / "runs").glob(f"{cohort}_*/summary.json"):
        summary = json.loads(path.read_text())
        if summary.get("epochs") == REFERENCE_EPOCHS:
            runs.append(path.parent)
    if not runs:
        sys.exit(f"FEHLER: kein {REFERENCE_EPOCHS}-Epochen-Run fuer {model} / {cohort}")
    ref_dir = max(runs, key=lambda p: p.name[-15:])
    ref = json.loads((ref_dir / "config.json").read_text())
    # Schluessel, die erst zur Laufzeit ergaenzt werden (z.B. omic_sizes), fehlen im Literal
    runtime = {"omic_sizes", "n_omic_features", "n_pathways"}
    keys = (set(ref) | set(config)) - runtime - {"epochs"}
    diff = {k: (ref.get(k, "<fehlt>"), config.get(k, "<fehlt>")) for k in keys if ref.get(k, "<fehlt>") != config.get(k, "<fehlt>")}
    if diff:
        sys.exit(f"FEHLER: Config weicht von {ref_dir.name} ab (alt, neu): {diff}")
    print(f"Config identisch zu {ref_dir.name} (bis auf epochs)")


def main():
    model, cohort, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
    source = load_source(model, cohort)
    source = patch(source, "cohort", f'"{cohort}"')
    source = patch(source, "epochs", str(epochs))
    check_against_reference(model, cohort, config_literal(source))

    header = "import os, sys\nsys.path.insert(0, os.getcwd())  # wie im Notebook\ndisplay = print\n\n"
    script = HERE / "generated" / f"{model}_{cohort}_{epochs}ep.py"
    script.parent.mkdir(exist_ok=True)
    script.write_text(header + source + "\n")
    compile(script.read_text(), str(script), "exec")
    if "--check" in sys.argv:
        return

    sys.stdout.flush()
    os.chdir(MODELS / model)
    os.execv(sys.executable, [sys.executable, "-u", str(script)])


if __name__ == "__main__":
    main()
