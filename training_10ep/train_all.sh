#!/usr/bin/env bash
# Trainiert alle Modelle auf allen Kohorten mit 10 Epochen, nacheinander (eine GPU, eine CSV pro Modell).
# Start (ueberlebt SSH-Abbrueche):  tmux new -d -s training "bash train_all.sh"
# Fortschritt:                      tail -f logs/progress.log      bzw.  tmux attach -t training
# Bereits erfolgreich beendete Laeufe (logs/<MODELL>_<KOHORTE>.done) werden bei einem Neustart uebersprungen.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
EPOCHS=10
ENVS=/loctmp/mue37538/miniconda3/envs
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
mkdir -p logs

# Modell-Ordner : Conda-Environment (schnelle Modelle zuerst)
JOBS="MLP:survpath SNN:survpath MLP_20k:survpath SNN_20k:survpath ABMIL:mil healnet:healnet MCAT:mcat SURVPATH:survpath TRANSMIL:transmil MOTCAT:motcat"
COHORTS="TCGA-KIRP TCGA-KIRC TCGA-GBMLGG"
TOTAL=$(( $(wc -w <<<"$JOBS") * $(wc -w <<<"$COHORTS") ))

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/progress.log; }

log "Start: $TOTAL Laeufe mit $EPOCHS Epochen"
i=0; failed=0
for cohort in $COHORTS; do
    for job in $JOBS; do
        model=${job%%:*}; env=${job##*:}; i=$((i + 1))
        name="${model}_${cohort}"
        if [[ -f logs/$name.done ]]; then log "($i/$TOTAL) $name bereits fertig, uebersprungen"; continue; fi
        log "($i/$TOTAL) START $name  (Log: logs/$name.log)"
        start=$SECONDS
        "$ENVS/$env/bin/python" run_one.py "$model" "$cohort" "$EPOCHS" 2>&1 | tee "logs/$name.log"
        rc=${PIPESTATUS[0]}
        dur="$(( (SECONDS - start) / 60 ))m $(( (SECONDS - start) % 60 ))s"
        if [[ $rc -eq 0 ]]; then
            touch "logs/$name.done"
            log "($i/$TOTAL) OK    $name nach $dur | $(grep -F '5-Fold CV' "logs/$name.log" | tail -1)"
        else
            failed=$((failed + 1))
            log "($i/$TOTAL) FEHLER $name (Exit-Code $rc) nach $dur"
        fi
    done
done
log "Training beendet: $failed von $TOTAL Laeufen fehlgeschlagen"

# Tabellen in eval.ipynb neu berechnen und im Notebook speichern
if "$ENVS/motcat/bin/python" -m jupyter nbconvert --to notebook --execute --inplace ../evaluation/eval.ipynb >logs/eval.log 2>&1; then
    log "evaluation/eval.ipynb aktualisiert"
else
    log "FEHLER beim Ausfuehren von eval.ipynb (siehe logs/eval.log)"
fi
log "FERTIG"
