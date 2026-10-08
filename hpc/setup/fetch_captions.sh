#!/bin/bash
# Download Part_Aa val captions from the instructor's Google Drive folder.
# MUST run on a LOGIN NODE — compute nodes have no internet by default.
#
# Drive folder (from piazza_faqs.txt):
#   https://drive.google.com/drive/folders/1I_aR_6KMaO-Ctagg65A-0PMmftbZPTfV

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$REPO_ROOT/A2_dataset/Part_Aa/Clevr_official/captions"
DRIVE_FOLDER="https://drive.google.com/drive/folders/1I_aR_6KMaO-Ctagg65A-0PMmftbZPTfV"

mkdir -p "$DEST"

# gdown is declared in environment.yml; activate the env so it's on PATH.
if ! command -v gdown >/dev/null 2>&1; then
    echo "[fetch] gdown not on PATH; activating conda env"
    source "$HOME/condaBaseSetup"
    conda activate col775_a2
fi

echo "[fetch] downloading Drive folder -> $DEST"
gdown --folder "$DRIVE_FOLDER" -O "$DEST"

# The Drive folder structure is unknown a priori — if the download created a
# nested directory, flatten the JSON(s) so retrieval.py finds them directly.
find "$DEST" -name "clevr_val_captions.json" | while read src; do
    if [ "$src" != "$DEST/clevr_val_captions.json" ]; then
        mv "$src" "$DEST/clevr_val_captions.json"
        echo "[fetch] moved $src -> $DEST/clevr_val_captions.json"
    fi
done

ls -l "$DEST"
