#!/usr/bin/env bash
# Fetch each public PPI source to a temp file (resumable), filter it to TMP
# proteins with extract_public.py, then delete the raw file.
# Usage: fetch_public.sh <source> [<source> ...]   (run `idmap` first)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TMPDIR_DL="${TMPDIR_DL:-$HERE/../../data/TMP/public_db/_download}"
mkdir -p "$TMPDIR_DL"

url_of() {
  case "$1" in
    idmap)   echo "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz" ;;
    string)  echo "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz" ;;
    biogrid) echo "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ORGANISM-LATEST.tab3.zip" ;;
    intact)  echo "https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/species/human.zip" ;;
    hippie)  echo "https://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/HIPPIE-current.mitab.txt" ;;
    huri)    echo "https://www.interactome-atlas.org/data/HI-union.tsv" ;;
    bioplex) echo "https://bioplex.hms.harvard.edu/data/BioPlex_293T_Network_10K_Dec_2019.tsv https://bioplex.hms.harvard.edu/data/BioPlex_HCT116_Network_5.5K_Dec_2019.tsv" ;;
    corum)   echo "https://mips.helmholtz-muenchen.de/fastapi-corum/public/file/download_current_file?file_id=human&file_format=txt" ;;
    *) echo "unknown source $1" >&2; return 1 ;;
  esac
}

download() {  # resumable; retries until the server-reported size is reached
  local url="$1" out="$2" try
  for try in $(seq 1 30); do
    if curl -sSL --fail -C - --connect-timeout 30 -o "$out" "$url"; then return 0; fi
    echo "  retry $try for $url" >&2; sleep 5
  done
  return 1
}

decode() {  # print the source's text to stdout
  case "$1" in
    *.gz)  gzip -dc "$1" ;;
    *.zip) case "$2" in
             biogrid) bsdtar -xOf "$1" '*Homo_sapiens*' ;;
             *)       bsdtar -xOf "$1" ;;
           esac ;;
    *)     cat "$1" ;;
  esac
}

for src in "$@"; do
  files=()
  for url in $(url_of "$src"); do
    name="$(basename "${url%%\?*}")"; [ "$src" = corum ] && name="corum_human.txt"
    out="$TMPDIR_DL/$name"
    echo "[$src] downloading $url" >&2
    download "$url" "$out"
    case "$out" in *.gz) gzip -t "$out" ;; *.zip) bsdtar -tf "$out" >/dev/null ;; esac
    files+=("$out")
  done
  if [ "$src" = bioplex ]; then
    # Concatenate both cell lines under one header.
    { cat "${files[0]}"; tail -n +2 "${files[1]}"; } | python3 "$HERE/extract_public.py" "$src"
  else
    decode "${files[0]}" "$src" | python3 "$HERE/extract_public.py" "$src"
  fi
  rm -f "${files[@]}"
done
rmdir "$TMPDIR_DL" 2>/dev/null || true
