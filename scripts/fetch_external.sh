#!/usr/bin/env bash
# Download the open datasets into external/<source>/. Idempotent: verified files are skipped,
# partial files resume. Downloads are data; nothing in them is ever executed.
#
#   scripts/fetch_external.sh [metrica] [idsse] [skillcorner]      (default: all three)
#
# Manifests that pin what was fetched live in docs/manifests/ so they can be committed;
# external/ itself is gitignored.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/external"
MAN="$ROOT/docs/manifests"
PY="$ROOT/.venv/bin/python"
HELPER="$ROOT/scripts/fetch_manifest.py"
mkdir -p "$EXT" "$MAN"

sha256_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
size_of() { stat -f %z "$1"; }

# Shallow clone that keeps LFS pointers as plain files. A global git-lfs filter config with no
# git-lfs binary installed makes an ordinary clone fail at checkout, so switch the filter off here.
clone_pointers() {
  GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 \
    -c filter.lfs.process= -c filter.lfs.smudge=cat -c filter.lfs.clean=cat \
    -c filter.lfs.required=false "$1" "$2"
}

# download <url> <dest> <size> : resume until the size matches, with a few retries.
download() {
  local url="$1" dest="$2" want="$3" tries=0
  mkdir -p "$(dirname "$dest")"
  while :; do
    if [ -f "$dest" ] && [ "$(size_of "$dest")" = "$want" ]; then return 0; fi
    tries=$((tries + 1))
    if [ "$tries" -gt 6 ]; then
      echo "FAILED after retries: $url" >&2
      return 1
    fi
    # An oversized leftover cannot be resumed.
    if [ -f "$dest" ] && [ "$(size_of "$dest")" -gt "$want" ]; then rm -f "$dest"; fi
    curl -sS -L --fail --retry 3 --retry-delay 5 -C - -o "$dest" "$url" \
      || { echo "curl failed (try $tries): $url" >&2; sleep 10; }
  done
}

fetch_metrica() {
  local dir="$EXT/metrica"
  if [ ! -d "$dir/.git" ]; then
    clone_pointers https://github.com/metrica-sports/sample-data "$dir"
  fi
  git -C "$dir" rev-parse HEAD > "$MAN/metrica.commit"
  echo "metrica: commit $(cat "$MAN/metrica.commit"), $(du -sh "$dir" | cut -f1)"
}

fetch_idsse() {
  local dir="$EXT/idsse" listing="$MAN/idsse_listing.json" manifest="$MAN/idsse_manifest.tsv"
  local base="https://huggingface.co/datasets/pysport/idsse-data/resolve/main"
  local failed=0 count=0 bytes=0
  mkdir -p "$dir"
  if [ ! -s "$listing" ]; then
    curl -sS -L --fail -o "$listing" \
      "https://huggingface.co/api/datasets/pysport/idsse-data/tree/main?recursive=true"
  fi
  "$PY" -I "$HELPER" hf "$listing" "$manifest"
  while IFS=$'\t' read -r path size algo digest; do
    local dest="$dir/$path"
    download "$base/$path" "$dest" "$size" || { failed=$((failed + 1)); continue; }
    if [ "$algo" = "sha256" ]; then
      local got; got="$(sha256_of "$dest")"
    else
      local got; got="$(git hash-object --no-filters "$dest")"
    fi
    if [ "$got" != "$digest" ]; then
      echo "IDSSE checksum mismatch, removing: $path" >&2
      rm -f "$dest"
      failed=$((failed + 1))
      continue
    fi
    count=$((count + 1)); bytes=$((bytes + size))
  done < "$manifest"
  echo "idsse: $count files verified, $bytes bytes, $failed failed"
  [ "$failed" = 0 ]
}

fetch_skillcorner() {
  local dir="$EXT/skillcorner" manifest="$MAN/skillcorner_lfs.tsv"
  if [ ! -d "$dir/.git" ]; then
    clone_pointers https://github.com/SkillCorner/opendata "$dir"
  fi
  local ref; ref="$(git -C "$dir" rev-parse HEAD)"
  echo "$ref" > "$MAN/skillcorner.commit"

  # Pointers come from the git objects, so this still works after real files replaced them.
  local dump; dump="$(mktemp)"
  git -C "$dir" ls-files -z | while IFS= read -r -d '' path; do
    local head; head="$(git -C "$dir" cat-file -p "HEAD:$path" | head -c 40 | tr -d '\0' || true)"
    case "$head" in
      "version https://git-lfs.github.com/spec/"*)
        printf '@@ %s\n' "$path" >> "$dump"
        git -C "$dir" cat-file -p "HEAD:$path" >> "$dump"
        ;;
    esac
  done
  "$PY" -I "$HELPER" lfs "$dump" "$manifest"
  rm -f "$dump"

  local base="https://media.githubusercontent.com/media/SkillCorner/opendata/$ref"
  local failed=0 count=0 bytes=0
  while IFS=$'\t' read -r path size digest; do
    local dest="$dir/$path" tmp="$dir/$path.part"
    if [ -f "$dest" ] && [ "$(size_of "$dest")" = "$size" ] && [ "$(sha256_of "$dest")" = "$digest" ]; then
      count=$((count + 1)); bytes=$((bytes + size))
      continue
    fi
    # A pointer file sits at dest until the real one is complete, so download beside it.
    if ! download "$base/$path" "$tmp" "$size"; then
      failed=$((failed + 1))
      continue
    fi
    if [ "$(sha256_of "$tmp")" = "$digest" ]; then
      mv -f "$tmp" "$dest"
      count=$((count + 1)); bytes=$((bytes + size))
    else
      echo "SkillCorner checksum mismatch, removing: $path" >&2
      rm -f "$tmp"
      failed=$((failed + 1))
    fi
  done < "$manifest"
  echo "skillcorner: commit $ref, $count LFS files verified, $bytes bytes, $failed failed"
  [ "$failed" = 0 ]
}

SOURCES=("$@")
if [ "${#SOURCES[@]}" = 0 ]; then SOURCES=(metrica idsse skillcorner); fi
status=0
for source in "${SOURCES[@]}"; do
  case "$source" in
    metrica) fetch_metrica || status=1 ;;
    idsse) fetch_idsse || status=1 ;;
    skillcorner) fetch_skillcorner || status=1 ;;
    *) echo "unknown source: $source" >&2; status=2 ;;
  esac
done
exit "$status"
