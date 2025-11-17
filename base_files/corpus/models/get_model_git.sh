#!/usr/bin/env bash
# get_model_git_hardened.sh — Clone a Hugging Face model via git + git-lfs on locked-down HPCs.
# Usage:
#   bash models/get_model_git_hardened.sh BAAI/bge-small-en-v1.5 models/bge-small-en-v1.5
#   bash models/get_model_git_hardened.sh BAAI/bge-base-en-v1.5  models/bge-base-en-v1.5
#
# Behavior:
#  1) Tries known CA bundle paths (& Python certifi) until git ls-remote succeeds.
#  2) If still failing, extracts the presented CA chain via openssl and builds a local CA bundle.
#  3) If STILL failing, offers a one-time insecure fallback (http.sslVerify=false) with a loud warning.
#
# Notes:
#  - Requires: git, git-lfs, openssl
#  - Respects proxy env (https_proxy/http_proxy)
#  - After success it prints: export EMBED_LOCAL_PATH=<dest>

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <model-id> <dest-dir>" >&2
  exit 1
fi

MODEL_ID="$1"
DEST_DIR="$2"

# ---- prerequisites ----
if ! command -v git >/dev/null 2>&1; then
  echo "[FATAL] git not found on PATH." >&2
  exit 1
fi
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "[FATAL] git-lfs not found on PATH. Install via 'conda install -c conda-forge git-lfs' or your modules." >&2
  exit 1
fi
if ! command -v openssl >/dev/null 2>&1; then
  echo "[FATAL] openssl not found on PATH." >&2
  exit 1
fi

echo "[INFO] Initializing git-lfs"
git lfs install --skip-repo

mkdir -p "$DEST_DIR"

# ---- Step 1: try common CA bundles ----
CANDIDATES=()

# env-specified
[ -n "${REQUESTS_CA_BUNDLE:-}" ] && CANDIDATES+=("${REQUESTS_CA_BUNDLE}")
[ -n "${SSL_CERT_FILE:-}" ]      && CANDIDATES+=("${SSL_CERT_FILE}")

# common distro paths
CANDIDATES+=(
  "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem"  # RHEL/CentOS/Rocky
  "/etc/ssl/certs/ca-certificates.crt"                 # Debian/Ubuntu
  "/etc/pki/tls/certs/ca-bundle.crt"                   # generic OpenSSL
)

# python certifi (best-effort)
CERTIFI_PATH="$(python - <<'PY'
try:
    import certifi; print(certifi.where())
except Exception:
    pass
PY
)"
[ -n "$CERTIFI_PATH" ] && CANDIDATES+=("$CERTIFI_PATH")

echo "[INFO] Probing CA bundles for git connectivity to huggingface.co..."
FOUND_CA=""
for p in "${CANDIDATES[@]}"; do
  if [ -f "$p" ]; then
    if git -c http.sslCAInfo="$p" ls-remote https://huggingface.co/ &>/dev/null; then
      FOUND_CA="$p"
      echo "[OK] Git trust established using: $p"
      break
    else
      echo "[WARN] Probe failed with: $p"
    fi
  fi
done

# ---- Step 2: build a local CA bundle by scraping the presented chain ----
if [ -z "$FOUND_CA" ]; then
  echo "[INFO] Attempting to capture the presented CA chain from huggingface.co ..."
  TMPDIR="$(mktemp -d)"
  pushd "$TMPDIR" >/dev/null
  # Grab the chain shown by your proxy / MITM
  # -servername SNI is important on some proxies
  if openssl s_client -showcerts -connect huggingface.co:443 -servername huggingface.co </dev/null 2>/dev/null \
      | awk '/BEGIN CERTIFICATE/{i++} {print > ("cert" i ".pem")} /END CERTIFICATE/{ } END{print i " certs"}' ; then
    # Build a CA bundle from any certs that look like CA:TRUE, or as a fallback concatenate all.
    CA_BUNDLE="corp_hf_chain.pem"
    > "$CA_BUNDLE"
    any_ca=false
    for c in cert*.pem; do
      if openssl x509 -in "$c" -noout -text 2>/dev/null | grep -q "CA:TRUE"; then
        cat "$c" >> "$CA_BUNDLE"
        any_ca=true
      fi
    done
    if [ "$any_ca" = false ]; then
      echo "[WARN] No explicit CA:TRUE found; concatenating all presented certs."
      cat cert*.pem > "$CA_BUNDLE"
    fi

    if git -c http.sslCAInfo="$CA_BUNDLE" ls-remote https://huggingface.co/ &>/dev/null; then
      FOUND_CA="$TMPDIR/$CA_BUNDLE"
      echo "[OK] Built working CA bundle at: $FOUND_CA"
    else
      echo "[WARN] Even the captured chain didn't satisfy git verification."
    fi
  else
    echo "[WARN] openssl couldn't retrieve the chain (network policy?)."
  fi
  popd >/dev/null
fi

# ---- Step 3: clone (or last-resort insecure) ----
if [ -n "$FOUND_CA" ]; then
  echo "[INFO] Cloning with CA bundle: $FOUND_CA"
  git -c http.sslCAInfo="$FOUND_CA" clone --depth=1 "https://huggingface.co/${MODEL_ID}" "$DEST_DIR"
else
  echo "[ERROR] Could not establish a trusted CA path automatically."
  echo "        As a LAST RESORT, you can proceed insecurely for this one clone."
  read -p "Proceed insecurely (http.sslVerify=false) [y/N]? " yn
  case "$yn" in
    [Yy]* )
      echo "[WARN] Proceeding without TLS verification for this clone."
      git -c http.sslVerify=false clone --depth=1 "https://huggingface.co/${MODEL_ID}" "$DEST_DIR"
      ;;
    * )
      echo "[ABORT] Not cloning without verification. Try the offline import route instead."
      exit 2
      ;;
  esac
fi

# Ensure LFS blobs download
echo "[INFO] Fetching LFS files..."
(
  cd "$DEST_DIR"
  if [ -n "${FOUND_CA:-}" ]; then
    GIT_LFS_SKIP_SMUDGE=0 git -c http.sslCAInfo="$FOUND_CA" lfs pull
  else
    GIT_LFS_SKIP_SMUDGE=0 git -c http.sslVerify=false lfs pull
  fi
)

echo ""
echo "[OK] Model ready at: $DEST_DIR"
echo "Export for your pipeline:"
echo "  export EMBED_LOCAL_PATH=$DEST_DIR"
