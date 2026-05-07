# oci-modelcar — Design Spec

**Date** : 2026-05-07
**Auteur** : codanael
**Status** : Approuvé pour implémentation

---

## 1. Contexte et objectif

### 1.1 Problème

Pousser un modèle HuggingFace dans un registry OCI pour un déploiement KServe avec OCI image volumes natifs (KEP-4639, GA dans OpenShift 4.21+) requiert aujourd'hui :

1. Un Dockerfile multi-stage avec `huggingface-cli download` puis `FROM scratch + COPY`
2. Une triple round-trip réseau : HuggingFace → cache local → registry
3. Un seul layer monolithique : aucun cross-repo blob mount, aucune réutilisation
4. Aucune reprise sur échec : si un shard de 5 GB plante à 4.5 GB, on recommence tout

### 1.2 Solution

Outil Python en streaming pur :
- Lit HuggingFace en streaming HTTP
- Pousse en streaming dans le registry OCI via Distribution API (POST → PATCH → PUT)
- Un layer tar non-compressé par fichier HF (`digest == diff_id`)
- Reprise sur échec à trois niveaux : intra-fichier HF (Range), intra-fichier OCI (resync session), inter-process (state file)
- Empreinte mémoire bornée à ~`workers × CHUNK × 2` (16 MiB par worker par défaut)
- Aucune persistance disque pour les bytes du modèle, aucun docker build

### 1.3 Non-objectifs (v1)

- Cross-repo blob mount actif (POST `?mount=`) — documenté en phase 3
- Compression gzip/zstd des layers (les safetensors compressent mal, gzip = burn CPU)
- Chiffrement de bout en bout des artefacts
- Support de registries non-OCI (Docker Hub v1, ECR avec quirks spécifiques au-delà du standard)
- UI graphique
- Gestion de cluster Kubernetes ou de ressources Tekton/Argo

---

## 2. Architecture et modules

Package Python `oci_modelcar`, distribué sur PyPI sous `oci-modelcar`. Python 3.14+ requis.

### 2.1 Structure du repo

```
oci-modelcar/
├── src/oci_modelcar/
│   ├── __init__.py
│   ├── __main__.py              # python -m oci_modelcar
│   ├── cli.py                   # argparse, dispatch sous-commandes
│   ├── config.py                # dataclass Config + parsing env/CLI + validation
│   ├── logging.py               # TextFormatter + AzureFormatter, file-scoped buffering
│   ├── http.py                  # session requests, auth, proxy, urllib3 Retry
│   ├── hf.py                    # listing fichiers + HfStream (Range-resume)
│   ├── oci.py                   # ChunkedBlobUpload + push_small_blob + HEAD
│   ├── tar_layer.py             # streaming tar wrapper
│   ├── manifest.py              # config + manifest builder, push tag(s)
│   ├── state.py                 # JsonStateStore (atomique, threading.Lock)
│   └── runner.py                # ThreadPoolExecutor, ordering, error policy
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── .github/workflows/
│   ├── ci.yml
│   ├── e2e.yml
│   └── release.yml
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
├── LICENSE                      # MIT
├── CHANGELOG.md
└── .gitignore                   # inclut shell.nix (dev local hors-repo)
```

### 2.2 Frontières des modules

- `hf.py` ne connaît pas OCI ; expose `HfStream` (file-like) et `list_files()`.
- `oci.py` ne connaît pas HF ; accepte un `IO[bytes]` writable et calcule SHA-256 incrémentalement.
- `tar_layer.py` est le pont : `tarfile.open(fileobj=upload, mode="w|")` branché sur les deux.
- `state.py` n'est invoqué que par `runner.py`.
- `logging.py` n'est utilisé que par `cli.py` et `runner.py`. Les modules I/O lèvent des exceptions, le runner les traduit.

### 2.3 Dépendances runtime

- `requests >= 2.32`
- `urllib3 >= 2.2`

Pas de `pydantic`, `click`, `rich`, `structlog`. Tout repose sur `dataclasses`, `argparse`, `logging`, `json`, `tomllib`, `concurrent.futures`.

### 2.4 Dépendances dev

- `pytest >= 8`, `pytest-cov`, `pytest-httpserver`
- `ruff >= 0.7`, `mypy >= 1.13`, `types-requests`
- `build`, `pre-commit`

---

## 3. Modèle de données : état persistant

### 3.1 Format

Fichier JSON unique. Écriture atomique via `tempfile.NamedTemporaryFile(dir=path.parent)` + `os.replace`. Lock `threading.Lock` côté process pour les écritures concurrentes.

**Chemin par défaut** : `${XDG_STATE_HOME:-$HOME/.local/state}/oci-modelcar/state.json`. Configurable via `--state-file PATH` ou `STATE_FILE` env. Permissions 0600 (refus de chargement si plus permissif).

### 3.2 Schéma

```json
{
  "version": 1,
  "jobs": {
    "<job_key>": {
      "source": {
        "hf_repo": "Qwen/Qwen3-30B-A3B",
        "hf_revision_input": "main",
        "hf_revision_resolved": "a3f47b09c8d2e6f1a89b7c4d3e8f2a1b5c6d7e8f"
      },
      "target": {
        "registry": "registry.example.com",
        "repo": "models/qwen3-30b",
        "tag": "a3f47b09c8d2",
        "also_tags": ["latest"]
      },
      "started_at": "2026-05-07T10:32:01Z",
      "updated_at": "2026-05-07T10:48:14Z",
      "completed_at": null,
      "manifest_digest": null,
      "files": {
        "model.safetensors": {
          "size": 5234567890,
          "digest": "sha256:abc...",
          "diff_id": "sha256:abc...",
          "pushed_at": "2026-05-07T10:35:22Z"
        }
      }
    }
  }
}
```

### 3.3 Job key

```
job_key = sha256(hf_repo + ":" + revision_resolved + "→" +
                 registry + "/" + target_repo + ":" + target_tag).hexdigest()[:16]
```

Indexer sur `revision_resolved` (la SHA résolue) garantit qu'un re-run avec `--hf-revision main` sur une branche qui a bougé crée un job_key distinct, donc une entrée indépendante. Pas d'invalidation destructive.

### 3.4 Sémantique de reprise

- Au démarrage : calcul `job_key`, chargement du state.
- Si l'entrée existe avec `manifest_digest != null` et `--force` non passé : exit 0 ("already completed").
- Si l'entrée existe sans `manifest_digest` : on parcourt `files{}`. Pour chaque fichier :
  - Si `pushed_at != null` ET `size` cohérent avec ce que retourne l'API tree HF maintenant : skip.
  - Sinon : retire l'entrée et re-pousse depuis zéro.
- On ne stocke PAS de session OCI partielle. Si un fichier était en cours, on le recommence à neuf (cohérent avec la décision file-level).

### 3.5 Idempotence garantie par construction

- `mtime=0`, `uid=gid=0`, `uname=gname=""` dans tous les TarInfo
- Tri alphabétique des fichiers HF avant traitement → ordre fixe
- `diff_ids[]` reconstruit dans l'ordre original (même avec workers > 1)
- Pas de champ `created` dans le config OCI

⇒ Mêmes inputs HF → mêmes tar bytes → mêmes layer digests → même config bytes → même config digest → même manifest bytes → même manifest digest.

---

## 4. Algorithme de streaming par fichier

### 4.1 Pipeline heureux

```
HF API tree → list[(path, size)] (filtré par allow_patterns, trié)
   │
   ▼
Pour chaque fichier non skippé :
   ┌────────────────────────────────────────────────────────────┐
   │ POST /v2/<repo>/blobs/uploads/   → 202 + Location           │
   │                                                             │
   │ tar = tarfile.open(fileobj=upload, mode="w|")               │
   │ tar.addfile(                                                │
   │   TarInfo(name=PREFIX+basename, size, mode=0o644,           │
   │           mtime=0, uid=gid=0, uname=gname=""),              │
   │   HfStream(path, size))                                     │
   │ tar.close()  # 1024 bytes de trailer                        │
   │                                                             │
   │ digest, layer_size = upload.close()  # PUT ?digest=         │
   │ HEAD /v2/<repo>/blobs/<digest>  → 200 + Docker-Content-     │
   │                                    Digest validation        │
   │ state.mark_pushed(file, digest, size)                       │
   └────────────────────────────────────────────────────────────┘
```

### 4.2 Reprise intra-fichier sur HF (Range request)

`HfStream` masque les ruptures de connexion derrière `read(n)`. Sur `ConnectionError` / `ChunkedEncodingError` / `ReadTimeout`, on rouvre une requête HF avec `Range: bytes=<bytes_buffered>-` et on continue. Le hasher SHA-256 et la session OCI ne savent rien de cette reprise.

```python
class HfStream:
    def __init__(self, path: str, expected_size: int):
        self.bytes_buffered = 0
        self.buf = b""
        self._open_stream(start=0)

    def _open_stream(self, start: int) -> None:
        headers = dict(HF_HDR)
        if start > 0:
            headers["Range"] = f"bytes={start}-"
        self.r = session.get(url, headers=headers, stream=True, timeout=600)
        self.r.raise_for_status()
        if start == 0:
            cl = int(self.r.headers.get("content-length", "0"))
            if cl and cl != self.expected_size:
                raise SizeMismatch(...)
        else:
            cr = self.r.headers.get("content-range", "")
            if not cr.startswith(f"bytes {start}-"):
                raise RuntimeError("server did not honor Range request")
        self.it = self.r.iter_content(chunk_size=CHUNK)

    def _next_chunk(self) -> bytes:
        for attempt in range(MAX_RETRIES_HF):
            try:
                return next(self.it)
            except StopIteration:
                raise
            except (ConnectionError, ChunkedEncodingError, ReadTimeout) as e:
                log.warning("HF read failed at offset %d: %s", self.bytes_buffered, e)
                sleep_backoff(attempt)
                with contextlib.suppress(Exception):
                    self.r.close()
                self._open_stream(start=self.bytes_buffered)
        raise RuntimeError("HF retries exhausted")
```

### 4.3 Reprise intra-fichier sur OCI (PATCH idempotent + resync via GET)

`StreamingBlobUpload._flush()` gère le cas où Artifactory/registry a reçu le chunk mais la réponse 202 n'est pas arrivée. Sur erreur transitoire, GET sur la session retourne `Range: 0-N` (inclusive) → on resynchronise `server_offset`.

```python
def _flush(self, chunk: bytes) -> None:
    start = self.server_offset
    end = start + len(chunk) - 1
    for attempt in range(MAX_RETRIES_OCI):
        try:
            r = session.patch(
                self.location, data=chunk,
                headers={
                    **REG_HDR,
                    "Content-Type": "application/octet-stream",
                    "Content-Range": f"{start}-{end}",  # OCI: pas de "bytes ", inclusive
                    "Content-Length": str(len(chunk)),
                },
                timeout=600,
            )
            if r.status_code == 202:
                self.location = r.headers.get("Location", self.location)
                self.server_offset = end + 1
                return
            if r.status_code == 416:
                self._resync()
                if self.server_offset >= end + 1:
                    return
                continue
            r.raise_for_status()
        except (ConnectionError, Timeout, ChunkedEncodingError) as e:
            log.warning("PATCH failed [%d-%d] attempt %d: %s", start, end, attempt, e)
            sleep_backoff(attempt)
            self._resync()
            if self.server_offset >= end + 1:
                return
    raise RuntimeError(f"PATCH retries exhausted at offset {start}")

def _resync(self) -> None:
    r = session.get(self.location, headers=REG_HDR, timeout=30)
    if r.status_code != 204:
        r.raise_for_status()
    rng = r.headers.get("Range", "")
    self.server_offset = int(rng.split("-")[1]) + 1 if rng else 0
```

Le hash SHA-256 est mis à jour dans `write()` au moment du buffering — il représente "octets logiques du blob", indépendant des retries d'envoi (les bytes resoumis sont bit-exacts).

### 4.4 Reprise inter-process (file-level)

Décrite en §3.4. Au démarrage, on saute les fichiers déjà présents dans `state.jobs[k].files{}` avec `pushed_at != null` et `size` cohérent.

### 4.5 Politique d'échec d'un fichier

- `--fail-fast` (défaut) : on cancelle les futurs en attente, on attend les in-flight, on log un summary, exit 2.
- `--continue-on-error` : on log l'échec, on continue les autres fichiers, on **n'écrit pas le manifest** à la fin, exit 3 avec liste des fichiers manquants.

### 4.6 Politique sur signal (SIGINT/SIGTERM)

- Cancellation des futurs en attente.
- Attente bornée des in-flight (timeout configurable, défaut 30s).
- `state.save()` avant l'exit.
- Exit 130.

### 4.7 Constantes

| Param | Défaut | Var env | CLI |
|---|---|---|---|
| Chunk size PATCH | 8 MiB | `CHUNK_MIB` | `--chunk-mib` |
| Workers | 1 (cap 8) | `WORKERS` | `--workers` |
| Max retries HF | 10 | `HF_MAX_RETRIES` | `--hf-max-retries` |
| Max retries OCI | 10 | `OCI_MAX_RETRIES` | `--oci-max-retries` |
| Backoff cap | 60 s | — | — |
| Heartbeat interval | 30 s | — | — |
| Graceful shutdown timeout | 30 s | — | — |

---

## 5. Résolution de la revision HF & dérivation du tag

### 5.1 Résolution de la revision

| Input utilisateur (`--hf-revision`) | Comportement |
|---|---|
| Non spécifié OU `"main"` | `GET {HF_ENDPOINT}/api/models/{HF_REPO}` → champ `sha` (commit du default branch) |
| SHA complète (40 hex) | Vérifiée via `GET /api/models/{repo}/revision/{sha}` |
| SHA partielle (≥ 7 hex), branche, ou tag HF | Canonicalisée via `GET /api/models/{repo}/revision/{revision}` → SHA complète. En cas de 404 ou d'erreur côté proxy : log warning, utilisation telle quelle (idempotence non garantie) |

### 5.2 Dérivation du tag d'image

| Cas | Tag |
|---|---|
| `--target-tag` explicite | Utilisé tel quel (tag user-driven, ex `v1`, `prod`) |
| `--target-tag` absent ET `revision_resolved` matche `^[0-9a-f]{40}$` | **`<sha[:12]>`** (ex `a3f47b09c8d2`) |
| `--target-tag` absent ET `revision_resolved` est un nom (branche/tag) | **`<name>`** sanitisé `[^a-zA-Z0-9._-]` → `_`, tronqué à 128 chars |

### 5.3 Tags additionnels

`--also-tag latest,prod` (CSV) → push du même manifest sous des aliases supplémentaires. Implémentation : N PUT du manifest sous chaque tag, sequentiel. Validation HEAD/GET sur tous.

---

## 6. CLI, configuration et credentials

### 6.1 Surface CLI

Sous-commandes :

```
oci-modelcar push      # principal
oci-modelcar status    # liste les jobs depuis state.json
oci-modelcar validate  # refait juste la validation HEAD/GET sur un tag existant
oci-modelcar --version
oci-modelcar --help
```

### 6.2 Arguments de `push`

```
oci-modelcar push \
  --hf-repo Qwen/Qwen3-30B-A3B \
  [--hf-revision main]                  # défaut main → résolu en SHA
  [--hf-endpoint https://huggingface.co] # défaut huggingface.co
  --registry registry.example.com \
  --target-repo models/qwen3-30b \
  [--target-tag a3f47b09c8d2]           # défaut: SHA[:12]
  [--also-tag latest,prod]
  [--allow-patterns ".safetensors .json .txt .md .model"]
  [--layer-prefix models/]
  [--chunk-mib 8]
  [--workers 1]                          # cap dur 8
  [--state-file PATH]
  [--hf-max-retries 10]
  [--oci-max-retries 10]
  [--fail-fast | --continue-on-error]
  [--force]                              # ignore state, repousse tout
  [--log-style text|azure]               # absence du flag = auto (TF_BUILD → azure, sinon text)
  [--verbose | --quiet]
  [--dry-run]                            # liste, ne pousse rien
```

### 6.3 Variables d'environnement

| Env var | CLI equivalent | Default |
|---|---|---|
| `HF_REPO` | `--hf-repo` | requis |
| `HF_REVISION` | `--hf-revision` | `main` |
| `HF_ENDPOINT` | `--hf-endpoint` | `https://huggingface.co` |
| `HF_TOKEN` | — | (lu depuis env ou `~/.cache/huggingface/token`) |
| `REGISTRY` | `--registry` | requis |
| `TARGET_REPO` | `--target-repo` | requis |
| `TARGET_TAG` | `--target-tag` | dérivé |
| `ALSO_TAGS` | `--also-tag` | — |
| `OCI_USERNAME` / `OCI_PASSWORD` | — | (sinon `~/.docker/config.json`) |
| `ALLOW_PATTERNS` | `--allow-patterns` | `.safetensors .json .txt .md .model` |
| `LAYER_PATH_PREFIX` | `--layer-prefix` | `models/` |
| `CHUNK_MIB` | `--chunk-mib` | `8` |
| `WORKERS` | `--workers` | `1` |
| `STATE_FILE` | `--state-file` | `${XDG_STATE_HOME:-$HOME/.local/state}/oci-modelcar/state.json` |
| `HF_MAX_RETRIES` | `--hf-max-retries` | `10` |
| `OCI_MAX_RETRIES` | `--oci-max-retries` | `10` |
| `FAIL_FAST` | `--fail-fast`/`--continue-on-error` | `1` |
| `FORCE` | `--force` | `0` |
| `LOG_STYLE` | `--log-style` | auto |
| `LOG_VERBOSE` / `LOG_QUIET` | `--verbose`/`--quiet` | — |
| `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` | — | honorées par requests |
| `TF_BUILD` | — | déclenche `log-style=azure` en auto |

### 6.4 Authentification

**HuggingFace (alignement avec `huggingface-cli`)** :
1. `HF_TOKEN` env var → `Authorization: Bearer <token>`
2. Fallback : `~/.cache/huggingface/token` (créé par `huggingface-cli login`)
3. Sinon : pas d'auth (modèles publics)

**Registry OCI** :
1. `OCI_USERNAME` + `OCI_PASSWORD` env vars → basic auth
2. `~/.docker/config.json` (lookup par hostname, format `{"auths": {"host": {"auth": "<base64>"}}}`)
3. `$XDG_RUNTIME_DIR/containers/auth.json` (Podman/containers)
4. Helper `credsStore` : non supporté en v1
5. Sinon : pas d'auth (registry public)

### 6.5 Validation au démarrage

`Config.validate()` lève `ConfigError` (exit 64) si :
- Champs requis manquants (`hf_repo`, `registry`, `target_repo`)
- `workers > 8` ou `workers < 1`
- `chunk_mib < 1` ou `chunk_mib > 1024`
- `--fail-fast` et `--continue-on-error` simultanés
- `state_file` répertoire non-créable
- `target_tag` contient caractères hors `[a-zA-Z0-9._-]` (limite 128 chars)
- Permissions du state_file > 0600

### 6.6 Codes de retour

| Code | Sémantique |
|---|---|
| 0 | Succès complet, manifest pushed et validé |
| 1 | Erreur générique inattendue |
| 2 | Au moins un fichier a définitivement échoué (`--fail-fast`) |
| 3 | `--continue-on-error` : job incomplet, manifest non écrit |
| 64 | Erreur de configuration / validation des arguments |
| 65 | Erreur d'authentification (401/403 sur registry ou HF) |
| 130 | Interrompu par signal — state sauvegardé |

### 6.7 Output stdout en fin de run (deux modes)

```
MANIFEST=sha256:abc1234...
IMAGE=registry.example.com/models/qwen3-30b:a3f47b09c8d2
```

Lignes `KEY=VALUE` parseable, présentes dans les deux modes (text et azure). Le mode azure les double avec `##vso[task.setvariable variable=manifestDigest;isOutput=true]…` pour les pipelines Azure.

---

## 7. Logging

### 7.1 Deux formatters

- **`TextFormatter`** (défaut hors Azure DevOps) : texte humain, sections délimitées par séparateurs Unicode, couleurs ANSI si TTY ET `NO_COLOR` non défini.
- **`AzureFormatter`** : balises Azure Pipelines (`##[section]`, `##[group]`, `##[endgroup]`, `##[warning]`, `##[error]`, `##[debug]`, `##vso[task.setprogress]`, `##vso[task.setvariable]`).

Auto-détection : `TF_BUILD=True` dans l'env → `azure`. Sinon `text`. Override par `--log-style` ou `LOG_STYLE`.

### 7.2 Implémentation

Stdlib `logging` avec `record.extra` enrichi : chaque log porte un type sémantique (`section_start`, `group_start`, `group_end`, `progress`, `warning`, `error`, `debug`, `info`, `output_var`). Les deux Formatters lisent ce type et produisent le rendu adéquat.

### 7.3 Mode parallèle (workers > 1)

Bufferisation par fichier dans une `FileScopedLogger(io.StringIO)`. Flush atomique (sous `threading.Lock` sur stdout) en fin de fichier (succès ou échec).

Heartbeats hors-groupe : émis sur stdout principal toutes les 30s, format `[HB] <path>: <bytes>/<total> (<rate> MB/s, idle <s>s)`. Pas de balise → lisible dans les deux styles.

### 7.4 Niveaux

- Défaut : `info` + warnings + errors + sections + groups + progress + heartbeats
- `--verbose` : ajoute debug (offsets, locations, header counts)
- `--quiet` : seulement groups + errors + summary final

### 7.5 stdout vs stderr

- stdout : tous les events (info, warnings, debug, progress, output vars)
- stderr : seulement les erreurs fatales causant l'exit non-zéro

---

## 8. Validation post-push

### 8.1 Séquence

Exécutée juste avant l'écriture de `manifest_digest` et `completed_at` dans le state.

```python
def validate_push(repo, tag, layers, config_descriptor, manifest_bytes, also_tags):
    # 1. HEAD chaque layer + Docker-Content-Digest match
    for layer in layers:
        r = session.head(reg(repo, "blobs", layer["digest"]),
                         headers=REG_HDR, timeout=30)
        if r.status_code != 200:
            raise ValidationError(f"missing layer blob {layer['digest']}")
        got = r.headers.get("Docker-Content-Digest", "")
        if got != layer["digest"]:
            raise ValidationError(
                f"digest mismatch on HEAD: expected {layer['digest']} got {got}")
        cl = r.headers.get("Content-Length")
        if cl and int(cl) != layer["size"]:
            raise ValidationError(
                f"size mismatch: manifest={layer['size']} registry={cl}")

    # 2. HEAD config
    r = session.head(reg(repo, "blobs", config_descriptor["digest"]),
                     headers=REG_HDR, timeout=30)
    if r.status_code != 200:
        raise ValidationError("missing config blob")

    # 3. GET manifest by tag, vérifier Docker-Content-Digest
    expected = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    for t in [tag, *also_tags]:
        r = session.get(reg(repo, "manifests", t),
                        headers={**REG_HDR, "Accept": ML_MAN}, timeout=30)
        r.raise_for_status()
        got = r.headers.get("Docker-Content-Digest", "")
        if got != expected:
            raise ValidationError(
                f"manifest digest mismatch on tag {t}: expected {expected} got {got}")
```

### 8.2 Retries

HEAD/GET peuvent échouer transitoirement aussi → backoff identique, max 5 tentatives par requête.

### 8.3 Sous-commande `validate`

`oci-modelcar validate --registry … --target-repo … --target-tag …` réutilise la fonction sur un job déjà pushé. N'a besoin ni du state file, ni de l'accès HF.

---

## 9. Compliance OCI Distribution v1.1 + Image Spec v1.1

### 9.1 Codes de retour stricts

| Endpoint | Code attendu |
|---|---|
| `POST /v2/<repo>/blobs/uploads/` | 202 Accepted + `Location` |
| `PATCH <location>` | 202 Accepted + `Location` + `Range: 0-<n>` |
| PATCH out-of-order | 416 Requested Range Not Satisfiable |
| `GET <location>` (resume status) | 204 No Content + `Range: 0-<n>` (header optionnel si 0 octet) |
| `PUT <location>?digest=<d>` | 201 Created |
| `HEAD /v2/<repo>/blobs/<digest>` | 200 OK + `Docker-Content-Digest` + `Content-Length` |
| `GET /v2/<repo>/manifests/<tag>` | 200 OK + `Docker-Content-Digest` |

Code à durcir vs `stream_modelcar.py` initial :
- `PUT` close blob : check explicite `status_code == 201`
- `GET` upload session : check explicite `status_code == 204`
- `HEAD` blob validation : ajouter check `Docker-Content-Digest == expected`

### 9.2 Format `Content-Range` sur PATCH

Spec OCI : `^[0-9]+-[0-9]+$`, **inclusive aux deux bornes**, sans préfixe `bytes ` (ce n'est PAS RFC 7233). Notre code envoie `f"{start}-{end}"` — conforme. Test unitaire `test_content_range_format()` pinne ce format.

### 9.3 Sémantique `Range` sur GET upload session

`Range: 0-<position-du-dernier-octet>` inclusive :
- Header absent → 0 octets reçus → `server_offset = 0`
- `Range: 0-0` → 1 octet → `server_offset = 1`
- `Range: 0-1023` → 1024 octets → `server_offset = 1024`

### 9.4 `diff_id == digest` sur layers tar non-compressés

Pour `application/vnd.oci.image.layer.v1.tar` (non-compressé) :
```
layer.digest == sha256(tar_bytes) == diff_id
```
Garantie native par la spec image. Si on ajoute `+gzip` plus tard (hors-scope v1), il faudra calculer deux hashs en parallèle.

### 9.5 Config OCI minimal

Champs requis : `architecture`, `os`, `rootfs.type` (= `"layers"`), `rootfs.diff_ids[]`. Optionnels : `config`, `created`, `history`.

Notre config :
```json
{
  "architecture": "amd64",
  "os": "linux",
  "rootfs": {"type": "layers", "diff_ids": [...]},
  "config": {}
}
```

Pas de champ `created` → préserve l'idempotence cross-run. Documenté dans le code.

### 9.6 Ordering des layers

Spec : layers en stack order, base layer à l'index 0. Pour OCI image volume (KEP-4639), pas de filesystem overlay réel. Mais l'ordre **doit être déterministe** pour l'idempotence : tri alphabétique sur `hf_path`.

### 9.7 Cross-repo blob mount : hors-scope v1

`POST … ?mount=<digest>&from=<source_repo>` n'est pas implémenté en v1. La conception streaming ne connaît pas le digest avant la fin du tar. Voie possible (phase 3) :
- Cache global `(hf_repo, hf_path, hf_blob_oid) → layer_digest`
- Avant push, lookup dans le cache, HEAD dans target repo, mount si cross-repo
- Pas implémenté pour limiter la complexité v1

---

## 10. Tests

### 10.1 Stack

- `pytest >= 8` + `pytest-cov` + `pytest-httpserver` (mock HTTP léger, pur Python)
- Pas de `vcr`, pas de `requests-mock`
- Markers : `e2e` pour les tests gated

### 10.2 Hiérarchie

```
tests/
├── conftest.py
├── unit/
│   ├── test_config.py
│   ├── test_state.py
│   ├── test_logging.py             # text + azure formatters, file-scoped buffering
│   ├── test_tar_layer.py           # tar bytes reproductibles, mtime=0
│   ├── test_manifest.py            # manifest bytes reproductibles
│   ├── test_hf_stream.py           # Range resume sur mock httpserver
│   └── test_oci_upload.py          # PATCH/GET/PUT semantics
├── integration/
│   ├── test_runner_sequential.py
│   ├── test_runner_parallel.py     # workers=2, ordering déterministe
│   ├── test_resume_from_state.py
│   ├── test_failure_modes.py       # fail-fast, continue-on-error, SIGINT
│   └── test_cli.py                 # subprocess oci-modelcar, exit codes
├── e2e/
│   ├── conftest.py                 # docker run registry:2, teardown
│   └── test_real_huggingface.py    # vrai HF tiny-llama, vrai registry:2
└── fixtures/
    ├── fake_safetensors.py
    └── fake_hf_tree.py
```

### 10.3 Tests obligatoires (issus de l'audit OCI)

- `test_patch_content_range_format()` : format `"N-M"` sans préfixe
- `test_resync_no_range_header()` : 204 sans header → offset 0
- `test_resync_with_range_header_0_0()` : `Range: 0-0` → offset 1
- `test_resync_with_range_header_0_1023()` : `Range: 0-1023` → offset 1024
- `test_416_triggers_resync_and_skip_if_already_accepted()`
- `test_diff_id_equals_digest()` : tar fixe → `sha256(tar) == diff_id == layer.digest`
- `test_head_validation_checks_docker_content_digest()`
- `test_put_close_expects_201()`

### 10.4 Tests d'idempotence

```python
def test_manifest_digest_deterministic_across_workers():
    digests = []
    for workers in [1, 2, 4]:
        run_pipeline(fake_files, workers=workers)
        digests.append(load_pushed_manifest_digest())
    assert len(set(digests)) == 1
```

### 10.5 Tests de reprise sur crash simulé

Le mock httpserver tue la connexion sur le N-ième PATCH du M-ième fichier. On lance le pipeline → exit non-zéro → state sauvegardé → on relance → succès, fichiers déjà poussés sont skipés.

### 10.6 Tests E2E avec vrai modèle HuggingFace

**Modèle de test** : `hf-internal-testing/tiny-random-LlamaForCausalLM`
- ~10 MB total, 6 fichiers (incluant safetensors)
- Maintenu par HuggingFace pour CI
- SHA pinnée dans `tests/e2e/conftest.py` ; CI échoue avec message clair si SHA disparaît

```python
HF_TEST_REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HF_TEST_REVISION = "<SHA pinnée à ré-évaluer au moment de l'écriture>"
```

**Tests E2E** :
- `test_push_real_tiny_llama_to_local_registry` : push complet, `skopeo inspect` valide mediaType + layer count
- `test_resume_after_killed_real_push` : SIGTERM mid-stream, relance, vérifier idempotence du manifest digest
- `test_revision_resolution_main_to_sha` : `--hf-revision main --dry-run`, vérifier la SHA loggée
- `test_pull_and_verify_bytes` : `skopeo copy` puis untar et compare bytes vs HF original

**Override pour intranet** : `OCI_MODELCAR_E2E_HF_ENDPOINT`, `OCI_MODELCAR_E2E_REGISTRY`, `OCI_MODELCAR_E2E_HF_TOKEN` permettent de rejouer les E2E contre une infra interne.

**Pré-requis runtime** : Docker (sauf override), `skopeo`, connectivité huggingface.co (sauf override).

### 10.7 Coverage cible

- Unit + integration : ≥ 90% sur `oci.py`, `hf.py`, `state.py`, `runner.py`
- Reste : ≥ 75%

---

## 11. Quality gates

### 11.1 Pre-commit

`.pre-commit-config.yaml` commité dans le repo. Tous les hooks utilisent `language: system` pour respecter le PATH du shell (nix-installed binaries en dev, pip-installed en CI).

```yaml
repos:
  - repo: local
    hooks:
      - id: ruff-check
        name: ruff check
        entry: ruff check --fix
        language: system
        types: [python]
      - id: ruff-format
        name: ruff format
        entry: ruff format
        language: system
        types: [python]
      - id: mypy
        name: mypy --strict
        entry: mypy --strict
        language: system
        types: [python]
        pass_filenames: false
        args: [src/]
      - id: pytest-fast
        name: pytest (not e2e)
        entry: pytest -m "not e2e" -q
        language: system
        pass_filenames: false
        stages: [pre-commit]
```

### 11.2 GitHub Actions

`.github/workflows/ci.yml` (push + PR vers `main`) :
- Job `lint` : `ruff check`, `ruff format --check`, `mypy --strict`
- Job `test` : `pytest -m "not e2e"` sur Python 3.14 (Ubuntu)

`.github/workflows/e2e.yml` (manuel + nightly cron) :
- Service `registry:2`, install `skopeo`
- `pytest -m e2e`

`.github/workflows/release.yml` (sur tag `v*`) :
- `python -m build` (wheel + sdist)
- `pypa/gh-action-pypi-publish` avec OIDC (Trusted Publishing PyPI)
- Création GitHub Release avec changelog

### 11.3 Branche protégée

`main` requiert : CI vert + 1 review (auto-merge possible pour le owner). Pas de push direct sur `main`.

---

## 12. Packaging et distribution

### 12.1 `pyproject.toml`

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "oci-modelcar"
version = "0.1.0"
description = "Stream HuggingFace models directly into OCI registries as multi-layer images"
readme = "README.md"
requires-python = ">=3.14"
license = "MIT"
license-files = ["LICENSE"]
authors = [{name = "codanael"}]
keywords = ["huggingface", "oci", "kserve", "modelcar", "registry"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: System Administrators",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.14",
    "Topic :: System :: Archiving :: Packaging",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "requests>=2.32",
    "urllib3>=2.2",
]

[project.urls]
Homepage = "https://github.com/codanael/oci-modelcar"
Issues = "https://github.com/codanael/oci-modelcar/issues"
Source = "https://github.com/codanael/oci-modelcar"

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "pytest-httpserver>=1.0",
    "ruff>=0.7",
    "mypy>=1.13",
    "types-requests",
    "build>=1.2",
    "pre-commit>=4",
]
e2e = ["pytest>=8"]

[project.scripts]
oci-modelcar = "oci_modelcar.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/oci_modelcar"]

[tool.ruff]
target-version = "py314"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM", "RUF"]

[tool.mypy]
python_version = "3.14"
strict = true
warn_return_any = true

[tool.pytest.ini_options]
addopts = "-v --strict-markers"
markers = ["e2e: end-to-end tests requiring Docker and network"]
```

### 12.2 Distribution sur PyPI

- Trusted Publishing (OIDC) : pas de token PyPI dans les secrets GitHub
- Configuration côté PyPI : Trusted Publisher lié au repo `codanael/oci-modelcar` + workflow `release.yml`
- `pypa/gh-action-pypi-publish` reçoit un OIDC token GitHub, l'échange contre un token PyPI éphémère
- Release par tag git `v0.1.0` → workflow auto

### 12.3 Image Docker (optionnelle, pas en v1)

Pas de Dockerfile commité dans le repo en v1. Si besoin plus tard, ajout simple :
```dockerfile
FROM python:3.14-slim
RUN pip install --no-cache-dir oci-modelcar
ENTRYPOINT ["oci-modelcar"]
```

### 12.4 Versioning

- SemVer 2.0
- Source de vérité : `pyproject.toml` `version`
- `oci_modelcar.__version__ = importlib.metadata.version("oci-modelcar")`
- `--version` affiche version

---

## 13. Environnement de développement (NixOS)

**Hors repo**. Le `shell.nix` est créé localement par chaque dev, listé dans `.gitignore` au cas où il finirait en root du projet.

```nix
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = with pkgs; [
    python314
    python314Packages.pip
    python314Packages.virtualenv
    ruff                          # binaire Rust patché par nixpkgs
    mypy                          # mypyc-compiled, idem
    pre-commit
    skopeo                        # tests E2E
    gh
    git
  ];
  shellHook = ''
    if [ ! -d .venv ]; then
      python -m venv .venv
      source .venv/bin/activate
      pip install -e '.[dev,e2e]'
      pre-commit install
    else
      source .venv/bin/activate
    fi
  '';
}
```

`ruff` et `mypy` sont fournis par nixpkgs (PATH du shell), pas par pip — évite les problèmes de loader sur NixOS. Hors NixOS (CI), pip suffit.

Docker est déjà installé sur la machine dev.

---

## 14. Hors-scope v1 (extensions futures)

- Cross-repo blob mount actif via cache `(hf_oid → layer_digest)` (phase 3)
- Compression `+gzip` ou `+zstd` des layers (avec calcul de double hash)
- Support `credsStore` (Docker credential helpers)
- Auth proxy NTLM/Kerberos
- Image Docker publiée
- Support GitHub Actions logging commands (`::group::` etc.) — flag `--log-style=github`
- Métriques Prometheus exportées en sidecar
- UI web pour status

---

## 15. Glossaire

- **OCI Distribution** : spec HTTP du registry (POST/PATCH/PUT pour blobs, PUT pour manifest)
- **OCI Image** : spec du format manifest + config + layers
- **KEP-4639** : Kubernetes Enhancement Proposal pour OCI image volumes natifs (GA OpenShift 4.21+)
- **ModelCar** : pattern KServe pour servir un modèle depuis une image OCI
- **diff_id** : hash sha256 du tar non-compressé d'un layer (= layer.digest pour layers `+tar`)
- **Trusted Publishing** : mécanisme PyPI utilisant OIDC GitHub Actions, sans token long-lived
- **Job key** : sha256 court identifiant un job (source HF + target OCI)

---

## 16. Décisions tracées

| Décision | Rationale |
|---|---|
| File-level resume cross-process | hashlib.sha256 non-sérialisable ; sessions OCI expirent ; complexité chunk-level cross-process > gain attendu |
| JSON state file | ~100 entrées par job, debuggable avec jq, atomicité simple via os.replace |
| `requests` + stdlib seulement | minimiser la surface, faciliter audit/sécurité, Python 3.14 stdlib suffisante |
| Layer tar non-compressé | safetensors compressent mal, gzip = burn CPU, `digest == diff_id` simplifie l'idempotence |
| Tag = SHA[:12] par défaut | équilibre lisibilité/collision, convention git short hash |
| Workers défaut 1 | simple par défaut ; speedup réel typique 1.5-2x sur N=4 limité par bandwidth |
| Auto-détection log style | `TF_BUILD` natif Azure DevOps ; sinon humain par défaut |
| Auth registry via `~/.docker/config.json` | standard de facto, pas de réinvention |
| HF auth via `HF_TOKEN` + `~/.cache/huggingface/token` | aligné avec `huggingface-cli` |
| Pas de champ `created` dans config OCI | préserve l'idempotence cross-run |
| MIT license | maximalement permissive, courante sur PyPI pour outils CLI |
| Trusted Publishing PyPI | pas de token long-lived dans GitHub Secrets |
| `shell.nix` hors repo | env dev personnel, pas la cible prod, pas tous les contributeurs sur NixOS |
