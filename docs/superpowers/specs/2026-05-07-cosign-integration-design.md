# Cosign integration — Design Spec

**Date** : 2026-05-07
**Auteur** : codanael
**Status** : Approuvé pour implémentation
**Cible** : v0.2.0

---

## 1. Contexte et objectif

### 1.1 Problème

`oci-modelcar` v0.1.0 publie deux artefacts dont l'authenticité ne peut pas
être vérifiée par un consommateur en aval :

1. **Le package PyPI `oci-modelcar`** est livré sans attestation. Un attaquant
   qui compromet la chaîne de build pourrait pousser un wheel malveillant ;
   `pip install` ne fournit aucun moyen de détecter une substitution.
2. **Les images modelcar OCI** poussées par l'outil ne portent aucune signature.
   Un opérateur KServe qui consomme `<registry>/<repo>@sha256:<digest>` n'a
   pas de façon native de vérifier que cette image a été produite par un
   pipeline de confiance.

L'écosystème Sigstore (Fulcio + Rekor + cosign) a standardisé la signature
keyless OIDC pour les deux cas, et PyPI supporte nativement PEP 740 (digital
attestations). Les coûts d'intégration sont aujourd'hui faibles.

### 1.2 Objectif

Intégrer la signature à deux niveaux du pipeline `oci-modelcar`, sans alourdir
la surface du code ni ajouter de dépendance runtime :

1. **PyPI** : générer et publier des attestations PEP 740 lors du tag de
   release, automatiquement, en mode keyless OIDC GitHub.
2. **Modelcar OCI** : faciliter la signature post-push par l'utilisateur via
   `cosign`, en exposant la référence canonique par digest dans la sortie de
   `oci-modelcar push`. Documenter les modes keyless (CI) et clé statique
   (offline / corporate). Aucune intégration intra-CLI : la signature reste
   un step shell séparé.

### 1.3 Non-objectifs

- Sous-commande `oci-modelcar sign` ou flag `--sign`. La signature reste hors
  du périmètre du CLI : `cosign` est l'outil canonique et l'utilisateur le
  pilote directement.
- Workflow GitHub Actions réutilisable distribué dans le repo
  (`examples/sign-modelcar.yml`). Le README montre le snippet à recopier.
- `cosign attest` (attestations SLSA provenance, SBOM). Repoussé à v0.3+.
- Bundles `.sigstore` standalone attachés à la GitHub Release. PEP 740 couvre
  100% du flux `pip install`.
- Vérification automatique côté `oci-modelcar` (`pull` ou `verify` qui
  appellerait cosign). Hors scope.
- Support de clés gérées en KMS, HSM, ou autres backends cosign avancés.
  Documentation montre uniquement keyless OIDC et clé locale (cosign
  `generate-key-pair`).

---

## 2. Architecture

Le design est *minimal-by-design*. Trois zones de modification, aucun nouveau
module, aucune nouvelle dépendance Python.

### 2.1 Surface modifiée

| Fichier | Nature du changement | Lignes (estim.) |
|---|---|---|
| `.github/workflows/release.yml` | Ajout `attestations: true` à `pypa/gh-action-pypi-publish` | +1 |
| `src/oci_modelcar/runner.py` | Calcul + émission `image_ref_digest` | +5 |
| `src/oci_modelcar/state.py` | Étendre `mark_completed(... image_ref_digest=...)`, persister | +6 |
| `src/oci_modelcar/cli.py` | `_run_status` affiche `image_ref_digest` (graceful absence) | +3 |
| `tests/integration/test_runner_single.py` | Assert sortie `IMAGEREFDIGEST=...@sha256:` | +5 |
| `tests/unit/test_state.py` | Round-trip `image_ref_digest` | +10 |
| `tests/integration/test_cli.py` | Status affiche digest, graceful sur absence | +15 |
| `README.md` | Section "Signing & verification" | +40 |
| `CHANGELOG.md` | Entrée `[Unreleased]` | +5 |

Aucune modification de `oci.py`, `manifest.py`, `tar_layer.py`, `hf.py`,
`http.py`, `tags.py`, `logging.py`, `config.py`, `__init__.py`, `__main__.py`,
ni des E2E. Les invariants OCI/wire-format ne bougent pas.

### 2.2 Pas de dépendance ajoutée

`cosign` est invoqué par l'utilisateur via subprocess shell, pas par le code
Python. `runtime deps` reste `requests + urllib3`. `dev deps` ne change pas.

PEP 740 est généré entièrement par `pypa/gh-action-pypi-publish@release/v1`,
qui embarque déjà `sigstore-python`. Aucune modification de `pyproject.toml`.

---

## 3. Niveau 1 — PyPI artifact (PEP 740)

### 3.1 Mécanisme

PEP 740 définit un format d'*attestation digitale* attachée à un artefact PyPI :
un bundle Sigstore (cert Fulcio + signature + entrée Rekor) calculé sur le
hash SHA-256 du fichier dist (`*.whl`, `*.tar.gz`). PyPI expose ces
attestations via `GET /integrity/<project>/<filename>/provenance`.

Côté publication, `pypa/gh-action-pypi-publish@release/v1` accepte le flag
`attestations: true`. Quand activé, l'action :

1. Récupère un OIDC token GitHub (`id-token: write` permission, déjà présente).
2. Demande un cert éphémère à Fulcio en échange du token.
3. Signe chaque dist file avec la clé éphémère, log dans Rekor.
4. Joint le bundle Sigstore à l'upload PyPI.

L'identité de signature est dérivée de l'OIDC token : pour une release tag
`vX.Y.Z`, l'identité est
`https://github.com/codanael/oci-modelcar/.github/workflows/release.yml@refs/tags/vX.Y.Z`,
émise par `https://token.actions.githubusercontent.com`.

### 3.2 Configuration Trusted Publisher PyPI

Pré-requis : sur pypi.org → Project Settings → Publishing, le Trusted Publisher
existant doit être conservé. Aucun changement de config nécessaire — le flag
`attestations: true` ne demande pas d'inscription supplémentaire côté PyPI.

### 3.3 Vérification consommateur

Trois voies équivalentes :

```bash
# pip-attestation (recommandé pour automation CI)
python -m pip install pypi-attestations
python -m pypi_attestations verify pypi \
  --repository codanael/oci-modelcar \
  oci_modelcar-0.2.0-py3-none-any.whl

# pip-audit (intégration avec audit existant)
python -m pip install pip-audit
python -m pip_audit --require-hashes ...

# manuel via cosign verify-blob (pour curiosité forensique)
cosign verify-blob \
  --bundle oci_modelcar-0.2.0-py3-none-any.whl.sigstore \
  --certificate-identity-regexp '^https://github\.com/codanael/oci-modelcar/' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  oci_modelcar-0.2.0-py3-none-any.whl
```

### 3.4 Changement code

```yaml
# .github/workflows/release.yml — step publish-pypi
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          attestations: true   # nouveau
        # Trusted Publisher: configure on pypi.org under
        # Project Settings -> Publishing, link this repo + workflow.
```

C'est l'unique modification CI. Le workflow garde sa structure
`build → publish-pypi + release` inchangée.

---

## 4. Niveau 2 — Modelcar OCI

### 4.1 Référence canonique par digest

Aujourd'hui, `runner.py` calcule
`image_ref = f"{cfg.registry}/{cfg.target_repo}:{target_tag}"` et l'expose
via `IMAGEREF=…:tag`. Pour cosign, la **référence stable** est par digest :
`<registry>/<repo>@sha256:<manifest-digest>`. Un tag est mutable, un digest
ne l'est pas — signer par tag créerait des trous de sécurité.

L'outil va exposer une variable supplémentaire :

```
IMAGEREFDIGEST=registry.example.com/models/qwen3-30b@sha256:<manifest-digest>
```

en plus de `IMAGEREF` (conservé pour rétrocompatibilité avec tout consumer
qui parse déjà la sortie).

### 4.2 Persistance dans le state file

Le state JSON gagne un champ `image_ref_digest` au niveau du job. Cela permet :

- À `oci-modelcar status` d'afficher la ref canonique sans avoir à la
  reconstruire (même en cas de re-run idempotent).
- À l'utilisateur de la récupérer programmatiquement :
  `jq -r '.jobs[].image_ref_digest' ~/.local/state/oci-modelcar/state.json`

Schéma du state file (extrait) :

```json
{
  "version": 1,
  "jobs": {
    "abc123def456": {
      "source": {...},
      "target": {...},
      "manifest_digest": "sha256:...",
      "image_ref_digest": "registry.example.com/models/qwen3-30b@sha256:...",
      "completed_at": "2026-05-07T12:00:00Z",
      ...
    }
  }
}
```

Aucun bump de `version`. Les state files existants n'ont pas le champ ; la
lecture est tolérante (`job.get("image_ref_digest")` peut être `None`).

### 4.3 Signature de `mark_completed`

```python
# state.py
def mark_completed(
    self,
    job_key: str,
    manifest_digest: str,
    image_ref_digest: str,
) -> None:
    with self._lock:
        job = self._data["jobs"][job_key]
        job["manifest_digest"] = manifest_digest
        job["image_ref_digest"] = image_ref_digest
        job["completed_at"] = _now_iso()
        job["updated_at"] = _now_iso()
```

Breaking pour les appelants internes (un seul site dans `runner.py`). Pas de
backward-compat shim : on n'a qu'un appelant et le breakage est intentionnel.

### 4.4 Chemins dans `runner.py`

#### 4.4.1 Job complété détecté en début de run (lignes 111-120)

```python
if state.is_completed(job_key) and not cfg.force:
    existing = state.get_job(job_key)
    assert existing is not None
    manifest_digest = str(existing["manifest_digest"])
    image_ref_digest = (
        existing.get("image_ref_digest")
        or f"{cfg.registry}/{cfg.target_repo}@{manifest_digest}"
    )
    plog.info(f"Job already completed: {manifest_digest}")
    plog.output_variable("manifestDigest", manifest_digest)
    plog.output_variable("imageRef", image_ref)
    plog.output_variable("imageRefDigest", image_ref_digest)
    return RunResult(
        job_key=job_key,
        manifest_digest=manifest_digest,
        image_ref=image_ref,
        layers=[],
    )
```

Le fallback sur la reconstruction (`f"{registry}/{repo}@{digest}"`) garantit
que les state files antérieurs à v0.2 ré-émettent quand même
`IMAGEREFDIGEST` au second run.

#### 4.4.2 Cas push réussi (autour des lignes 271-275)

```python
image_ref_digest = f"{cfg.registry}/{cfg.target_repo}@{manifest_digest}"

state.mark_completed(
    job_key,
    manifest_digest=manifest_digest,
    image_ref_digest=image_ref_digest,
)
state.save()

plog.output_variable("manifestDigest", manifest_digest)
plog.output_variable("imageRef", image_ref)
plog.output_variable("imageRefDigest", image_ref_digest)
```

### 4.5 `oci-modelcar status`

La sortie actuelle est dense sur une ligne :

```
job=abc123def456  hf_repo@sha  -> registry/repo:tag  digest=sha256:...  completed=...
```

Ajouter une seconde ligne (indentée) seulement quand `image_ref_digest` est
présent :

```
job=abc123def456  hf_repo@sha  -> registry/repo:tag  digest=sha256:...  completed=...
    ref=registry.example.com/models/qwen3-30b@sha256:...
```

Pour les jobs en cours (pas de `image_ref_digest`), la ligne supplémentaire
est omise.

### 4.6 Documentation README

Nouvelle section après "OCI compliance", avant "Releasing (maintainers)".
Le bloc ci-dessous utilise quatre backticks pour englober un fragment
Markdown qui contient lui-même des fences ```bash` à trois backticks.

````markdown
## Signing & verification

`oci-modelcar` itself does not sign artifacts — signature is delegated to
[cosign](https://github.com/sigstore/cosign), the canonical OCI signing tool.
Each `push` exposes the canonical digest reference for direct piping into
cosign:

```bash
oci-modelcar push --hf-repo ... --registry ... --target-repo ...
# IMAGEREFDIGEST=registry.example.com/models/qwen3-30b@sha256:...

# Sign keyless (CI with OIDC, e.g. GitHub Actions with id-token: write)
cosign sign $IMAGEREFDIGEST

# Sign with a static key (offline / regulated environments)
cosign generate-key-pair                       # one-time, produces cosign.key + cosign.pub
cosign sign --key cosign.key $IMAGEREFDIGEST

# Verify (consumer side, e.g. KServe operator)
cosign verify $IMAGEREFDIGEST \
    --certificate-identity-regexp '^https://github\.com/your-org/' \
    --certificate-oidc-issuer 'https://token.actions.githubusercontent.com'

# Or with the static public key
cosign verify --key cosign.pub $IMAGEREFDIGEST
```

The signature is stored as an additional artifact in the same OCI registry,
attached to the manifest by digest (referrers API for OCI Distribution v1.1+,
or `:sha256-<digest>.sig` tag for legacy registries — cosign auto-detects).

### PyPI artifact

The `oci-modelcar` PyPI distribution is signed with PEP 740 attestations
generated by GitHub Actions in keyless OIDC mode. Verify after install:

```bash
pip install pypi-attestations
python -m pypi_attestations verify pypi \
    --repository codanael/oci-modelcar \
    "$(pip download --no-deps --no-build-isolation -d . oci-modelcar | tail -1 | awk '{print $NF}')"
```
````

### 4.7 Entrée CHANGELOG

````markdown
## [Unreleased]

### Added
- `IMAGEREFDIGEST=<registry>/<repo>@sha256:<digest>` variable in `push`
  output, suitable for direct piping into `cosign sign`.
- `image_ref_digest` field persisted in the state file and surfaced by
  `oci-modelcar status`.
- PEP 740 digital attestations published with PyPI artifacts via Sigstore
  keyless OIDC. Verifiable with `pypi-attestations` or `cosign verify-blob`.
- README section documenting cosign sign/verify recipes (keyless and
  static-key) for modelcar OCI images.

### Changed
- `JsonStateStore.mark_completed()` now requires `image_ref_digest=` keyword
  argument. Breaking change for direct API consumers; CLI users unaffected.
````

---

## 5. Stratégie de tests

### 5.1 Tests unitaires (modules)

#### `tests/unit/test_state.py` — 1 nouveau test

```python
def test_mark_completed_persists_image_ref_digest(tmp_path):
    store = JsonStateStore(tmp_path / "state.json")
    store.upsert_job("k", JobState(...))
    store.mark_completed(
        "k",
        manifest_digest="sha256:aaa",
        image_ref_digest="r.example.com/repo@sha256:aaa",
    )
    store.save()

    fresh = JsonStateStore(tmp_path / "state.json")
    job = fresh.get_job("k")
    assert job["image_ref_digest"] == "r.example.com/repo@sha256:aaa"
    assert job["manifest_digest"] == "sha256:aaa"
```

Pas de test pour la lecture d'un state file ancien sans le champ — couvert
par les tests d'intégration via le fallback `runner.py`.

### 5.2 Tests d'intégration (end-to-end CLI)

#### `tests/integration/test_runner_single.py` — extension

Le test existant qui valide une push complète asserte déjà `IMAGEREF=...:tag`
dans la sortie. Ajouter une ligne :

```python
assert "IMAGEREFDIGEST=" in out
assert "@sha256:" in out  # par opposition au :tag
# Forme attendue : IMAGEREFDIGEST=<host>/<repo>@sha256:<hex>
import re
m = re.search(r"IMAGEREFDIGEST=([^\s]+)@sha256:([0-9a-f]{64})", out)
assert m is not None, f"missing IMAGEREFDIGEST in:\n{out}"
```

#### `tests/integration/test_cli.py` — extension de `_run_status`

```python
def test_status_shows_image_ref_digest_when_present(tmp_path):
    # Préparer un state file avec un job complété qui a image_ref_digest.
    # Lancer `oci-modelcar status --state-file <path>`.
    # Asserter que la sortie contient "ref=<...>@sha256:<...>".

def test_status_graceful_when_image_ref_digest_missing(tmp_path):
    # State file legacy v0.1.0 sans image_ref_digest.
    # `status` doit afficher la ligne principale sans crasher,
    # sans afficher la ligne "ref=...".
```

### 5.3 Idempotence

Test additionnel dans `test_runner_single.py` :

```python
def test_second_run_reemits_image_ref_digest(...):
    # Push une fois (état completed).
    # Re-run : la sortie doit contenir IMAGEREFDIGEST même si rien n'est
    # poussé. Le job est lu depuis state, le runner ré-émet la variable.
```

Et un cas dégradé qui simule un state file legacy :

```python
def test_legacy_state_reconstructs_image_ref_digest(tmp_path):
    # State avec manifest_digest mais sans image_ref_digest (v0.1.0).
    # Re-run --force=False : la sortie reconstruit la ref via le fallback
    # f"{registry}/{repo}@{manifest_digest}" et émet IMAGEREFDIGEST.
```

### 5.4 Pas de tests CI/PyPI

Le flag `attestations: true` est purement déclaratif côté `release.yml`. Sa
validation se fait à la prochaine release tag (vérification manuelle sur
pypi.org → project → publishing). Aucun test unitaire ne peut couvrir ça.

### 5.5 Pas de tests E2E

L'E2E pousse vers un `registry:2` local et n'invoque pas cosign. Ajouter
cosign dans la chaîne E2E demanderait Fulcio mock + Rekor mock (sigstore
sandbox) — rapport coût/valeur défavorable. La signature est documentation,
pas comportement de l'outil. Couvert par tests unitaires/intégration de la
sortie.

---

## 6. Compatibilité ascendante

### 6.1 State files v0.1.0

Les state files générés par v0.1.0 n'ont pas de `image_ref_digest`. La
lecture est tolérante :

- `oci-modelcar status` détecte l'absence et omet la seconde ligne.
- `oci-modelcar push` (re-run idempotent sur job complété) reconstruit la ref
  via `f"{registry}/{repo}@{manifest_digest}"` et la persiste au prochain
  `mark_completed`. Pas de migration explicite.

### 6.2 Sortie de `push`

Tout consumer existant qui parse `MANIFESTDIGEST=…` ou `IMAGEREF=…:tag`
continue de marcher : ces variables sont conservées telles quelles. Seule
une nouvelle variable `IMAGEREFDIGEST` est ajoutée.

### 6.3 API interne `JsonStateStore.mark_completed`

Breaking : la signature ajoute un kwarg requis `image_ref_digest`. Aucun
appelant externe documenté ; un seul appelant interne (`runner.py`) est mis
à jour dans le même PR. Pas de shim de transition.

### 6.4 PyPI download

Les versions antérieures restent accessibles sans attestation. À partir de
v0.2.0, chaque release tag publie attestations + dist files. PyPI conserve
les attestations indéfiniment. Pas de re-signature des releases v0.1.x.

---

## 7. Plan de release

1. Branche `feat/cosign-integration`, TDD task-par-task.
2. Tests unitaires + intégration verts. Hooks pre-commit verts.
3. Merge fast-forward sur `main`.
4. Bump `pyproject.toml` `version = "0.2.0"`, finaliser `CHANGELOG.md`
   `[0.2.0]`.
5. `git tag -a v0.2.0 -m "..."` et `git push origin v0.2.0`.
6. Le workflow `release.yml` (avec `attestations: true`) déclenche la
   release, publie le wheel + sdist + attestations PEP 740 sur PyPI.
7. Vérifier sur pypi.org → oci-modelcar → 0.2.0 que les attestations sont
   listées.
8. `gh release view v0.2.0` pour vérifier les artefacts (sans bundles
   .sigstore — décision §1.3).

---

## 8. Risques et mitigations

| Risque | Impact | Mitigation |
|---|---|---|
| Attestations PEP 740 désactivées par défaut PyPI à l'avenir | Le flag silently no-op | `attestations: true` est explicite ; surveiller le changelog `pypa/gh-action-pypi-publish` |
| Trusted Publisher PyPI mal configuré | Release échoue à publier | Déjà en place pour v0.1.0, pas de changement |
| Utilisateur ne sait pas où vérifier l'identité d'origine | Vérification cosign échoue ou est trop permissive | README montre l'expression `--certificate-identity-regexp` exacte pour le repo |
| Cosign cassant la rétrocompat referrers/tag-based | Échec sur registries anciens | Documenté : cosign auto-détecte ; aucun verrouillage par notre tool |
| Signing keyless dépend de Fulcio/Rekor uptime | Si Sigstore down, pas de signature | Acceptable ; mode clé statique disponible en fallback offline |
