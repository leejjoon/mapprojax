# Stage 5 validation closure — 2026-11-07

**Status: validated payload; closure commit/push, main promotion and publication pending.** This is a prepared evidence-only closure, not a claim that the closure SHA has passed CI or reached main. Fresh closure commit/push authorization remains required by the approved plan. Do not rewrite this evidence after it is committed; append corrections or a new-candidate request instead.

## Exact identities and original files

- Tested mapprojax payload / CI producer: `24a8e33db256ae0a9f0a5914b9d5b270979059b6` on `rocm-extra`.
- Downstream source and installed consumer: reprojax `79ee7299b209a83c6f247a1fe2e6f208f50baac6`, version 0.2.0; no source edit or dependency re-pin.
- Planning/runtime-invariance baseline: mapprojax `e5b588afb4965a27803ffe752d0cfb3e27712dc2`. The prior Stage 4 recorded runtime pair was mapprojax `300f11b53264c6e64e393a03b8292fffdb7b9b13` / the same reprojax revision, not this producer revision.
- Successful [CI run 42348866254](https://github.com/leejjoon/mapprojax/actions/runs/42348866254), attempt 1; all three jobs succeeded on the exact payload SHA.
- Original distribution artifact: `release-original`, immutable artifact ID `13564609432`. Downloaded by that ID, container digest verified, then the original wheel/sdist manifests and package-input identity verified before local use. No replacement build was used.
- Actual API creation `2026-11-07T07:02:44Z`; expiry `2027-02-05T06:52:15Z`; observed unexpired on 2026-11-07. Retention requested 90 days; the actual API expiry governs. Missing, deleted or expired artifacts require a newly validated candidate even if matching local files remain.
- Hosted installed-validation logs: artifact `13564439601`, downloaded and container digest verified separately from distributions.

| Original file | Bytes | SHA-256 |
|---|---:|---|
| `mapprojax-0.3.0-py3-none-any.whl` | 31303 | `64cd69708f9149b99d2ca01b7dfa2ff449513e9ce4a8e7cce84f70462e174c32` |
| `mapprojax-0.3.0.tar.gz` | 95908 | `e31cc1e2c2e21d263ad0e693b07d4cdfe5ea2d105eb020bc06e4eb0b51602bd0` |

The complete operative producer record follows. Its timestamps are retained exactly as returned by the API.

```stage5-artifact-json
{
  "schema_version": 1,
  "repository": "leejjoon/mapprojax",
  "version": "0.3.0",
  "producer_sha": "24a8e33db256ae0a9f0a5914b9d5b270979059b6",
  "input_sha256": "704fee5fcfeaba9d23963db92510845fba171f31336a4640ef6e112e8148be9b",
  "run_id": 42348866254,
  "run_attempt": 1,
  "artifact_id": 13564609432,
  "artifact_name": "release-original",
  "artifact_digest": "sha256:35e8d3710dc61d477e4261d0a3ff47c8b93b8fe615e7439a8b6b8eff34d0ef9b",
  "created_at": "2026-11-07T07:02:44Z",
  "expires_at": "2027-02-05T06:52:15Z",
  "expired": false,
  "deleted": false,
  "manifest": {
    "schema_version": 1,
    "project": "mapprojax",
    "version": "0.3.0",
    "artifacts": {
      "mapprojax-0.3.0-py3-none-any.whl": {
        "size": 31303,
        "sha256": "64cd69708f9149b99d2ca01b7dfa2ff449513e9ce4a8e7cce84f70462e174c32"
      },
      "mapprojax-0.3.0.tar.gz": {
        "size": 95908,
        "sha256": "e31cc1e2c2e21d263ad0e693b07d4cdfe5ea2d105eb020bc06e4eb0b51602bd0"
      }
    }
  }
}
```

## Environments, source and installed gates

Hosted CPU/build jobs use Ubuntu 24.04 and standard-GIL interpreters. The initial run `42348583912` for payload `b3d11321b7ea497eaf9fdf33cf0d6910c53aa260` failed before producing distributions: floating `ubuntu-latest` had moved to 26.04, without the exact Python 3.14.4 setup-python distribution. The official Python manifest confirmed both required versions on 24.04; the appended compatibility fix produced the green payload above. No Python/dependency floor, test or warning was weakened.

- Hosted minimum: Python 3.11.15, JAX/jaxlib 0.10.2, NumPy 2.4.6, Astropy 7.2.0, pytest 9.0.3; exact [minimum constraints](../../ci/minimum.txt).
- Hosted latest recorded combination: standard Python 3.14.4, JAX/jaxlib 0.12.0.1, NumPy 2.5.4, Astropy 8.0.2, pytest 9.2.0; exact [latest constraints](../../ci/latest.txt).
- Local reference CPU/ROCm: Python 3.12.13, JAX/jaxlib 0.11.1, NumPy 2.5.1, Astropy 8.0.2, pytest 9.1.1; ROCm plugins 0.11.1. ROCm reports backend `gpu`, device `AMD Radeon Graphics` on Strix Halo, client `PJRT C API\nrocm 70253211`, not CUDA. Backend/device assertions passed on source and installed paths.
- Exact build environment: build 1.6.1, setuptools 84.2.1, wheel 0.48.0, Twine 7.0.0, bootstrapped with uv 0.11.8 from [build constraints](../../ci/build.txt). Producer-side Twine checked both original files before manifest/upload.

| Gate | Ordinary pytest result |
|---|---|
| Hosted minimum CPU source | 949 passed, 1 warning |
| Hosted latest CPU source | 949 passed, 1 warning |
| Hosted original wheel CPU install | 949 passed, 1 warning |
| Hosted original sdist CPU install | 949 passed, 1 warning |
| Local clean committed mapprojax source, reference CPU | 949 passed, 1 warning |
| Local clean committed mapprojax source, reference ROCm | 949 passed, 1 warning |
| Local original CI wheel, reference CPU | 949 passed, 1 warning |
| Local original CI sdist, reference CPU | 949 passed, 1 warning |
| Local original CI wheel, reference ROCm | 949 passed, 1 warning |
| Exact reprojax source pair, CPU | 100 passed, 2 GPU-only skips |
| Exact reprojax source pair, ROCm | 102 passed |
| Exact reprojax installed consumer of original wheel, CPU | 100 passed, 2 GPU-only skips |
| Exact reprojax installed consumer of original wheel, ROCm | 102 passed |

The retained warning is the pre-existing x64-off float64-truncation warning in `test_works_without_jax_enable_x64`. Existing ROCm SMI/bandwidth diagnostics remain visible. CPU skips are exactly `test_batch.py:118` and `:153`, requiring a GPU; ROCm runs both. No new skips, xfails, filters or tolerance increases were introduced.

Both CPU installation paths ran the complete external fixture payload and all literal README/manual Python examples in fresh CPU children. The ROCm installed suite was run once on the direct original wheel; its documentation examples deliberately run on CPU. Parent and original-helper child provenance assertions checked actual purelib module origins, interpreter, prefix and distribution version. Both original subprocess helpers remain unchanged. The intentional shadow package failed specifically with the required mapprojax-origin AssertionError.

The comparison wheel built from the original sdist matched the original direct wheel's member contents and normalized RECORD; no third full suite ran on that comparison wheel. Downstream dependency installation did not substitute mapprojax or edit reprojax. Separate source-pair assertions resolved both packages into their exact clean source archives; installed-pair assertions resolved them into the original-wheel consumer venv.

## Frozen evidence and numerical provenance

- Explicit protected-list checkout diff against `e5b588a` passed. All 165 original protected files matched the SHA manifest captured before extraction and checked again in the gitless committed source archive; new Stage 5 evidence is not part of that old protected set.
- All nine original wheel package modules were byte-equal to their committed archive sources. Their docstring-stripped executable ASTs equalled `e5b588a`; executable constants/runtime error strings were preserved. The two original helper function source segments were identical.
- Existing accuracy CLI budgets passed on hosted minimum/latest and local reference CPU/ROCm. Reference CPU regenerated markdown was byte-identical to the full committed `docs/benchmarks/accuracy_results.md`; no non-reference rendered-row equality was required. No caps or protected reports were rewritten.
- Original reference lock SHA-256: `6c2331ea97458144b198538b4e03752bb0cccfcedf18a622c06c8ea23217f7c6`. Reference constraints were exported frozen from the preserved original revision before refreshing metadata/lock, with test/ROCm extras, no dev and no project.
- Git archive identities: mapprojax producer tar SHA-256 `bc1a67dd7d4034063ef3c63dc91481527fa7319c566dcd3de7d196a4221e47f9`; rooted gzip reprojax archive SHA-256 `f2113834691e120761e9fa2a34e3af02f0d688120af8809c7593f392e5e7b8fd`; protected manifest SHA-256 `b1ee6cbaf7a37bc4c4253db1acc49f452392f018c24420dbbfd5e096d08b68dc`. Archive byte identities and the separate package-input digest above are not interchangeable.
- Corrected precision measurements and raw CPU/ROCm reports are indexed in [the dated report](../../docs/benchmarks/precision_results_stage5_2026-11-07.md): warm timings cover 512²–4096² only, three warmups/ten repeats. The CPU attempt including 8192² exited 137 before reporting, cause unproven; no 8192² timing is claimed. Original 5000-point sampling and 4096²/CRPIX-2048 plate-scale geometry remain intact. Sampled agreement is not a universal error bound or independent truth.

## Retained evidence and reproduction

This execution retained the original ZIP/bundle, producer/API metadata, source archives, protected manifest, runtime/origin observations, complete hosted logs and individual source/installed/downstream logs under `/var/tmp/mapprojax-stage5-ci-24a8e33`. Hosted run/log artifact IDs above provide the retained service-side evidence until actual expiry. Local paths alone are not durable publication authority and never bypass expired service artifacts.

Use the bounded commands in [DEVELOPER.md](../../DEVELOPER.md) with the immutable original artifact and exact clean source pair. Verify its manifest before and after checks; use `ci/reference-cpu.txt` and `ci/reference-rocm.txt` for local checks, and `ci/build.txt` for each build backend. The artifact has not been repacked or rebuilt for publication. The closure-only input exclusion was designed to let subsequent exact closure/main CI reuse these same original bytes.

## Actual outward state and remaining gates

Authenticated read-only API observations on 2026-11-07 found:

- Remote main still `f2a6592145016e399c45be004b0dff9accafc4d3`; no main promotion was performed. This observation is not permission or a substitute for a fresh ancestry/live-state check before a separately authorized fast-forward.
- Repository description is null and topics are empty; their agreed updates remain separately unauthorized, not silently applied.
- `GET repos/leejjoon/mapprojax/environments/pypi` returned HTTP 404 while the authenticated repository response reported admin access. Therefore no protected environment/reviewer configuration is established by this evidence. PyPI pending/trusted-publisher configuration and its exact owner/repository/workflow/environment tuple have not been verified. No service setting was created or changed.
- Upload remains hard-disabled (`if: false`). No arming, tag, protected upload approval, PyPI publication/verification or downstream re-pin was performed or authorized.

Fresh closure commit/push permission, green CI on the exact closure SHA, separately authorized ff-only main promotion/push and its applicable green checks remain required. Publication additionally requires actual operator reviewer/publisher proof and separate arming/tag/upload permissions. Arming changes workflow inputs and requires a fresh validated producer if done after this payload. This document does not claim those pending gates passed, does not cite its own future commit SHA and does not declare the full Stage 5 objective complete.

## 2026-11-07 — operator setup, authorized arming and new candidate

The preceding closure is preserved byte-for-byte as historical evidence. Operator-authorized arming changes the workflow input digest, so the old producer is not the publication candidate for the armed workflow. A fresh original CI artifact and the full exact-byte local gate are required.

Dated authenticated read-only `gh api --include` evidence was captured on 2026-11-07 at approximately 20:40 UTC+09:00, with the HTTP Date headers retained separately from the local observation clock, under `/var/tmp/mapprojax-stage5-arming-cycle`:

- `repos/leejjoon/mapprojax/environments/pypi`: environment ID `26561854210`, required reviewer `leejjoon`, `prevent_self_review: false`, `can_admins_bypass: false`, custom deployment policies enabled and protected-branches-only disabled.
- Its `deployment-branch-policies` endpoint: exactly one policy, type `tag`, pattern `v*` (ID `65841023`); no branch deployment policy.
- Active main ruleset `26754429`, no bypass actors and current-user bypass `never`, targets the default branch `main`, blocks deletion and non-fast-forward updates, and requires GitHub Actions integration `15368` checks `packaging`, `cpu (3.11.15, ci/minimum.txt)` and `cpu (3.14.4, ci/latest.txt)`. The effective `rules/branches/main` response confirms those checks and restrictions.
- Active tag ruleset `26754613`, no bypass actors and current-user bypass `never`, targets `refs/tags/v*`, blocks update, deletion and non-fast-forward changes; tag creation is not restricted by a creation rule.
- Repository description is `Differentiable JAX coordinate kernel for FITS: TAN/SIN WCS with SIP`; topics are `astronomy`, `fits`, `jax`, `wcs`.
- `https://pypi.org/pypi/mapprojax/json` returned HTTP 404; response headers/body and observation time were retained. No project publication or installation was attempted.

Operator statement dated 2026-11-07: PyPI account `lee_j_joon` has configured a pending publisher for project `mapprojax`, GitHub owner `leejjoon`, repository `mapprojax`, workflow `publish.yml`, environment `pypi`. The supplied screenshot at `/home/jjlee/.paseo/uploads/upload_e1e54dca-6fc8-41f3-bc26-1103f9841cdd/clipboard.png` was visually inspected: it shows that pending-publisher tuple and account. The statement and screenshot identity were recorded separately; this is operator/screenshot evidence, not an authenticated PyPI API query or proof that publication has happened.

Read-only starting checks confirmed remote main and `rocm-extra` both at `3d2c62c253e7fcf6ab57a13848c78f8f5b943356`, no local or remote tags, and no tracked/staged changes. The bounded authorization permits only the four-file arming commit, a newly produced candidate's full validation, one appended evidence closure and guarded promotion of that green closure to main. No tags, protected publish approval, upload, settings changes, PyPI install verification, downstream edits or additional closure appends are authorized. Publication remains pending.

<!-- stage5-new-candidate -->

## 2026-11-08 — armed candidate exact-byte validation closure

**Status: candidate validated; closure CI, main promotion and publication pending.** This appended entry supersedes the preceding pending-candidate selection without changing any historical bytes. It records the tested producer, not its own future closure commit SHA.

Commit A is `d6bb72e400c973500142811ed3e15d134ff9569f`, parent `3d2c62c253e7fcf6ab57a13848c78f8f5b943356`. Its four explicitly staged paths arm the protected upload condition, update that assertion and disabled-upload guidance, and append the operator setup/new-candidate request. The local release-workflow suite passed **92 tests**; producer [CI 42369336825](https://github.com/leejjoon/mapprojax/actions/runs/42369336825), attempt 1, passed all three jobs and built/uploaded a new original artifact rather than reusing the old one.

The first local preparation attempt stopped on 2026-11-07: an uncompressed Git archive was passed to the gzip-only extractor before any local suite began. The operator explicitly authorized resumption on the same candidate on 2026-11-08. That failed `source.tar` remains untouched at `/var/tmp/mapprojax-stage5-arming-cycle/source.tar`, SHA-256 `a14b27cf02a30ee82953152c36641302000ca4aac3a8e4a417fa5dc599c9eb4a`. The resumed gate used fresh absolute OUT `/var/tmp/mapprojax-stage5-resume-26cyrfi5`, fresh per-step outputs and correctly prefixed gzip archives. No preparation retry or verification failure occurred during this resumed gate.

### Archives and original artifact identity

| Archive | SHA-256 |
|---|---|
| `source.tar.gz`, `git archive --format=tar.gz --prefix=mapprojax/` at Commit A | `888ef02cbf08f4816e4077051853828bcf00062dc3ebbbdfcd2b47802ee654c8` |
| `reprojax.tar.gz`, exact preserved rooted gzip archive at `79ee7299b209a83c6f247a1fe2e6f208f50baac6` | `f2113834691e120761e9fa2a34e3af02f0d688120af8809c7593f392e5e7b8fd` |

Original `release-original` artifact **13573590400** was downloaded again by immutable ID into fresh OUT. Its container digest, per-file sizes/hashes, canonical input digest, successful producer run/SHA and actual unexpired API metadata were checked before use and again after all local gates. Actual creation is `2026-11-07T11:52:15Z`; actual expiry is `2027-02-05T11:42:56Z` (2027-02-05 20:42:56 UTC+09:00). Those API timestamps, rather than a requested retention period or the host clock alone, govern availability. No locally built distribution substituted for either original file. The sdist install/comparison builds were limited to the approved installation and RECORD/content checks, not replacement publication artifacts.

```stage5-artifact-json
{
  "schema_version": 1,
  "repository": "leejjoon/mapprojax",
  "version": "0.3.0",
  "producer_sha": "d6bb72e400c973500142811ed3e15d134ff9569f",
  "input_sha256": "932d4842a4f356aa8917330b5228cefaadbb7c3054b05889899968b859717882",
  "run_id": 42369336825,
  "run_attempt": 1,
  "artifact_id": 13573590400,
  "artifact_name": "release-original",
  "artifact_digest": "sha256:dbb47436bd75e7b6496f006a2c1d94a0d371db4ac6f6057ee9643480141ba624",
  "created_at": "2026-11-07T11:52:15Z",
  "expires_at": "2027-02-05T11:42:56Z",
  "expired": false,
  "deleted": false,
  "manifest": {
    "schema_version": 1,
    "project": "mapprojax",
    "version": "0.3.0",
    "artifacts": {
      "mapprojax-0.3.0-py3-none-any.whl": {
        "size": 31303,
        "sha256": "504296da0cea2c03cb1e058e5096310616fd92a2cafb3a20650bd0aa0b34aa8d"
      },
      "mapprojax-0.3.0.tar.gz": {
        "size": 95921,
        "sha256": "61a8958d9796fe2154eb5a7f28b857335a5f829302d3e51b391eeb2be8b5bf9f"
      }
    }
  }
}
```

### Restarted full local gate

| Gate | Ordinary result |
|---|---|
| Clean committed mapprojax source, reference CPU | 949 passed, 1 warning |
| Clean committed mapprojax source, reference ROCm | 949 passed, 1 warning |
| Original CI wheel, reference CPU | 949 passed, 1 warning |
| Original CI sdist, reference CPU | 949 passed, 1 warning |
| Original CI wheel, reference ROCm | 949 passed, 1 warning |
| Exact reprojax source pair, CPU | 100 passed, 2 GPU-only skips |
| Exact reprojax source pair, ROCm | 102 passed |
| Exact reprojax installed consumer, CPU | 100 passed, 2 GPU-only skips |
| Exact reprojax installed consumer, ROCm | 102 passed |

The source suites resolved mapprojax from the clean Commit A archive with explicit source provenance. Their environments used the original CI wheel for matching distribution metadata/dependencies, not a locally rebuilt release. Downstream source probes resolved both mapprojax and reprojax into their exact archives; installed downstream probes resolved the original wheel and installed consumer into purelib. Reprojax's archive was not edited or re-pinned.

Each of the five full mapprojax suites includes **11 literal README/manual examples**, each in its own fresh CPU process, including when the parent suite uses ROCm. Example children clear source PYTHONPATH and use the original wheel in their environment; all nine wheel modules were independently byte-equal to the clean archive sources. Installed parent/child origin audits passed through both unchanged original helpers. Both shadow-PYTHONPATH controls failed specifically with the expected mapprojax-origin AssertionError. The sdist-derived comparison wheel matched the original direct wheel's contents and normalized RECORD; no third full comparison-wheel suite was run.

All **165 protected file hashes** matched the preserved pre-extraction manifest in the checkout and extracted archive, including after source tests. All **nine executable module ASTs**, after docstring stripping, equalled baseline `e5b588afb4965a27803ffe752d0cfb3e27712dc2`; both original subprocess helper source segments were identical. Existing numerical CLI budgets passed on reference CPU and ROCm. The complete regenerated reference-CPU markdown was byte-identical to `docs/benchmarks/accuracy_results.md`; no non-reference rendered-row comparison, cap edit or tolerance change was used.

Reference environments: Python **3.12.13**, JAX/jaxlib **0.11.1**, NumPy **2.5.1**, Astropy **8.0.2**, pytest **9.1.1**; ROCm plugins **0.11.1**. Actual ROCm backend is `gpu`, device `AMD Radeon Graphics`, client `PJRT C API\nrocm 70253211`, not CUDA. Original build constraints remain unchanged and pinned. The retained warning is the existing x64-off truncation warning; CPU downstream skips are exactly the two existing GPU-only batch cases. ROCm SMI/bandwidth diagnostics remained visible. No new skip, xfail or warning suppression was introduced.

### Reconfirmed operator protection and remaining boundary

Fresh dated authenticated `gh api --include` responses on 2026-11-08, with server HTTP Date headers and separately labelled observation times, reconfirmed the setup recorded in Commit A: pypi required reviewer `leejjoon`, self-review prevention off, administrator bypass off, and exactly one `tag` deployment policy `v*`; both active no-bypass rulesets and all three effective GitHub Actions checks on main were unchanged. Main ruleset remains `26754429`, tag ruleset `26754613`, environment `26561854210`. Repository description/topics and default branch `main` were unchanged. Remote main remained the preceding closure `3d2c62c253e7fcf6ab57a13848c78f8f5b943356`, remote candidate was Commit A, with no tags or tracked/staged changes at restart.

The operator's 2026-11-07 PyPI pending-publisher statement and inspected screenshot remain the evidence for account `lee_j_joon`, project `mapprojax`, owner `leejjoon`, repository `mapprojax`, workflow `publish.yml`, environment `pypi`. This is not a fresh authenticated PyPI publisher API query or proof of upload. No repository/PyPI setting, description/topic, downstream source or excluded tracked file was changed during resumption.

OUT retains archive identities, original ZIP/bundle, producer record, dated API evidence, runtime/origin observations, individual full logs and `local-gate-receipt.json`. That receipt records the completed gates without claiming closure or main CI has already passed. The authorized next steps are this single append-only closure commit/candidate push, green exact-closure CI reusing artifact 13573590400 with candidate build/original-distribution upload skipped, guarded normal main promotion/local-ref update, and green main-push reuse CI. Thereafter stop before tags. The armed workflow does not itself authorize any tag, protected publish approval or PyPI upload; publication and post-publication verification remain separately gated. No further closure append is authorized by this cycle.

## 2026-11-09 — post-publication verification evidence

This single evidence-only append follows separate operator authorization after publication. All existing bytes and selection fences above remain historical and unchanged. The immutable lightweight tag **`v0.3.0`** resolves to **`a714f1cc6dc11ce2ebf0da5f0a99fe5b1fc8722f`**. The completed [Publish run 42484721264](https://github.com/leejjoon/mapprojax/actions/runs/42484721264) reports success. Its actual review-history response records **`leejjoon`**, state **`approved`**, environment **`pypi`** (ID **26561854210**), with comment “Approve mapprojax 0.3.0 release,  a714f1c, artifact 13573590400.” The upload log records both original filenames and successful HTTP 200 responses. This verification did not approve a deployment or perform an upload.

### Public release identity

Both project and version JSON were read from PyPI. **0.3.0 was the only release**, and **neither file was yanked**. Both files were downloaded from their public `files.pythonhosted.org` URLs into fresh OUT `/var/tmp/mapprojax-stage5-postpublication-e5h5w1nj`; actual byte lengths and SHA-256 values matched the approved original files:

| Public filename | Bytes | SHA-256 | PyPI upload time (UTC, as returned) |
|---|---:|---|---|
| `mapprojax-0.3.0-py3-none-any.whl` | 31303 | `504296da0cea2c03cb1e058e5096310616fd92a2cafb3a20650bd0aa0b34aa8d` | `2026-11-09T03:47:02.326010Z` |
| `mapprojax-0.3.0.tar.gz` | 95921 | `61a8958d9796fe2154eb5a7f28b857335a5f829302d3e51b391eeb2be8b5bf9f` | `2026-11-09T03:47:03.640281Z` |

Those upload times are 2026-11-09 **12:47:02.326010** and **12:47:03.640281 UTC+09:00**, respectively. Both PyPI integrity endpoints `/integrity/mapprojax/0.3.0/<filename>/provenance` returned **HTTP 200 with JSON records**. Provenance records therefore exist for both files; this records their presence and retained content, not a separate cryptographic attestation-verification claim.

### Independent public-index installations

Two fresh standard-library venvs used the local-gate reference **Python 3.12.13**, with **system site-packages disabled**. All install, audit and quick-start processes ran outside both repositories, from `OUT/A-wheel-cwd` or `OUT/B-sdist-cwd`. `PYTHONPATH` and `PYTHONHOME` were absent. `PIP_CONFIG_FILE=/dev/null`, an explicit sole index `https://pypi.org/simple`, no cache, no extra index, no find-links and no local source were used; inherited pip settings were removed. Pip reports and command/environment records were saved separately for each installation.

| Environment | Public-index selection | pip / JAX / NumPy | Result |
|---|---|---|---|
| `OUT/A-wheel` | `pip install --only-binary mapprojax mapprojax==0.3.0` | 25.0.1 / 0.12.0.1 / 2.5.4 | Install, file/origin audit and exact quick start passed |
| `OUT/B-sdist` | `pip install --no-binary mapprojax mapprojax==0.3.0` | 25.0.1 / 0.12.0.1 / 2.5.4 | Install, file/origin audit and exact quick start passed |

Both reports selected exactly the corresponding verified public URL and approved SHA-256; all reported download URLs were public `files.pythonhosted.org` URLs. The sdist's ordinary isolated installation build did not replace or rebuild a release artifact. `importlib.metadata` reported **0.3.0** in both venvs, and each `mapprojax.__file__` resolved to that venv's own `lib/python3.12/site-packages/mapprojax/__init__.py`. Both audits verified **nine package files** byte-for-byte against the downloaded wheel and sdist, **25 installed RECORD entries**, matching metadata requirements/license, and the **CPU backend**. The wheel install additionally matched all original wheel members other than the install-rewritten RECORD. No source-path fallback was used.

The README **Quick start** Python fence was extracted from immutable commit `a714f1cc6dc11ce2ebf0da5f0a99fe5b1fc8722f` and executed exactly as written, unchanged, in a fresh process in each venv with `JAX_PLATFORMS=cpu`. Both completed successfully, including the block's own finite-coordinate assertion. These are two public-install smoke checks and provenance/content audits, not new full-suite counts.

### Observation and closure boundary

Individual API/download observation times are retained in `OUT/observations.json`; approval JSON, upload log, both public/provenance responses, both downloaded distributions, pip reports/logs, runtime audits and quick-start outputs are retained alongside them. The two quick starts completed at `2026-11-09T04:01:49.753325Z` and `2026-11-09T04:01:51.280183Z`. Pre-edit checks completed at **`2026-11-09T04:02:35.150159Z`** (**13:02:35.150159 UTC+09:00**): both remote branches and the tag were still at the immutable release commit, tracked files/index were clean, and the closure bytes equalled the committed file (SHA-256 `abb9d469a5201c36bd2da6a9df470bcb2fa38423c390a1efb22e3db204e74139`). Untracked/user-owned files were left untouched.

The original CI artifact **13573590400** remained live and selected for reuse; canonical input digest remained **`932d4842a4f356aa8917330b5228cefaadbb7c3054b05889899968b859717882`**. This append introduces no artifact-selection fence or candidate request. CI for the new evidence commit and its guarded main promotion is **not yet claimed** here. No further append is authorized by this cycle, and `v0.3.0` must remain on the immutable release commit. No upload, yank/delete, GitHub Release, tag alteration, workflow rerun, settings change, downstream re-pin or other tracked-file edit was performed by this verification.

## 2026-11-09 — post-release development

mapprojax 0.3.0 is published and verified (sections above). The first post-release commit changes tracked release inputs (CHANGELOG date, README and user-manual install lines, DEVELOPER.md, the roadmap, and the development version `0.3.1.dev0` in `pyproject.toml` and `uv.lock`), so the canonical input digest no longer equals the `932d4842…` recorded for artifact 13573590400, and reuse mode would fail. The marker below therefore switches CI packaging to produce mode. While this request remains pending, each applicable CI run builds and validates fresh development distributions. Candidate and main runs build independently; only `rocm-extra` push runs emit the provenance record eligible for a future release.

These artifacts are development builds, not the published 0.3.0 distributions, and must not be published as 0.3.0. Tag `v0.3.0` is immutable on `a714f1cc6dc11ce2ebf0da5f0a99fe5b1fc8722f` under the `v*` tag ruleset, PyPI rejects re-uploads of the existing 0.3.0 file names, and `release_artifact.py` rejects the pending state for any tag. Before producing the next publication candidate, set its intended version. After successful producer CI and the required local gate on those exact bytes, append its complete artifact record after this marker. All records above remain historical and unchanged.

<!-- stage5-new-candidate -->

## 0.4.0 candidate

Version `0.4.0` was set in `pyproject.toml` and the `from_astropy` work landed on `rocm-extra` before this candidate was requested, so `choose` reports `mode = produce` with `reason = new-candidate-request` and an empty prior closure record for this version. `input_sha256` was computed from the final tracked commit `f2029c46d79265e01cef984da7620536d4b3c8bc` (the CHANGELOG commit) and matches `inputs.json` inside the producer artifact below, so nothing tracked changed between selection and production except this closure file, which the digest excludes.

Producer CI `43357904249` (attempt 1, branch `rocm-extra`, push) built and validated these distributions; all three jobs passed. The local gate ran on the same original bytes with `--reprojax-archive` omitted (D11):

| Check | Constraints | Result |
|---|---|---|
| `check_installed_artifacts.py verify` | — | exit 0 |
| `check --backend cpu` | `ci/reference-cpu.txt` | exit 0, 1031 passed (direct wheel and sdist-derived wheel) |
| `check --backend rocm` | `ci/reference-rocm.txt` | exit 0, 1031 passed, device `AMD Radeon Graphics` |

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `mapprojax-0.4.0-py3-none-any.whl` | 37328 | `0747ab28af8a444746c63017512fa14fcdba0ba98c9cd926eab118646323fef6` |
| `mapprojax-0.4.0.tar.gz` | 111649 | `f9477a6b51dc54c8689e4003602049d50a89d72dfb40ba571e89f17710820681` |

The complete operative producer record follows. Its timestamps are retained exactly as returned by the API.

```stage5-artifact-json
{
  "schema_version": 1,
  "repository": "leejjoon/mapprojax",
  "version": "0.4.0",
  "producer_sha": "f2029c46d79265e01cef984da7620536d4b3c8bc",
  "input_sha256": "642b2d5c20e328b4e0de717dca74fab7fd8509327a4215a8267b9ea81766bb83",
  "run_id": 43357904249,
  "run_attempt": 1,
  "artifact_id": 14006509905,
  "artifact_name": "release-original",
  "artifact_digest": "sha256:be131e12ec43d92976931a7b388c185e9217b57b5879a34fb1559bed031cdef4",
  "created_at": "2026-11-13T03:25:33Z",
  "expires_at": "2027-02-11T03:16:58Z",
  "expired": false,
  "deleted": false,
  "manifest": {
    "schema_version": 1,
    "project": "mapprojax",
    "version": "0.4.0",
    "artifacts": {
      "mapprojax-0.4.0-py3-none-any.whl": {
        "size": 37328,
        "sha256": "0747ab28af8a444746c63017512fa14fcdba0ba98c9cd926eab118646323fef6"
      },
      "mapprojax-0.4.0.tar.gz": {
        "size": 111649,
        "sha256": "f9477a6b51dc54c8689e4003602049d50a89d72dfb40ba571e89f17710820681"
      }
    }
  }
}
```

### Superseded request

The candidate above was produced from `f2029c4`, but a following commit `b1aad1f` fixed a duplicated `## 0.3.0` heading in `CHANGELOG.md`, which is a release input. The recorded `input_sha256` therefore no longer matches HEAD, so those bytes cannot be published as 0.4.0. A fresh candidate is requested below; the record above stays as tested history and is not rewritten.

<!-- stage5-new-candidate -->

Producer CI `43361647063` (attempt 1, branch `rocm-extra`, push) rebuilt from `2f3fd919` after the CHANGELOG fix; all three jobs passed and its `inputs.json` equals the new `input_sha256`. The local gate ran on these original bytes with `--reprojax-archive` omitted (D11):

| Check | Constraints | Result |
|---|---|---|
| `check --backend cpu` | `ci/reference-cpu.txt` | exit 0, 1031 passed (direct wheel and sdist-derived wheel) |
| `check --backend rocm` | `ci/reference-rocm.txt` | exit 0, 1031 passed, device `AMD Radeon Graphics` |

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `mapprojax-0.4.0-py3-none-any.whl` | 37328 | `184f170920c94213ede340e5692fd618f0c88c320848b137bbf3a70756c77375` |
| `mapprojax-0.4.0.tar.gz` | 111626 | `ea8d261826341c50112e4e4f540e4fcffedcaae59e6d43edcd73f32676942d49` |

This record supersedes the `f2029c4` candidate above. The complete operative producer record follows.

```stage5-artifact-json
{
  "schema_version": 1,
  "repository": "leejjoon/mapprojax",
  "version": "0.4.0",
  "producer_sha": "2f3fd919af7ec64511f8ec569b10200a57872a41",
  "input_sha256": "2a3e9eb89fbecff4e1c0eb0dfdb20b0ca1cd7e424606af5ed583f843695798fa",
  "run_id": 43361647063,
  "run_attempt": 1,
  "artifact_id": 14007609778,
  "artifact_name": "release-original",
  "artifact_digest": "sha256:b48e5e6da6ae87b2a56e4ba7635fa52987b20b55aff894e08fc873a449b241d5",
  "created_at": "2026-11-13T03:57:57Z",
  "expires_at": "2027-02-11T03:50:57Z",
  "expired": false,
  "deleted": false,
  "manifest": {
    "schema_version": 1,
    "project": "mapprojax",
    "version": "0.4.0",
    "artifacts": {
      "mapprojax-0.4.0-py3-none-any.whl": {
        "size": 37328,
        "sha256": "184f170920c94213ede340e5692fd618f0c88c320848b137bbf3a70756c77375"
      },
      "mapprojax-0.4.0.tar.gz": {
        "size": 111626,
        "sha256": "ea8d261826341c50112e4e4f540e4fcffedcaae59e6d43edcd73f32676942d49"
      }
    }
  }
}
```

### Superseded request

A later commit `94e144d` restored two tests that `3b72181` had deleted as unintended collateral (the four-fixture round-trip test and the JSON loader-equivalence test). `tests/test_astropy_from.py` is a release input, so the candidate produced from `2f3fd919` no longer matches HEAD. A fresh candidate is requested below; both records above stay as tested history.

<!-- stage5-new-candidate -->

Producer CI `43365655461` (attempt 1, branch `rocm-extra`, push) rebuilt from `89f68562` after the test restoration; all jobs passed and its `inputs.json` equals `input_sha256` `79d54516...`. Gate on these original bytes, `--reprojax-archive` omitted (D11):

| Check | Constraints | Result |
|---|---|---|
| `check --backend cpu` | `ci/reference-cpu.txt` | exit 0, 1036 passed (direct wheel and sdist-derived wheel) |
| `check --backend rocm` | `ci/reference-rocm.txt` | exit 0, 1036 passed, device `AMD Radeon Graphics` |

The count rose from 1031 to 1036 because `94e144d` restored the five fixture-wide round-trip and JSON loader-equivalence cases that `3b72181` had deleted.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `mapprojax-0.4.0-py3-none-any.whl` | 37328 | `183f18a8e42ade7eaf17fbb3b70c61d981e3cda6587f6e05de68672ab01b0613` |
| `mapprojax-0.4.0.tar.gz` | 111972 | `e95683f7cea72135d04182e0b26230b730674804c4702ab232e84d9b11490202` |

This record supersedes both earlier 0.4.0 candidates. The complete operative producer record follows.

```stage5-artifact-json
{
  "schema_version": 1,
  "repository": "leejjoon/mapprojax",
  "version": "0.4.0",
  "producer_sha": "89f685622f97e7f7fa9dcd03b58d1a14071beafa",
  "input_sha256": "79d54516d92d86308daa921707b6182a01b56b9454732703e051253e1f8b3318",
  "run_id": 43365655461,
  "run_attempt": 1,
  "artifact_id": 14010159274,
  "artifact_name": "release-original",
  "artifact_digest": "sha256:f44c0fddb433a650625b5600acfb224c075bf499ea8e70fead07f783631349f6",
  "created_at": "2026-11-13T04:34:30Z",
  "expires_at": "2027-02-11T04:27:32Z",
  "expired": false,
  "deleted": false,
  "manifest": {
    "schema_version": 1,
    "project": "mapprojax",
    "version": "0.4.0",
    "artifacts": {
      "mapprojax-0.4.0-py3-none-any.whl": {
        "size": 37328,
        "sha256": "183f18a8e42ade7eaf17fbb3b70c61d981e3cda6587f6e05de68672ab01b0613"
      },
      "mapprojax-0.4.0.tar.gz": {
        "size": 111972,
        "sha256": "e95683f7cea72135d04182e0b26230b730674804c4702ab232e84d9b11490202"
      }
    }
  }
}
```

### Superseded request

The committee implementation review found blocking defects that change files the release is built from: the x64-off subprocess test did not pin the CPU backend, several acceptance tests were looser than the agreed contract, and the manual's SIP order 0/1 row contradicted astropy. `cef12a9` applies them, so the candidate from `89f68562` no longer matches HEAD. A fresh candidate is requested below; all earlier records stay as tested history.

<!-- stage5-new-candidate -->

Producer CI `43416389659` (attempt 1, branch `rocm-extra`, push) rebuilt from `8dda086d` after the committee review fixes; all jobs passed and its `inputs.json` equals `input_sha256` `b880bb4a...`. Gate on these original bytes, `--reprojax-archive` omitted (D11):

| Check | Constraints | Result |
|---|---|---|
| `check --backend cpu` | `ci/reference-cpu.txt` | exit 0, 1039 passed (direct wheel and sdist-derived wheel) |
| `check --backend rocm` | `ci/reference-rocm.txt` | exit 0, 1039 passed, device `AMD Radeon Graphics` |

The count is 1039: the review fixes tightened existing tests and added the real `DistortionLookupTable` and SIN-horizon cases.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `mapprojax-0.4.0-py3-none-any.whl` | 37640 | `84a4c73055fcf5fc38efb3012c60a4c6149ed28fab1e54d8e9731350f6ab57ae` |
| `mapprojax-0.4.0.tar.gz` | 114565 | `165e0684b4d16410461886b9f536ac38d5be9f15bf740ba58b7d8e53e282e66f` |

This record supersedes all earlier 0.4.0 candidates. The complete operative producer record follows.

```stage5-artifact-json
{
  "schema_version": 1,
  "repository": "leejjoon/mapprojax",
  "version": "0.4.0",
  "producer_sha": "8dda086d7562e0d2144489e8cd61337d0f52badd",
  "input_sha256": "b880bb4a30a81ce2d0a41679f2b733a0f081427185c5e6da4139c7caed2d6440",
  "run_id": 43416389659,
  "run_attempt": 1,
  "artifact_id": 14032566368,
  "artifact_name": "release-original",
  "artifact_digest": "sha256:70fcfb458852e9ac071128980ad399126dd846dd99195abaa89674929abc5d2a",
  "created_at": "2026-11-13T11:27:45Z",
  "expires_at": "2027-02-11T11:21:20Z",
  "expired": false,
  "deleted": false,
  "manifest": {
    "schema_version": 1,
    "project": "mapprojax",
    "version": "0.4.0",
    "artifacts": {
      "mapprojax-0.4.0-py3-none-any.whl": {
        "size": 37640,
        "sha256": "84a4c73055fcf5fc38efb3012c60a4c6149ed28fab1e54d8e9731350f6ab57ae"
      },
      "mapprojax-0.4.0.tar.gz": {
        "size": 114565,
        "sha256": "165e0684b4d16410461886b9f536ac38d5be9f15bf740ba58b7d8e53e282e66f"
      }
    }
  }
}
```

### Superseded request

The re-review left three non-blocking items; `973012c` applies them (a loose `rtol` in the off-triangle inverse, a false convergence comment, and a wrong SIP matrix size in the manual). Two of the three live in `tests/` and the manual, which are release inputs, so the candidate from `8dda086d` no longer matches HEAD. A fresh candidate is requested below; all earlier records stay as tested history.

<!-- stage5-new-candidate -->

Producer CI `43434219124` (attempt 1, branch `rocm-extra`, push) rebuilt from `88c0d17d` after the re-review items; all jobs passed and its `inputs.json` equals `input_sha256` `d03fbf52...`. Gate on these original bytes, `--reprojax-archive` omitted (D11):

| Check | Constraints | Result |
|---|---|---|
| `check --backend cpu` | `ci/reference-cpu.txt` | exit 0, 1039 passed (direct wheel and sdist-derived wheel) |
| `check --backend rocm` | `ci/reference-rocm.txt` | exit 0, 1039 passed, device `AMD Radeon Graphics` |

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `mapprojax-0.4.0-py3-none-any.whl` | 37640 | `ee4d7ad73f099e3069d539bb8fb94873f9a8e42a94a5999568ba268819714960` |
| `mapprojax-0.4.0.tar.gz` | 114764 | `eb07161b065734c9245bd68f399bf95a166e2e618409b4510207e16e6c6c6218` |

This record supersedes all earlier 0.4.0 candidates. The complete operative producer record follows.

```stage5-artifact-json
{
  "schema_version": 1,
  "repository": "leejjoon/mapprojax",
  "version": "0.4.0",
  "producer_sha": "88c0d17d1ab9f6a1381fb0d940de7f5dba58ff59",
  "input_sha256": "d03fbf524d27ca76845a758807e751d54b1af7b0886b0bd89eb53c92d92f9660",
  "run_id": 43434219124,
  "run_attempt": 1,
  "artifact_id": 14042017965,
  "artifact_name": "release-original",
  "artifact_digest": "sha256:ff6d4ce2cd611e6b00d08e9a9e8a2b0b012622e96e9c5ddca0e748f55f6636a3",
  "created_at": "2026-11-13T13:37:42Z",
  "expires_at": "2027-02-11T13:30:24Z",
  "expired": false,
  "deleted": false,
  "manifest": {
    "schema_version": 1,
    "project": "mapprojax",
    "version": "0.4.0",
    "artifacts": {
      "mapprojax-0.4.0-py3-none-any.whl": {
        "size": 37640,
        "sha256": "ee4d7ad73f099e3069d539bb8fb94873f9a8e42a94a5999568ba268819714960"
      },
      "mapprojax-0.4.0.tar.gz": {
        "size": 114764,
        "sha256": "eb07161b065734c9245bd68f399bf95a166e2e618409b4510207e16e6c6c6218"
      }
    }
  }
}
```
