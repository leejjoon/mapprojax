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
