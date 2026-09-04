# Final-sprint repository hygiene

## Result

The stage 1–6 production implementation is mechanically organized and verified.
No deployment, DNS change, commit, or push occurred. Production remains the
stateless FastAPI/Lambda path with embedded standalone `pi1`, separate optional
Bedrock narration, Regional API Gateway HTTP API, and the GitHub Pages
`/Dracula/` frontend. SQLite remains local-development storage only.

The selected ignored model and local gameplay records were preserved:

| Protected artifact | Verification |
| --- | --- |
| `runs/bgc-policy-pi1-001/artifacts/unaccepted-candidate.pt` | SHA-256 `70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c` |
| `.local/dracula.sqlite3` | SHA-256 `adf507cfb41c22dfbe63cbc5a7cf8cc97affb972d053ba9d2894e7468f3694ba`; `PRAGMA integrity_check` returned `ok` |
| `.local/dracula.sqlite3-wal` | SHA-256 `4e4d5d7a11a11392f4815c9a850cd25d6409193fc58c0ebb7b3c13f30ed76a02` before and after validation |

The SQLite SHM file remains protected but is transient lock metadata; opening
the database for a read-only integrity check legitimately changed that file.
Card packs, the Dracula portrait, frontend card assets, lockfiles, historical
reports, archived experiments, and local staging evidence remain intact.

## Repairs

- Separated the selected standalone `pi1` setting
  (`DRACULA_BGC_POLICY_ARTIFACT`) from the historical accepted-`pi0`
  continuation setting (`DRACULA_BGC_PI0_ARTIFACT`). Local gameplay,
  Playwright, and the Lambda image now bind `bgc-policy` to the former only.
  A regression test proves a `pi0` environment setting cannot silently select
  the release opponent.
- Changed `make release-candidate` to rebuild both the Lambda image and
  production frontend before sealing their identities. It can no longer seal
  an older image tag accidentally.
- Made the release source inventory ignore `__pycache__`, bytecode,
  `.pytest_cache`, `egg-info`, and `dist-info`. A regression test covers the
  filter. Absolute output paths are now reported safely by the release tool.
- Added root `.env` and `.env.*` protection while keeping `.env.example`
  eligible for Git.
- Replaced the one tracked absolute workstation path in the collector-isolation
  report with `<repository>`.
- Removed duplicate release instructions from the root README in favor of the
  authoritative production release plan. Updated deployment, report, tracker,
  and infrastructure breadcrumbs for the actual alarm surface and current
  configuration names.
- Removed rebuildable Python bytecode, pytest state, package metadata, wheel
  output, frontend distribution/test output, temporary container validation
  output, and old local Docker-image tags.
- Deleted the generated Lambda context after validation. It was the only build
  copy of the ignored `pi1` binary; no `.pt` file remains under `build/` or
  `frontend/`.
- Retained only `build/release-candidate/manifest.json` as ignored, reproducible
  release evidence. Historical top-level output logs remain ignored evidence;
  reorganizing or archiving them is outside this restrained deployment-hygiene
  stage.

No debug production route was found. Production logging is structured and
bounded to request status, timing, token counts, and policy work metrics; it
does not log recovery histories, seeds, hands, stock, logits, tensors, or
engine objects.

## Release identity after hygiene

The resealed local candidate is reproducible byte for byte:

| Component | Identity |
| --- | --- |
| Candidate content | `fa9f31c2aadfa9f103e65b94844af86a95314e94a3aebdab06f3cff1087d6a29` |
| Manifest file | `4ca9db52eba4be1ec50bdf897b7b3a2964b453c51278eaa9d8b52af7187c446d` |
| Release source tree | `5a788bf3f9fe447fdf48acf276c226059a8bc45d921a3cce69010fdcdd8ddda3` |
| Embedded `pi1` | `70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c` |
| Local arm64 image | `sha256:41af66013b874447c120c04fa0cdb9ac8fd4e67868567ed23614af23966a0626` |
| Lambda context | `9a60772fa8e8a35167c4d283095460d40857769dfd98e65d87ee3554d4027df7` |
| Infrastructure tree | `a467a6335f97ecc556b52ec1fe94d43c34ee31cf835cae28ed1e1163f881e58c` |
| Frontend build | `e69836fb4a4b0ab42eec4c830ca8d91f3cf7431af2c838de75460da03cd80e1f` |
| Dependency inputs | `5b783ad8d7667e491294bb8ef74b3f968f44b9ec1cf61d7bff999ff4a05fcd05` |

The temporary local Docker tag was deleted after validation. Rebuilding with
`make release-candidate` reproduces the candidate inputs; an ECR registry
digest remains intentionally unavailable until an authorized publication.

## Validation

| Surface | Command/evidence | Result |
| --- | --- | --- |
| Complete Python suite | `.venv/bin/python -m pytest` | 679 passed in 629.34 seconds |
| Frontend unit/components, TypeScript, lint, build | `npm --prefix frontend run check` | 12 files, 84 tests; TypeScript, ESLint, and Vite production build passed |
| Browser production flow | `make pages-test` | 2 Playwright tests passed, including a complete stateless six-round game |
| Package/import | `pip check`, `compileall`, isolated `pip wheel`, direct production-module imports | passed; wheel built successfully |
| Infrastructure | `cfn-lint infrastructure/ecr.yaml infrastructure/application.yaml`; parameter JSON parsing | passed |
| Lambda image | `tools/validate_lambda_container.py` | 4 games and 124 requests; replay/cache/privacy/legal-action checks passed |
| Container metrics | same validator | 996,549,884-byte final image; 1.46–1.79 s local initialization; 8.96 ms mean warm gameplay; 1.75 ms mean `pi1` inference; 212,860,928-byte peak memory |
| Release seal | two independent `build_release_candidate.py` runs plus `cmp` | exact byte match |
| Markdown | internal-link validator over tracked and eligible untracked Markdown | 69 files, zero broken links after this report was added |
| Tracker | repository-wide tracker-ID scan | IDs confined to `docs/design-tracker.md` |
| Formatting | `git diff --check` | passed |
| Secrets | high-signal scan of every tracked and eligible untracked file | no AWS access key, private key, account ARN, GitHub/Slack token, or Cloudflare token match |
| Workstation paths | tracked and eligible untracked text scan | zero absolute `/Users/...` or `/home/...` paths |
| Build context | file inventory and content inspection | one verified model before deletion; no database, secret, certificate, symlink, workstation path, or private state |
| Production frontend | production-dist inspection | `/Dracula/`, `https://api.ian-tincknell.com`, and no localhost reference |
| Protected records | SHA-256 plus SQLite integrity check | preserved as described above |

The release manifest itself is ignored generated evidence. The selected model
is ignored under `runs/`, local gameplay is ignored under `.local/`, and
frontend/build output remains ignored. No generated run data is eligible for
Git. The untracked eligible files reported by Git are the intentional stage
1–6 source, tests, workflows, templates, documentation, and reports—not runtime
payloads.

## Retained conditions

- Hosted staging remains unperformed because the available AWS credentials
  returned `ExpiredToken`; no account or region was guessed.
- The final Bedrock model, narration voice, region, Lambda sizing/concurrency,
  alarms/budget, Cloudflare cutover mode, public disclosure, and go-live
  authorization remain explicit later user decisions.
- Historical controller and training implementations remain for artifact
  inspection and comparison. They are not silent production fallbacks.

These are release decisions or historical compatibility, not repository
hygiene defects. No genuine unresolved mechanical hygiene defect remains.
