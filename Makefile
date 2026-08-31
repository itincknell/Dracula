CANDIDATE ?= $(CURDIR)/runs/training-004/archives/policy-2-policy-2-v20.pt
GUIDED_ARTIFACT ?= $(CURDIR)/runs/search-warmstart-smoke-001/warm-start-smoke.pt
RESPONSE_RANKER_ARTIFACT ?= $(CURDIR)/runs/teacher-v2-response-ranker-001/artifacts/response-ranker.pt
SAM_POLICY_ARTIFACT ?= $(CURDIR)/runs/sam-policy-training-001/artifacts/policy.pt
BGC_POLICY_ARTIFACT ?= $(CURDIR)/runs/bgc-policy-pi1-001/artifacts/unaccepted-candidate.pt
BGC_PI0_ARTIFACT ?= $(CURDIR)/runs/bgc-policy-pi0-001/artifacts/unaccepted-candidate.pt
DATABASE ?= $(CURDIR)/.local/dracula.sqlite3
OPPONENT ?= bgc-policy
SEARCH_SIMULATIONS ?= 32
SEARCH_EXPLORATION ?= 1.4142135623730951
SEARCH_RESPONSE_COMPLETIONS ?= 1
SEARCH_RESPONSE_SIMULATIONS ?= 32
BELIEF_COMPLETIONS ?= 8
GUIDED_SIMULATIONS ?= 100
INFERENCE_PROFILE ?= argmax-v1
API_PORT ?= 8000
DEV_PORT ?= 5173
PREVIEW_PORT ?= 4173
SAM_CORPUS ?= $(CURDIR)/runs/sam-32-continuous-corpus-001
SAM_FROZEN_SOURCE ?= $(CURDIR)/.local/collector-source/sam-32-continuous-corpus-001
SAM_LOG ?= $(CURDIR)/output-sam-32-continuous-corpus-001
SAM_PID_FILE ?= $(CURDIR)/.local/sam-32-continuous-corpus-001.pid

.DEFAULT_GOAL := help

.PHONY: help check-api-port check-candidate check-opponent dev preview play preview-sam preview-policy preview-bgc-policy preview-control preview-belief-greedy preview-pi0-bgc dev-api dev-frontend play-report play-game reset-local test test-web test-e2e corpus-health corpus-inspect corpus-verify corpus-stop corpus-resume

help:
	@echo "Local gameplay"
	@echo "  make dev              FastAPI and Vite development servers"
	@echo "  make preview          production frontend with standalone pi1"
	@echo "  make preview-sam      explicit nested Sam 32x32 comparison"
	@echo "  make preview-policy   production frontend with standalone Sam policy"
	@echo "  make preview-bgc-policy  production frontend with standalone pi1 artifact"
	@echo "  make preview-belief-greedy  belief-greedy 32-outer comparison"
	@echo "  make preview-pi0-bgc  BGC-128 with explicit pi0 continuations"
	@echo "  make preview-control  explicit archived-policy comparison"
	@echo "  make play-report      aggregate recorded SQLite game results"
	@echo
	@echo "Verification"
	@echo "  make test             complete Python and frontend checks"
	@echo "  make test-e2e         browser end-to-end checks"
	@echo
	@echo "Historical Sam-32 corpus"
	@echo "  make corpus-health    process and committed-corpus summary"
	@echo "  make corpus-inspect   inspect through the frozen source"
	@echo "  make corpus-verify    verify through the frozen source"
	@echo "  make corpus-stop      graceful SIGINT and wait"
	@echo "  make corpus-resume    foreground resume from frozen source"

check-api-port:
	@.venv/bin/python -c 'import socket, sys; sock = socket.socket(); result = sock.connect_ex(("127.0.0.1", $(API_PORT))); sock.close(); sys.exit(result == 0)' || (echo "API port $(API_PORT) is already in use" >&2; exit 1)

check-candidate:
	@test -f "$(CANDIDATE)" || (echo "Candidate archive not found: $(CANDIDATE)" >&2; exit 1)

check-opponent:
	@test "$(SEARCH_SIMULATIONS)" -gt 0 2>/dev/null || (echo "SEARCH_SIMULATIONS must be a positive integer" >&2; exit 1)
	@case "$(OPPONENT)" in \
		search) ;; \
		search-v2) case "$(SEARCH_RESPONSE_COMPLETIONS)" in 1|2|4) ;; *) echo "SEARCH_RESPONSE_COMPLETIONS must be 1, 2, or 4" >&2; exit 1 ;; esac ;; \
		search-v2-nested) test "$(SEARCH_RESPONSE_SIMULATIONS)" -gt 0 2>/dev/null || (echo "SEARCH_RESPONSE_SIMULATIONS must be a positive integer" >&2; exit 1) ;; \
		search-belief-greedy) test "$(BELIEF_COMPLETIONS)" -gt 0 2>/dev/null || (echo "BELIEF_COMPLETIONS must be a positive integer" >&2; exit 1) ;; \
		search-belief-greedy-pi0) test "$(BELIEF_COMPLETIONS)" -gt 0 2>/dev/null || (echo "BELIEF_COMPLETIONS must be a positive integer" >&2; exit 1); test -f "$(BGC_PI0_ARTIFACT)" || (echo "BGC pi0 artifact not found: $(BGC_PI0_ARTIFACT)" >&2; exit 1) ;; \
		bgc-policy) test -f "$(BGC_POLICY_ARTIFACT)" || (echo "BGC policy artifact not found: $(BGC_POLICY_ARTIFACT)" >&2; exit 1) ;; \
		search-v2-student-direct|search-v2-student-top-2|search-v2-student-top-3) \
			case "$(SEARCH_RESPONSE_COMPLETIONS)" in 1|2|4) ;; *) echo "SEARCH_RESPONSE_COMPLETIONS must be 1, 2, or 4" >&2; exit 1 ;; esac; \
			test -f "$(RESPONSE_RANKER_ARTIFACT)" || (echo "Response-ranker artifact not found: $(RESPONSE_RANKER_ARTIFACT)" >&2; exit 1) ;; \
		guided) test -f "$(GUIDED_ARTIFACT)" || (echo "Guided artifact not found: $(GUIDED_ARTIFACT)" >&2; exit 1) ;; \
		archive) test -f "$(CANDIDATE)" || (echo "Candidate archive not found: $(CANDIDATE)" >&2; exit 1) ;; \
		sam-policy) test -f "$(SAM_POLICY_ARTIFACT)" || (echo "Standalone Sam-policy artifact not found: $(SAM_POLICY_ARTIFACT)" >&2; exit 1) ;; \
		*) echo "OPPONENT must be bgc-policy, search, search-v2, search-v2-nested, search-belief-greedy, search-belief-greedy-pi0, a documented search-v2 student mode, guided, archive, or sam-policy" >&2; exit 1 ;; \
	esac

dev: check-opponent check-api-port
	@set -e; \
	DRACULA_DATABASE_PATH="$(DATABASE)" DRACULA_NARRATION_ENABLED=false DRACULA_OPPONENT_MODE="$(OPPONENT)" DRACULA_SEARCH_SIMULATIONS="$(SEARCH_SIMULATIONS)" DRACULA_SEARCH_EXPLORATION="$(SEARCH_EXPLORATION)" DRACULA_SEARCH_RESPONSE_COMPLETIONS="$(SEARCH_RESPONSE_COMPLETIONS)" DRACULA_SEARCH_RESPONSE_SIMULATIONS="$(SEARCH_RESPONSE_SIMULATIONS)" DRACULA_BELIEF_COMPLETIONS="$(BELIEF_COMPLETIONS)" DRACULA_BGC_PI0_ARTIFACT="$(BGC_POLICY_ARTIFACT)" DRACULA_RESPONSE_RANKER_ARTIFACT="$(RESPONSE_RANKER_ARTIFACT)" DRACULA_POLICY_VALUE_ARTIFACT="$(GUIDED_ARTIFACT)" DRACULA_GUIDED_SIMULATIONS="$(GUIDED_SIMULATIONS)" DRACULA_POLICY_ARCHIVE="$(CANDIDATE)" DRACULA_SAM_POLICY_ARTIFACT="$(SAM_POLICY_ARTIFACT)" DRACULA_POLICY_INFERENCE_PROFILE="$(INFERENCE_PROFILE)" .venv/bin/uvicorn dracula.api.app:app --reload --host 127.0.0.1 --port "$(API_PORT)" & \
	api_pid=$$!; \
	trap 'kill $$api_pid 2>/dev/null || true' EXIT INT TERM; \
	DRACULA_API_PROXY_TARGET="http://127.0.0.1:$(API_PORT)" npm --prefix frontend run dev -- --port "$(DEV_PORT)" --strictPort

preview: check-opponent check-api-port
	@set -e; \
	npm --prefix frontend run build; \
	DRACULA_DATABASE_PATH="$(DATABASE)" DRACULA_NARRATION_ENABLED=false DRACULA_OPPONENT_MODE="$(OPPONENT)" DRACULA_SEARCH_SIMULATIONS="$(SEARCH_SIMULATIONS)" DRACULA_SEARCH_EXPLORATION="$(SEARCH_EXPLORATION)" DRACULA_SEARCH_RESPONSE_COMPLETIONS="$(SEARCH_RESPONSE_COMPLETIONS)" DRACULA_SEARCH_RESPONSE_SIMULATIONS="$(SEARCH_RESPONSE_SIMULATIONS)" DRACULA_BELIEF_COMPLETIONS="$(BELIEF_COMPLETIONS)" DRACULA_BGC_PI0_ARTIFACT="$(BGC_POLICY_ARTIFACT)" DRACULA_RESPONSE_RANKER_ARTIFACT="$(RESPONSE_RANKER_ARTIFACT)" DRACULA_POLICY_VALUE_ARTIFACT="$(GUIDED_ARTIFACT)" DRACULA_GUIDED_SIMULATIONS="$(GUIDED_SIMULATIONS)" DRACULA_POLICY_ARCHIVE="$(CANDIDATE)" DRACULA_SAM_POLICY_ARTIFACT="$(SAM_POLICY_ARTIFACT)" DRACULA_POLICY_INFERENCE_PROFILE="$(INFERENCE_PROFILE)" .venv/bin/uvicorn dracula.api.app:app --host 127.0.0.1 --port "$(API_PORT)" & \
	api_pid=$$!; \
	trap 'kill $$api_pid 2>/dev/null || true' EXIT INT TERM; \
	DRACULA_API_PROXY_TARGET="http://127.0.0.1:$(API_PORT)" npm --prefix frontend run preview -- --port "$(PREVIEW_PORT)" --strictPort

play: preview

preview-policy:
	@$(MAKE) preview OPPONENT=sam-policy

preview-bgc-policy:
	@$(MAKE) preview OPPONENT=bgc-policy

preview-sam:
	@$(MAKE) preview OPPONENT=search-v2-nested SEARCH_SIMULATIONS=32 SEARCH_RESPONSE_SIMULATIONS=32

preview-control:
	@$(MAKE) preview OPPONENT=archive

preview-belief-greedy:
	@$(MAKE) preview OPPONENT=search-belief-greedy SEARCH_SIMULATIONS=32 BELIEF_COMPLETIONS=8

preview-pi0-bgc:
	@$(MAKE) preview OPPONENT=search-belief-greedy-pi0 SEARCH_SIMULATIONS=128 BELIEF_COMPLETIONS=8 BGC_POLICY_ARTIFACT="$(BGC_PI0_ARTIFACT)"

dev-api: check-opponent
	DRACULA_DATABASE_PATH="$(DATABASE)" DRACULA_NARRATION_ENABLED=false DRACULA_OPPONENT_MODE="$(OPPONENT)" DRACULA_SEARCH_SIMULATIONS="$(SEARCH_SIMULATIONS)" DRACULA_SEARCH_EXPLORATION="$(SEARCH_EXPLORATION)" DRACULA_SEARCH_RESPONSE_COMPLETIONS="$(SEARCH_RESPONSE_COMPLETIONS)" DRACULA_SEARCH_RESPONSE_SIMULATIONS="$(SEARCH_RESPONSE_SIMULATIONS)" DRACULA_BELIEF_COMPLETIONS="$(BELIEF_COMPLETIONS)" DRACULA_BGC_PI0_ARTIFACT="$(BGC_POLICY_ARTIFACT)" DRACULA_RESPONSE_RANKER_ARTIFACT="$(RESPONSE_RANKER_ARTIFACT)" DRACULA_POLICY_VALUE_ARTIFACT="$(GUIDED_ARTIFACT)" DRACULA_GUIDED_SIMULATIONS="$(GUIDED_SIMULATIONS)" DRACULA_POLICY_ARCHIVE="$(CANDIDATE)" DRACULA_SAM_POLICY_ARTIFACT="$(SAM_POLICY_ARTIFACT)" DRACULA_POLICY_INFERENCE_PROFILE="$(INFERENCE_PROFILE)" .venv/bin/uvicorn dracula.api.app:app --reload --host 127.0.0.1 --port "$(API_PORT)"

dev-frontend:
	DRACULA_API_PROXY_TARGET="http://127.0.0.1:$(API_PORT)" npm --prefix frontend run dev -- --port "$(DEV_PORT)" --strictPort

play-report:
	@test -f "$(DATABASE)" || (echo "Gameplay database not found: $(DATABASE)" >&2; exit 1)
	@sqlite3 -readonly "$(DATABASE)" < queries/local-game-report.sql

play-game:
	@test -f "$(DATABASE)" || (echo "Gameplay database not found: $(DATABASE)" >&2; exit 1)
	@test -n "$(GAME_ID)" || (echo "GAME_ID is required" >&2; exit 1)
	@sqlite3 -readonly -cmd ".parameter init" -cmd ".parameter set @game_id '$(GAME_ID)'" "$(DATABASE)" < queries/local-game-detail.sql

reset-local:
	rm -f "$(DATABASE)" "$(DATABASE)-shm" "$(DATABASE)-wal"

test-web:
	.venv/bin/python -m pytest tests/test_api.py tests/test_gameplay_api.py
	npm --prefix frontend run check

test-e2e: check-candidate
	DRACULA_POLICY_ARCHIVE="$(CANDIDATE)" npm --prefix frontend run test:e2e

test:
	.venv/bin/python -m pytest
	npm --prefix frontend run check

corpus-health:
	@test -f "$(SAM_PID_FILE)" || (echo "Corpus PID file not found: $(SAM_PID_FILE)" >&2; exit 1)
	@set -e; pid=$$(tr -d '[:space:]' < "$(SAM_PID_FILE)"); \
	test -n "$$pid" && kill -0 "$$pid" 2>/dev/null || (echo "Corpus process is not running" >&2; exit 1); \
	children=$$(pgrep -P "$$pid" | wc -l | tr -d '[:space:]'); \
	echo "parent_pid=$$pid child_processes=$$children"; \
	.venv/bin/python -c 'import json, pathlib; p = pathlib.Path("$(SAM_CORPUS)") / "corpus-manifest.json"; d = json.loads(p.read_text())["content"]; print("decks={} rows={} leaves={}".format(d["deck_count"], d["row_count"], d["terminal_leaf_count"]))'; \
	tail -n 5 "$(SAM_LOG)"

corpus-inspect:
	@cd "$(SAM_FROZEN_SOURCE)" && \
	PYTHONPATH="$(SAM_FROZEN_SOURCE)/src" "$(CURDIR)/.venv/bin/python" \
		-m dracula.sam_miner inspect --output "$(SAM_CORPUS)"

corpus-verify:
	@cd "$(SAM_FROZEN_SOURCE)" && \
	PYTHONPATH="$(SAM_FROZEN_SOURCE)/src" "$(CURDIR)/.venv/bin/python" \
		-m dracula.sam_miner verify --output "$(SAM_CORPUS)"

corpus-stop:
	@test -f "$(SAM_PID_FILE)" || (echo "Corpus PID file not found: $(SAM_PID_FILE)" >&2; exit 1)
	@pid=$$(tr -d '[:space:]' < "$(SAM_PID_FILE)"); \
	test -n "$$pid" && kill -0 "$$pid" 2>/dev/null || (echo "Corpus process is not running" >&2; exit 1); \
	kill -INT "$$pid"; \
	while kill -0 "$$pid" 2>/dev/null; do sleep 1; done; \
	echo "Corpus stopped cleanly."

corpus-resume:
	@test -d "$(SAM_FROZEN_SOURCE)/src" || (echo "Frozen collector source not found: $(SAM_FROZEN_SOURCE)" >&2; exit 1)
	@if test -f "$(SAM_PID_FILE)"; then \
		pid=$$(tr -d '[:space:]' < "$(SAM_PID_FILE)"); \
		if test -n "$$pid" && kill -0 "$$pid" 2>/dev/null; then \
			echo "Corpus process is already running: $$pid" >&2; exit 1; \
		fi; \
	fi
	@cd "$(SAM_FROZEN_SOURCE)"; \
	echo $$$$ > "$(SAM_PID_FILE)"; \
	exec env PYTHONPATH="$(SAM_FROZEN_SOURCE)/src" PYTHONUNBUFFERED=1 \
		"$(CURDIR)/.venv/bin/python" -m dracula.sam_miner resume \
		--output "$(SAM_CORPUS)" >> "$(SAM_LOG)" 2>&1
