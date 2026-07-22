CANDIDATE ?= $(CURDIR)/runs/training-004/archives/policy-2-policy-2-v20.pt
DATABASE ?= $(CURDIR)/.local/dracula.sqlite3
OPPONENT ?= search
SEARCH_SIMULATIONS ?= 500
SEARCH_EXPLORATION ?= 1.4142135623730951
INFERENCE_PROFILE ?= argmax-v1
API_PORT ?= 8000
DEV_PORT ?= 5173
PREVIEW_PORT ?= 4173

.PHONY: check-api-port check-candidate check-opponent dev preview preview-control dev-api dev-frontend play-report play-game reset-local test-web test-e2e

check-api-port:
	@.venv/bin/python -c 'import socket, sys; sock = socket.socket(); result = sock.connect_ex(("127.0.0.1", $(API_PORT))); sock.close(); sys.exit(result == 0)' || (echo "API port $(API_PORT) is already in use" >&2; exit 1)

check-candidate:
	@test -f "$(CANDIDATE)" || (echo "Candidate archive not found: $(CANDIDATE)" >&2; exit 1)

check-opponent:
	@case "$(OPPONENT)" in \
		search) ;; \
		archive) test -f "$(CANDIDATE)" || (echo "Candidate archive not found: $(CANDIDATE)" >&2; exit 1) ;; \
		*) echo "OPPONENT must be search or archive" >&2; exit 1 ;; \
	esac

dev: check-opponent check-api-port
	@set -e; \
	DRACULA_DATABASE_PATH="$(DATABASE)" DRACULA_NARRATION_ENABLED=false DRACULA_OPPONENT_MODE="$(OPPONENT)" DRACULA_SEARCH_SIMULATIONS="$(SEARCH_SIMULATIONS)" DRACULA_SEARCH_EXPLORATION="$(SEARCH_EXPLORATION)" DRACULA_POLICY_ARCHIVE="$(CANDIDATE)" DRACULA_POLICY_INFERENCE_PROFILE="$(INFERENCE_PROFILE)" .venv/bin/uvicorn dracula.api.app:app --reload --host 127.0.0.1 --port "$(API_PORT)" & \
	api_pid=$$!; \
	trap 'kill $$api_pid 2>/dev/null || true' EXIT INT TERM; \
	DRACULA_API_PROXY_TARGET="http://127.0.0.1:$(API_PORT)" npm --prefix frontend run dev -- --port "$(DEV_PORT)" --strictPort

preview: check-opponent check-api-port
	@set -e; \
	npm --prefix frontend run build; \
	DRACULA_DATABASE_PATH="$(DATABASE)" DRACULA_NARRATION_ENABLED=false DRACULA_OPPONENT_MODE="$(OPPONENT)" DRACULA_SEARCH_SIMULATIONS="$(SEARCH_SIMULATIONS)" DRACULA_SEARCH_EXPLORATION="$(SEARCH_EXPLORATION)" DRACULA_POLICY_ARCHIVE="$(CANDIDATE)" DRACULA_POLICY_INFERENCE_PROFILE="$(INFERENCE_PROFILE)" .venv/bin/uvicorn dracula.api.app:app --host 127.0.0.1 --port "$(API_PORT)" & \
	api_pid=$$!; \
	trap 'kill $$api_pid 2>/dev/null || true' EXIT INT TERM; \
	DRACULA_API_PROXY_TARGET="http://127.0.0.1:$(API_PORT)" npm --prefix frontend run preview -- --port "$(PREVIEW_PORT)" --strictPort

preview-control:
	@$(MAKE) preview OPPONENT=archive

dev-api: check-opponent
	DRACULA_DATABASE_PATH="$(DATABASE)" DRACULA_NARRATION_ENABLED=false DRACULA_OPPONENT_MODE="$(OPPONENT)" DRACULA_SEARCH_SIMULATIONS="$(SEARCH_SIMULATIONS)" DRACULA_SEARCH_EXPLORATION="$(SEARCH_EXPLORATION)" DRACULA_POLICY_ARCHIVE="$(CANDIDATE)" DRACULA_POLICY_INFERENCE_PROFILE="$(INFERENCE_PROFILE)" .venv/bin/uvicorn dracula.api.app:app --reload --host 127.0.0.1 --port "$(API_PORT)"

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
