POLICY_ARTIFACT ?= $(CURDIR)/runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt
DATABASE ?= $(CURDIR)/.local/dracula.sqlite3
API_PORT ?= 8000
DEV_PORT ?= 5173
PREVIEW_PORT ?= 4173
API_APP ?= dracula.api.app:app
LAMBDA_CONTEXT ?= $(CURDIR)/build/lambda-context
LAMBDA_IMAGE ?= dracula-api:local
LAMBDA_PORT ?= 8080
PAGES_API_ORIGIN ?= https://api.ian-tincknell.com
PAGES_BASE_PATH ?= /Dracula/

LOCAL_GAMEPLAY_ENV = \
	DRACULA_GAMEPLAY_MODE=stateless \
	DRACULA_NARRATION_ENABLED=false \
	DRACULA_OPPONENT_MODE=pi1 \
	DRACULA_POLICY_ARTIFACT="$(POLICY_ARTIFACT)"

.DEFAULT_GOAL := help

.PHONY: help check-api-port check-policy dev preview preview-dialogue play dev-api dev-frontend play-report play-game reset-local test test-web test-e2e pages-build pages-test lambda-context lambda-build lambda-package lambda-run lambda-validate release-candidate

help:
	@echo "Local gameplay"
	@echo "  make dev              FastAPI and Vite development servers"
	@echo "  make preview          production frontend with standalone pi1"
	@echo "  make preview-dialogue standalone pi1 with deterministic test dialogue"
	@echo "  make play-report      aggregate recorded SQLite game results"
	@echo
	@echo "Verification and release"
	@echo "  make test             complete Python and frontend checks"
	@echo "  make test-e2e         browser end-to-end checks"
	@echo "  make lambda-build     verified pi1 Lambda container image"
	@echo "  make lambda-run       read-only local production container"
	@echo "  make lambda-validate  full local container validation"
	@echo "  make pages-build      verified /Dracula/ production frontend"
	@echo "  make pages-test       stateless six-round browser validation"
	@echo "  make release-candidate  seal local production candidate manifest"

check-api-port:
	@.venv/bin/python -c 'import socket, sys; sock = socket.socket(); result = sock.connect_ex(("127.0.0.1", $(API_PORT))); sock.close(); sys.exit(result == 0)' || (echo "API port $(API_PORT) is already in use" >&2; exit 1)

check-policy:
	@test -f "$(POLICY_ARTIFACT)" || (echo "Policy artifact not found: $(POLICY_ARTIFACT)" >&2; exit 1)

dev: check-policy check-api-port
	@set -e; \
	$(LOCAL_GAMEPLAY_ENV) .venv/bin/uvicorn "$(API_APP)" --reload --host 127.0.0.1 --port "$(API_PORT)" & \
	api_pid=$$!; \
	trap 'kill $$api_pid 2>/dev/null || true' EXIT INT TERM; \
	DRACULA_API_PROXY_TARGET="http://127.0.0.1:$(API_PORT)" npm --prefix frontend run dev -- --port "$(DEV_PORT)" --strictPort

preview: check-policy check-api-port
	@set -e; \
	VITE_API_ORIGIN=/api VITE_BASE_PATH=/ npm --prefix frontend run build; \
	$(LOCAL_GAMEPLAY_ENV) .venv/bin/uvicorn "$(API_APP)" --host 127.0.0.1 --port "$(API_PORT)" & \
	api_pid=$$!; \
	trap 'kill $$api_pid 2>/dev/null || true' EXIT INT TERM; \
	VITE_API_ORIGIN=/api VITE_BASE_PATH=/ DRACULA_API_PROXY_TARGET="http://127.0.0.1:$(API_PORT)" npm --prefix frontend run preview -- --port "$(PREVIEW_PORT)" --strictPort

play: preview

preview-dialogue:
	@$(MAKE) preview API_APP=dracula.api.local_preview:app

dev-api: check-policy
	$(LOCAL_GAMEPLAY_ENV) .venv/bin/uvicorn "$(API_APP)" --reload --host 127.0.0.1 --port "$(API_PORT)"

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
	.venv/bin/python -m pytest tests/test_api.py tests/test_gameplay_api.py tests/test_stateless_gameplay_api.py
	npm --prefix frontend run check

test-e2e:
	DRACULA_POLICY_ARTIFACT="$(POLICY_ARTIFACT)" npm --prefix frontend run test:e2e

pages-build:
	VITE_API_ORIGIN="$(PAGES_API_ORIGIN)" VITE_BASE_PATH="$(PAGES_BASE_PATH)" \
		npm --prefix frontend run verify:build

pages-test:
	DRACULA_POLICY_ARTIFACT="$(POLICY_ARTIFACT)" npm --prefix frontend run test:e2e

test:
	.venv/bin/python -m pytest
	npm --prefix frontend run check

lambda-context:
	.venv/bin/python tools/build_lambda_context.py --output "$(LAMBDA_CONTEXT)"

lambda-build: lambda-context
	docker build --platform linux/arm64 --tag "$(LAMBDA_IMAGE)" "$(LAMBDA_CONTEXT)"

lambda-package: lambda-build
	@docker image inspect "$(LAMBDA_IMAGE)" --format 'image={{.Id}} size={{.Size}}'

lambda-run:
	docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=512m \
		--publish "127.0.0.1:$(LAMBDA_PORT):8080" "$(LAMBDA_IMAGE)"

lambda-validate:
	.venv/bin/python tools/validate_lambda_container.py \
		--image "$(LAMBDA_IMAGE)" --port "$(LAMBDA_PORT)"

release-candidate: lambda-build pages-build
	.venv/bin/python tools/build_release_candidate.py \
		--image "$(LAMBDA_IMAGE)" \
		--output build/release-candidate/manifest.json
