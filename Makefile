POLICY_ARTIFACT ?= $(CURDIR)/runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt
API_PORT ?= 8000
DEV_PORT ?= 5173
PREVIEW_PORT ?= 4173
STAGING_API_ORIGIN ?=
API_APP ?= dracula.api.development:app
LAMBDA_CONTEXT ?= $(CURDIR)/build/lambda-context
LAMBDA_IMAGE ?= dracula-api:local
LAMBDA_PORT ?= 8080
WEB_API_ORIGIN ?= /Dracula/api
WEB_BASE_PATH ?= /Dracula/
BEDROCK_REGION ?= us-east-1
BEDROCK_MODEL_ID ?= amazon.nova-lite-v1:0
NARRATION_ENABLED ?= false

LOCAL_GAMEPLAY_ENV = \
	DRACULA_NARRATION_ENABLED="$(NARRATION_ENABLED)" \
	DRACULA_BEDROCK_REGION="$(BEDROCK_REGION)" \
	DRACULA_BEDROCK_MODEL_ID="$(BEDROCK_MODEL_ID)" \
	DRACULA_POLICY_ARTIFACT="$(POLICY_ARTIFACT)"

.DEFAULT_GOAL := help

.PHONY: help check-api-port check-policy dev preview preview-dialogue preview-bedrock preview-app preview-staging dev-api dev-frontend test test-web test-e2e web-build web-test lambda-context lambda-build lambda-run lambda-validate release-candidate

help:
	@echo "Local gameplay"
	@echo "  make preview-app      combined FastAPI frontend/API preview on port $(PREVIEW_PORT)"
	@echo "  make dev              FastAPI and Vite development servers"
	@echo "  make preview          production frontend with standalone pi1"
	@echo "  make preview-dialogue standalone pi1 with deterministic test dialogue"
	@echo "  make preview-bedrock  standalone pi1 with live Bedrock dialogue"
	@echo "  make preview-staging STAGING_API_ORIGIN=https://...  local frontend, hosted gameplay"
	@echo
	@echo "Verification and release"
	@echo "  make test             complete Python and frontend checks"
	@echo "  make test-e2e         browser end-to-end checks"
	@echo "  make lambda-build     verified pi1 Lambda container image"
	@echo "  make lambda-run       read-only local production container"
	@echo "  make lambda-validate  full local container validation"
	@echo "  make web-build        verified /Dracula/ production frontend"
	@echo "  make web-test         stateless six-round browser validation"
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

preview-app: check-policy web-build
	$(LOCAL_GAMEPLAY_ENV) DRACULA_FRONTEND_DIR="$(CURDIR)/frontend/dist" .venv/bin/uvicorn dracula.api.production:app --host 127.0.0.1 --port "$(PREVIEW_PORT)"

# Keep this testing build separate from the frontend copied into the Lambda image.
# Vite proxies /api to AWS; no local Python server or permissive CORS is needed.
preview-staging:
	@test -n "$(STAGING_API_ORIGIN)" || (echo "Set STAGING_API_ORIGIN to the staging HTTPS origin" >&2; exit 1)
	VITE_API_ORIGIN=/api VITE_BASE_PATH=/Dracula/ npm --prefix frontend run build -- --outDir ../build/staging-frontend --emptyOutDir
	VITE_API_ORIGIN=/api VITE_BASE_PATH=/Dracula/ DRACULA_API_PROXY_TARGET="$(STAGING_API_ORIGIN)/Dracula/api" npm --prefix frontend run preview -- --outDir ../build/staging-frontend --port "$(PREVIEW_PORT)" --strictPort

preview-dialogue:
	@$(MAKE) preview API_APP=dracula.api.preview:app

preview-bedrock:
	@$(MAKE) preview NARRATION_ENABLED=true

dev-api: check-policy
	$(LOCAL_GAMEPLAY_ENV) .venv/bin/uvicorn "$(API_APP)" --reload --host 127.0.0.1 --port "$(API_PORT)"

dev-frontend:
	DRACULA_API_PROXY_TARGET="http://127.0.0.1:$(API_PORT)" npm --prefix frontend run dev -- --port "$(DEV_PORT)" --strictPort

test-web:
	.venv/bin/python -m pytest tests/test_api.py tests/test_stateless_gameplay_api.py
	npm --prefix frontend run check

test-e2e:
	DRACULA_POLICY_ARTIFACT="$(POLICY_ARTIFACT)" npm --prefix frontend run test:e2e

web-build:
	VITE_API_ORIGIN="$(WEB_API_ORIGIN)" VITE_BASE_PATH="$(WEB_BASE_PATH)" \
		npm --prefix frontend run verify:build

web-test: test-e2e

test:
	.venv/bin/python -m pytest
	npm --prefix frontend run check

lambda-context: web-build
	.venv/bin/python tools/build_lambda_context.py --output "$(LAMBDA_CONTEXT)"

lambda-build: lambda-context
	docker build --platform linux/arm64 --tag "$(LAMBDA_IMAGE)" "$(LAMBDA_CONTEXT)"

lambda-run:
	docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=512m \
		--publish "127.0.0.1:$(LAMBDA_PORT):8080" "$(LAMBDA_IMAGE)"

lambda-validate:
	.venv/bin/python tools/validate_lambda_container.py \
		--image "$(LAMBDA_IMAGE)" --port "$(LAMBDA_PORT)"

release-candidate: lambda-build
	.venv/bin/python tools/build_release_candidate.py \
		--image "$(LAMBDA_IMAGE)" \
		--output build/release-candidate/manifest.json
