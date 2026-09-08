FROM public.ecr.aws/awsguru/aws-lambda-adapter@sha256:1e5ab4d9242167500ed8a7bed8a79b448228aaa51cf382fb51fe4bf8a5f9a811 AS lambda-adapter

FROM python@sha256:48a11b7ba705fd53bf15248d1f94d36c39549903c5d59edcfa2f3f84126e7b44

# The adapter converts Lambda invocation events into ordinary HTTP requests for
# Uvicorn, allowing the FastAPI application to remain provider-neutral.
COPY --from=lambda-adapter /lambda-adapter /opt/extensions/lambda-adapter

# PyTorch inference is deliberately single-threaded inside each Lambda process;
# Lambda concurrency supplies request-level parallelism without CPU oversubscription.
ENV AWS_LWA_PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/Dracula/api/health \
    AWS_LWA_READINESS_CHECK_HEALTHY_STATUS=200-399 \
    DRACULA_POLICY_ARTIFACT=/opt/dracula/artifacts/policy.pt \
    DRACULA_FRONTEND_DIR=/opt/dracula/frontend \
    DRACULA_NARRATION_ENABLED=false \
    DRACULA_REPLAY_CACHE_ENTRIES=256 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PORT=8080 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/var/task/src \
    PYTHONUNBUFFERED=1

WORKDIR /var/task

COPY requirements-constraints.txt requirements-lambda.txt requirements-torch.txt ./
# Install the CPU-only PyTorch wheel separately because it uses PyTorch's package
# index. Both installations remain pinned by the generated constraint file.
RUN python -m pip install --disable-pip-version-check --no-cache-dir \
    --constraint requirements-constraints.txt \
    --requirement requirements-torch.txt && \
    python -m pip install --disable-pip-version-check --no-cache-dir \
    --constraint requirements-constraints.txt \
    --requirement requirements-lambda.txt

COPY src ./src
COPY frontend /opt/dracula/frontend

COPY logging.json /opt/dracula/logging.json
# The build-context tool verifies and copies the ignored selected artifact before
# Docker sees it. Neither the original training run nor other artifacts enter the image.
COPY artifacts/policy.pt /opt/dracula/artifacts/policy.pt

CMD ["uvicorn", "dracula.api.production:app", "--host", "0.0.0.0", "--port", "8080", "--log-config", "/opt/dracula/logging.json"]
