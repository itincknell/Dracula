FROM public.ecr.aws/awsguru/aws-lambda-adapter@sha256:1e5ab4d9242167500ed8a7bed8a79b448228aaa51cf382fb51fe4bf8a5f9a811 AS lambda-adapter

FROM python@sha256:48a11b7ba705fd53bf15248d1f94d36c39549903c5d59edcfa2f3f84126e7b44

COPY --from=lambda-adapter /lambda-adapter /opt/extensions/lambda-adapter

ENV AWS_LWA_PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health \
    AWS_LWA_READINESS_CHECK_HEALTHY_STATUS=200-399 \
    DRACULA_POLICY_ARTIFACT=/opt/dracula/artifacts/pi1.pt \
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
RUN python -m pip install --disable-pip-version-check --no-cache-dir \
    --constraint requirements-constraints.txt \
    --requirement requirements-torch.txt && \
    python -m pip install --disable-pip-version-check --no-cache-dir \
    --constraint requirements-constraints.txt \
    --requirement requirements-lambda.txt

COPY pyproject.toml ./pyproject.toml
COPY src ./src

COPY logging.json /opt/dracula/logging.json
COPY artifacts/pi1.pt /opt/dracula/artifacts/pi1.pt
COPY release-manifest.json /opt/dracula/release-manifest.json

CMD ["uvicorn", "dracula.api.production:app", "--host", "0.0.0.0", "--port", "8080", "--log-config", "/opt/dracula/logging.json"]
