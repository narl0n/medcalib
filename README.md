# medcalib-api

A small research service from the MedCalib project: a **calibrated** risk model for
early Alzheimer's disease exposed as a REST API, plus a complete CI/CD pipeline
(Docker + Jenkins + Kubernetes). Built for Assignment 5 (DevOps and CI/CD),
Applied Software Development Project, Astana IT University.

> Research prototype trained on a seeded **synthetic** cohort. Not for clinical use.

## What it does

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness / readiness check |
| `GET /model` | Features and the training report (AUC, ECE and Brier before/after calibration) |
| `POST /predict` | Calibrated probability of early AD / MCI conversion and a risk band |
| `GET /docs` | Interactive OpenAPI documentation |

Model: Gaussian naive Bayes on five features (age, MMSE, hippocampal volume,
entorhinal thickness, APOE-ε4), calibrated with isotonic regression
(`CalibratedClassifierCV`, 5-fold). With seed 42 the hold-out ECE falls from
0.121 to 0.052 and the Brier score from 0.156 to 0.139 (AUC 0.882). The model is
trained **at image build time**, so every container serves the same artefact.

## Run locally

```bash
pip install -r requirements-dev.txt
pytest
uvicorn medcalib_api.app:app --app-dir src --reload
```

## Docker

```bash
docker build -t medcalib-api .
docker run --rm -p 8000:8000 medcalib-api
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":74,"mmse":24,"hippocampal_volume":6.5,"entorhinal_thickness":3.1,"apoe4_carrier":true}'
```

## CI/CD with Jenkins

Jenkins runs locally in Docker and is configured entirely as code
([jenkins/casc.yaml](jenkins/casc.yaml)): admin user, security and the
`medcalib-api` pipeline job are created automatically.

```bash
cd jenkins
cp .env.example .env        # set the admin password and your repository URL
docker compose up -d --build
# open http://localhost:8080
```

Pipeline ([Jenkinsfile](Jenkinsfile)), triggered by every commit (SCM polling every 2 min):

1. **Checkout**
2. **Set up Python**: virtual environment with pinned dependencies
3. **Lint**: Ruff
4. **Unit tests**: pytest with JUnit report, coverage ≥ 90 %, and model quality
   gates (AUC ≥ 0.85, calibrated ECE ≤ 0.07 and lower than uncalibrated)
5. **Build Docker image**: multi-stage build, tagged `<build>-<commit>`
6. **Smoke test container**: starts the image, calls `/health` and `/predict`
7. **Push to Docker Hub** (optional): runs when `DOCKERHUB_REPO` is set and a
   `dockerhub-creds` credential exists
8. **Deploy to Kubernetes**: `kubectl apply` + rolling update on Docker Desktop Kubernetes
9. **Verify deployment**: rollout status and an in-cluster health check

After a successful run the API is served by Kubernetes at <http://localhost:30080/docs>.

## Layout

```
src/medcalib_api/   data.py (synthetic cohort), metrics.py (ECE, Brier),
                    model.py (train/calibrate/save), app.py (FastAPI)
tests/              unit, model-quality and API tests
Dockerfile          multi-stage image, non-root user, HEALTHCHECK
Jenkinsfile         CI/CD pipeline
jenkins/            Jenkins image, plugins, JCasC config, docker-compose
k8s/                Deployment (2 replicas, probes, limits) and NodePort Service
```

## License

MIT
