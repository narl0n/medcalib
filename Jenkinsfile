// CI/CD pipeline for the MedCalib calibrated risk API.
// CI: lint -> unit tests + model quality gates -> Docker build -> container smoke test
// CD: optional push to Docker Hub -> rolling deploy to Kubernetes -> in-cluster check
pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timeout(time: 30, unit: 'MINUTES')
    }

    // Every commit pushed to GitHub is picked up within ~2 minutes.
    // (A webhook cannot reach a Jenkins running on localhost, so SCM polling is used.)
    triggers {
        pollSCM('H/2 * * * *')
    }

    environment {
        IMAGE     = 'medcalib-api'
        IMAGE_TAG = "${env.BUILD_NUMBER}-${env.GIT_COMMIT?.take(7) ?: 'local'}"
        SMOKE     = "medcalib-smoke-${env.BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git log -1 --pretty="%h %an: %s"'
            }
        }

        stage('Set up Python') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --quiet --upgrade pip
                    pip install --quiet -r requirements-dev.txt
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '. .venv/bin/activate && ruff check src tests'
            }
        }

        stage('Unit tests') {
            steps {
                sh '''
                    . .venv/bin/activate
                    mkdir -p reports
                    pytest --junitxml=reports/junit.xml \
                           --cov=medcalib_api --cov-report=xml:reports/coverage.xml \
                           --cov-report=term --cov-fail-under=90
                '''
            }
            post {
                always {
                    junit 'reports/junit.xml'
                    archiveArtifacts artifacts: 'reports/*.xml', allowEmptyArchive: true
                }
            }
        }

        stage('Build Docker image') {
            steps {
                sh '''
                    docker build \
                      --label org.opencontainers.image.revision=${GIT_COMMIT} \
                      -t ${IMAGE}:${IMAGE_TAG} -t ${IMAGE}:latest .
                    docker image inspect ${IMAGE}:${IMAGE_TAG} --format "Image size: {{.Size}} bytes"
                '''
            }
        }

        stage('Smoke test container') {
            steps {
                sh '''
                    docker run -d --rm --name ${SMOKE} --network medcalib-ci ${IMAGE}:${IMAGE_TAG}
                    for i in $(seq 1 30); do
                      if curl -fsS http://${SMOKE}:8000/health; then break; fi
                      sleep 1
                    done
                    echo
                    curl -fsS -X POST http://${SMOKE}:8000/predict \
                      -H "Content-Type: application/json" \
                      -d '{"age":74,"mmse":24,"hippocampal_volume":6.5,"entorhinal_thickness":3.1,"apoe4_carrier":true}' \
                      | tee reports/smoke-predict.json
                    echo
                    curl -fsS http://${SMOKE}:8000/model > reports/model-report.json
                '''
            }
            post {
                always {
                    sh 'docker rm -f ${SMOKE} || true'
                    archiveArtifacts artifacts: 'reports/*.json', allowEmptyArchive: true
                }
            }
        }

        stage('Push to Docker Hub') {
            when { expression { return env.DOCKERHUB_REPO?.trim() } }
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds',
                                                  usernameVariable: 'DH_USER',
                                                  passwordVariable: 'DH_TOKEN')]) {
                    sh '''
                        echo "$DH_TOKEN" | docker login -u "$DH_USER" --password-stdin
                        docker tag ${IMAGE}:${IMAGE_TAG} ${DOCKERHUB_REPO}:${IMAGE_TAG}
                        docker tag ${IMAGE}:${IMAGE_TAG} ${DOCKERHUB_REPO}:latest
                        docker push ${DOCKERHUB_REPO}:${IMAGE_TAG}
                        docker push ${DOCKERHUB_REPO}:latest
                        docker logout
                    '''
                }
            }
        }

        stage('Deploy to Kubernetes') {
            environment { KUBECONFIG = "${env.WORKSPACE}/.kubeconfig" }
            steps {
                sh '''
                    # Docker Desktop's kubeconfig points to 127.0.0.1, which inside the
                    # Jenkins container is the container itself; use the host alias.
                    sed -E 's#https://(127\\.0\\.0\\.1|localhost):6443#https://kubernetes.docker.internal:6443#' \
                        /kube/config > "$KUBECONFIG"
                    kubectl config use-context docker-desktop
                    kubectl apply -f k8s/
                    kubectl set image deployment/medcalib-api api=${IMAGE}:${IMAGE_TAG}
                    kubectl annotate deployment/medcalib-api \
                        kubernetes.io/change-cause="Jenkins build ${BUILD_NUMBER} (${IMAGE_TAG})" --overwrite
                    kubectl rollout status deployment/medcalib-api --timeout=180s
                '''
            }
        }

        stage('Verify deployment') {
            environment { KUBECONFIG = "${env.WORKSPACE}/.kubeconfig" }
            steps {
                sh '''
                    kubectl get deployment,pods,service -l app=medcalib-api -o wide || true
                    kubectl get service medcalib-api
                    kubectl exec deploy/medcalib-api -- python -c \
                      "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
                    kubectl rollout history deployment/medcalib-api
                '''
            }
        }
    }

    post {
        success { echo "Deployed ${IMAGE}:${IMAGE_TAG} - API at http://localhost:30080/docs" }
        failure { echo 'Pipeline failed - see the stage log above.' }
    }
}
