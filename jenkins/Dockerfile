# Jenkins controller with the tools the MedCalib pipeline needs:
# Python (unit tests), Docker CLI (image build) and kubectl (deployment).
FROM jenkins/jenkins:2.580.2-lts-jdk21

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip curl \
 && rm -rf /var/lib/apt/lists/*

# Docker CLI only; the daemon is the host's Docker Desktop (socket is mounted).
COPY --from=docker:27.4-cli /usr/local/bin/docker /usr/local/bin/docker

ARG KUBECTL_VERSION=v1.36.1
RUN curl -fsSLo /usr/local/bin/kubectl \
      "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
 && chmod +x /usr/local/bin/kubectl

USER jenkins
COPY --chown=jenkins:jenkins plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt
# Outside JENKINS_HOME (a volume), so edits to casc.yaml apply on rebuild.
COPY --chown=jenkins:jenkins casc.yaml /var/jenkins_conf/casc.yaml

ENV JAVA_OPTS="-Djenkins.install.runSetupWizard=false" \
    CASC_JENKINS_CONFIG=/var/jenkins_conf/casc.yaml
