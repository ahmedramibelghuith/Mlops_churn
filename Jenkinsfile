// Source definition only: requires a configured Linux agent and Jenkins credentials.
// No deployment is performed by this pipeline.
pipeline {
    agent any
    parameters {
        string(name: 'MIN_F1', defaultValue: '0.80', description: 'Required macro F1 (choose for your use case)')
    }
    environment {
        MLFLOW_TRACKING_URI = 'https://dagshub.com/rami4real/mymlproject.mlflow'
        MLFLOW_TRACKING_USERNAME = 'rami4real'
        MLFLOW_EXPERIMENT_NAME = 'churn-experiment'
        MLFLOW_TRACKING_PASSWORD = credentials('dagshub-token')
    }
    stages {
        stage('Checkout') {
            steps { checkout scm }
        }
        stage('Setup on CI agent') {
            steps {
                sh 'python3 -m venv .venv'
                sh '.venv/bin/python -m pip install -r requirements-dev.txt "dvc[s3]"'
            }
        }
        stage('Test') {
            steps { sh '.venv/bin/python -m pytest -q' }
        }
        stage('Retrieve data') {
            steps {
                sh '''#!/bin/sh
                    set -eu
                    export AWS_ACCESS_KEY_ID="$MLFLOW_TRACKING_USERNAME"
                    export AWS_SECRET_ACCESS_KEY="$MLFLOW_TRACKING_PASSWORD"
                    .venv/bin/dvc pull
                '''
            }
        }
        stage('Train') {
            steps { sh '.venv/bin/python -m backend.src.modeling' }
        }
        stage('Register candidate') {
            steps {
                withEnv(["MIN_F1=${params.MIN_F1}"]) {
                    sh '.venv/bin/python -m scripts.register_best_model --min-f1 "$MIN_F1"'
                }
            }
        }
    }
    post {
        success { archiveArtifacts artifacts: 'artifacts/production_model.json', fingerprint: true }
    }
}
