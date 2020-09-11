#!/bin/bash

# docker tag elbotto eu.gcr.io/modular-source-288719/cnnbot
# docker push eu.gcr.io/modular-source-288719/cnnbot

gcloud container clusters create jass-cluster --num-nodes=2
gcloud container clusters get-credentials jass-cluster

kubectl apply -f gcp/jasschallenge.yaml
kubectl apply -f gcp/challengeservice.yaml

kubectl apply -f gcp/cnnbot1.yaml
kubectl apply -f gcp/cnnbot2.yaml
kubectl apply -f gcp/relubot1.yaml
kubectl apply -f gcp/relubot2.yaml

kubectl apply -f gcp/cnnbot1tb.yaml
kubectl apply -f gcp/cnnbot2tb.yaml
kubectl apply -f gcp/relubot1tb.yaml
kubectl apply -f gcp/relubot2tb.yaml

