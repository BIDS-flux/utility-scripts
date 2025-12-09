#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$PWD"
INPUT="/data/cpip_workflow.mmd"
PNG_OUT="/data/diagram@6x.png"
SVG_OUT="/data/diagram.svg"

sudo docker run --rm \
	-u "$(id -u)":"$(id -g)" \
	-v "$WORKDIR":/data \
	minlag/mermaid-cli:latest \
	-i "$INPUT" \
	-o "$PNG_OUT" \
	-b transparent \
	--scale 12

sudo docker run --rm \
	-u "$(id -u)":"$(id -g)" \
	-v "$WORKDIR":/data \
	minlag/mermaid-cli:latest \
	-i "$INPUT" \
	-o "$SVG_OUT"