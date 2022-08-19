#!/bin/bash

PLAN_REPO=${PLAN_REPO}

if [[ -z "$PLAN_REPO" ]]; then
	echo "PLAN_REPO environment variable not set, please set this variable"
	return
fi

script_path=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
exec_path="$script_path/../"

TARGET="plan.py"

models=("resnet50" "resnet101" "bert_base" "bert_large" "roberta_base" "roberta_large" "gpt2" "gpt2_medium")

if [ ! -d "$PLAN_REPO" ]; then
	mkdir -p "$PLAN_REPO"
	echo "Create $PLAN_REPO directory"
fi

for model in ${models[@]}; do
	cmd="python3 $exec_path/$TARGET -m $model -p $PLAN_REPO --trace --profile"
	$cmd
done
