#!/bin/bash

PLAN_REPO=${PLAN_REPO}

if [[ -z "$PLAN_REPO" ]]; then
	echo "PLAN_REPO environment variable not set, please set this variable"
	return
fi

export PLAN_REPO=${PLAN_REPO}

ENGINE="pipeline"

script_path=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
build_path="$script_path/../../build"

TARGET="cache_study"

models=("resnet50" "bert_base" "roberta_large")
batch_sizes=(1)



for model in "${models[@]}"; do
	tmp_file="/tmp/cache_study_$model.out"
	printf "" > $tmp_file

	for batch_size in "${batch_sizes[@]}"; do
		# Baseline
		cmd="$build_path/$TARGET -m $model -b $batch_size -e $ENGINE -v"
		echo "Run $cmd"

		output=`$cmd`
		echo "$output"
		echo ""

		n_layers=$(echo "$output" | awk '{if ($2 == "Number") { print $(NF)}}')
		model_size=$(echo "$output" | awk '{if ($2 == "Model") { print $(NF-1)}}')
		in_memory_lat=$(echo "$output" | awk '{if ($1 == "In-Memory") { print $(NF-1)}}')
		opt_n_caches=$(echo "$output" | awk '{if ($3 == "Number") { print $(NF)}}')
		opt_load_size=$(echo "$output" | awk '{if ($3 == "Load") { print $(NF-1)}}')
		opt_lat=$(echo "$output" | awk '{if ($3 == "Inference") { print $(NF-1)}}')

		printf "$batch_size, " >> $tmp_file
		printf "$n_layers, " >> $tmp_file
		printf "$model_size, " >> $tmp_file
		printf "$in_memory_lat, " >> $tmp_file
		printf "$opt_n_caches, " >> $tmp_file
		printf "$opt_load_size, " >> $tmp_file
		printf "$opt_lat, " >> $tmp_file

		echo "" >> $tmp_file
	done
done

log_path="$script_path/logs"

# Check for log_path existence
if [ ! -d "$log_path" ]; then
	echo "Created $log_path directory where log files will be stored"
	mkdir -p $log_path
fi

date=`date +%y-%m-%d`

log_path="$log_path/$date"
if [ ! -d "$log_path" ]; then
	mkdir -p "$log_path"
fi

output_path="$log_path/report"

version=0
while true; do
	_output="${output_path}$version"
	if [ -d "$_output" ]; then
		((version++))
	else
		break
	fi
done

output_path="$_output"

mkdir -p $output_path

for model in "${models[@]}"; do
	tmp_file="/tmp/cache_study_$model.out"
	cp $tmp_file $output_path
done

echo "Created log files in '$output_path'"
