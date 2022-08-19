#!/bin/bash

export MODEL_REPO=${MODEL_REPO}

script_path="$( cd "$(dirname "$0")" ; pwd -P )"
build_path="$script_path/../../build"

TARGET="benchmark"

device_maps=("0" "0 2")
models=("resnet50" "resnet101" "bert_base" "bert_large" "roberta_base" "roberta_large" "gpt2" "gpt2_medium")
engines=("pipeline" "deepplan")
batch_size=1

tmp_file="/tmp/deepplan_fig10"
echo "" > $tmp_file

for model in "${models[@]}"; do
	# Baseline
	baseline_cmd="$build_path/$TARGET -m $model -e demand -b $batch_size -d 0"
	echo "Run $baseline_cmd"

	output=`$baseline_cmd`
	echo "$output"
	echo ""

	latency=$(echo "$output" | awk '{if ($2 == "Latency") { print $(NF-1)}}')
	printf "$latency, " >> $tmp_file

	for device_map in "${device_maps[@]}"; do
		for engine in "${engines[@]}"; do
			cmd="$build_path/$TARGET -m $model -e $engine -b $batch_size -d $device_map"
			echo "Run $cmd"

			output=`$cmd`
			echo "$output"
			echo ""

			latency=$(echo "$output" | awk '{if ($2 == "Latency") { print $(NF-1)}}')
			printf "$latency, " >> $tmp_file
		done
	done

	echo "" >> $tmp_file
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

output_file="$log_path/report"

version=0
while true; do
	_output="${output_file}$version.csv"
	if [ -f "$_output" ]; then
		((version++))
	else
		break
	fi
done

output_file="$_output"

cp $tmp_file $output_file
echo "Created '$output_file' log file"

eval "python graph.py $output_file figure.pdf"
