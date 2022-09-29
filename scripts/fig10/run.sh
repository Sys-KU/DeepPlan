#!/bin/bash

PLAN_REPO=${PLAN_REPO}

if [[ -z "$PLAN_REPO" ]]; then
	echo "PLAN_REPO environment variable not set, please set this variable"
	return
fi

export PLAN_REPO=${PLAN_REPO}

script_path=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
build_path="$script_path/../../build"

TARGET="benchmark"

device_maps=("0" "0 2")
models=("resnet50" "resnet101" "bert_base" "bert_large" "roberta_base" "roberta_large" "gpt2" "gpt2_medium")
engines=("pipeline" "deepplan")
batch_size=1

tmp_avg_file="/tmp/deepplan_fig10_avg"
tmp_min_file="/tmp/deepplan_fig10_min"
tmp_max_file="/tmp/deepplan_fig10_max"

printf "" > $tmp_avg_file
printf "" > $tmp_min_file
printf "" > $tmp_max_file

for model in "${models[@]}"; do
	# Baseline
	baseline_cmd="$build_path/$TARGET -m $model -e demand -b $batch_size -d 0"
	echo "Run $baseline_cmd"

	output=`$baseline_cmd`
	echo "$output"
	echo ""

	avg_lat=$(echo "$output" | awk '{if ($1 == "Average") { print $(NF-1)}}')
	min_lat=$(echo "$output" | awk '{if ($1 == "Min") { print $(NF-1)}}')
	max_lat=$(echo "$output" | awk '{if ($1 == "Max") { print $(NF-1)}}')

	printf "$avg_lat, " >> $tmp_avg_file
	printf "$min_lat, " >> $tmp_min_file
	printf "$max_lat, " >> $tmp_max_file

	for device_map in "${device_maps[@]}"; do
		for engine in "${engines[@]}"; do
			cmd="$build_path/$TARGET -m $model -e $engine -b $batch_size -d $device_map"
			echo "Run $cmd"

			output=`$cmd`
			echo "$output"
			echo ""

			avg_lat=$(echo "$output" | awk '{if ($1 == "Average") { print $(NF-1)}}')
			min_lat=$(echo "$output" | awk '{if ($1 == "Min") { print $(NF-1)}}')
			max_lat=$(echo "$output" | awk '{if ($1 == "Max") { print $(NF-1)}}')

			printf "$avg_lat, " >> $tmp_avg_file
			printf "$min_lat, " >> $tmp_min_file
			printf "$max_lat, " >> $tmp_max_file
		done
	done

	echo "" >> $tmp_avg_file
	echo "" >> $tmp_min_file
	echo "" >> $tmp_max_file

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

avg_file="$output_path/result_avg.csv"
min_file="$output_path/result_min.csv"
max_file="$output_path/result_max.csv"

cp $tmp_avg_file $avg_file
cp $tmp_min_file $min_file
cp $tmp_max_file $max_file

echo "Created log files in '$output_path'"

is_installed=$(pip list | grep -F matplotlib)

if [ -z "$is_installed" ]; then
	echo "Matplotlib is not installed. So the graph can not be created."
else
	eval "python3 graph.py $avg_file $min_file $max_file fig10.pdf"
fi
