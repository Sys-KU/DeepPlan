#!/bin/bash

PLAN_REPO=${PLAN_REPO}
if [[ -z "$PLAN_REPO" ]]; then
	echo "PLAN_REPO environment variable not set, please set this variable"
	return
fi

export PLAN_REPO=${PLAN_REPO}

script_path=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
build_path="$script_path/../../build"

TARGET="client"

model_name="bert_base"
min_conc=20
max_conc=200
rate=100
engines=("pipeline" "deepplan" "deepplan+")

server_cmd="$build_path/server"

echo "Run Server"
$server_cmd & 1> /dev/null

echo "Wait 30 seconds for the server to be ready."
sleep 30

for engine in "${engines[@]}"; do
	p_option=1

	_engine=$engine
	if [ "$engine" = "deepplan+" ]; then
		_engine="deepplan"
		p_option=2
	fi

	tmp_file="/tmp/deepplan_${_engine}_fig12"
	echo "" > $tmp_file

	echo "Model Setup"
	client_cmd="$build_path/client -m $model_name -e $_engine -r $rate -c $max_conc -w simple -p $p_option"
	$client_cmd 1> /dev/null

	echo "Start Experiment ($engine)"
	for ((c=$min_conc; c<=$max_conc; c+=10)); do
		echo "== Concurrency $c =="
		client_cmd="$build_path/client -m $model_name -e $_engine -r $rate -c $c -w simple -p $p_option"
		output=`$client_cmd`

		latency=$(echo "$output" | awk '{if ($2 == "Latency:") { print $(NF-1)}}')
		cold_rate=$(echo "$output" | awk '{if ($1 == "Cold") { print $(NF-1)}}')
		goodput_rate=$(echo "$output" | awk '{if ($1 == "Goodput") { print $(NF-1)}}')
		echo "$output"
		printf "$latency, $cold_rate, $goodput_rate" >> $tmp_file
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

	output_file="$log_path/report_$_engine"

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

	#eval "python graph.py $output_file figure.pdf"
done

server_pid=$(ps -ef | grep -v grep | grep "$server_cmd" | awk '{print $2}')
kill -s SIGINT $server_pid

echo "Closing Server"

wait
return
