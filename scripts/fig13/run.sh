#!/bin/bash

PLAN_REPO=${PLAN_REPO}

if [[ -z "$PLAN_REPO" ]]; then
	echo "PLAN_REPO environment variable not set, please set this variable"
	return
fi

export PLAN_REPO=${PLAN_REPO}

script_path=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd
) build_path="$script_path/../../build"

TARGET="client"

engines=("deepplan+" "deepplan" "pipeline")

server_cmd="$build_path/server"

echo "Run Server"
$server_cmd & 1> /dev/null

echo "Wait 30 seconds for the server to be ready."
sleep 30

model_name="bert_large" min_conc=5 max_conc=55 step_conc=5 rate=30

for engine in "${engines[@]}"; do p_option=1

	_engine=$engine
	if [ "$engine" = "deepplan+" ]; then
		_engine="deepplan"
		p_option=2
	fi

	tmp_file="/tmp/deepplan_${model_name}_${engine}_fig13"
	printf "" > $tmp_file

	echo "Model Setup"
	client_cmd="$build_path/client -m $model_name -e $_engine -r $rate -c 60 -w simple -p $p_option"
	$client_cmd 1> /dev/null

	echo "Start Experiment ($engine)"
	for ((c=$min_conc; c<=$max_conc; c+=$step_conc)); do
		echo "== Concurrency $c =="
		client_cmd="$build_path/client -m $model_name -e $_engine -r $rate -c $c -w simple -p $p_option"
		output=`$client_cmd`

		latency=$(echo "$output" | awk '{if ($2 == "Latency:") { print $(NF-1)}}')
		goodput_rate=$(echo "$output" | awk '{if ($1 == "Goodput") { print $(NF-1)}}')
		cold_rate=$(echo "$output" | awk '{if ($1 == "Cold") { print $(NF-1)}}')
		echo "$output"
		printf "$latency, $goodput_rate, $cold_rate" >> $tmp_file
		echo "" >> $tmp_file

	done
done

server_pid=$(ps -ef | grep -v grep | grep "$server_cmd" | awk '{print $2}')
kill -s SIGINT $server_pid

echo "Closing Server"

wait

echo "Run Server"
$server_cmd & 1> /dev/null

echo "Wait 30 seconds for the server to be ready."
sleep 30

model_name="gpt2"
min_conc=20
max_conc=200
step_conc=20
rate=90

for engine in "${engines[@]}"; do
	p_option=1

	_engine=$engine
	if [ "$engine" = "deepplan+" ]; then
		_engine="deepplan"
		p_option=2
	fi

	tmp_file="/tmp/deepplan_${model_name}_${engine}_fig13"
	printf "" > $tmp_file

	echo "Model Setup"
	client_cmd="$build_path/client -m $model_name -e $_engine -r $rate -c $max_conc -w simple -p $p_option"
	$client_cmd 1> /dev/null

	echo "Start Experiment ($engine)"
	for ((c=$min_conc; c<=$max_conc; c+=$step_conc)); do
		echo "== Concurrency $c =="
		client_cmd="$build_path/client -m $model_name -e $_engine -r $rate -c $c -w simple -p $p_option -s 200"
		output=`$client_cmd`

		latency=$(echo "$output" | awk '{if ($2 == "Latency:") { print $(NF-1)}}')
		goodput_rate=$(echo "$output" | awk '{if ($1 == "Goodput") { print $(NF-1)}}')
		cold_rate=$(echo "$output" | awk '{if ($1 == "Cold") { print $(NF-1)}}')
		echo "$output"
		printf "$latency, $goodput_rate, $cold_rate" >> $tmp_file
		echo "" >> $tmp_file
	done


done

server_pid=$(ps -ef | grep -v grep | grep "$server_cmd" | awk '{print $2}')
kill -s SIGINT $server_pid

echo "Closing Server"

wait

log_path="$script_path/logs"

# Check for log_path existence
if [ ! -d "$log_path" ]; then
	mkdir -p $log_path
	echo "Created $log_path directory where log files will be stored"
fi

date=`date +%y-%m-%d`

log_path="$log_path/$date/report"

version=0
while true; do
	_log_path="${log_path}$version"
	if [ -d "$_log_path" ]; then
		((version++))
	else
		break
	fi
done

log_path=$_log_path
mkdir -p "$log_path"

model_names=("bert_large" "gpt2")

for model in "${model_names[@]}"; do
	for engine in "${engines[@]}"; do
		tmp_file="/tmp/deepplan_${model}_${engine}_fig13"

		output_file="$log_path/${model}_${engine}.csv"

		cp $tmp_file $output_file

		echo "Created '$output_file' log file"
	done
done

is_installed=$(pip list | grep -F matplotlib)

if [ -z "$is_installed" ]; then
	echo "Matplotlib is not installed. So the graph can not be created."
else
	eval "python3 graph.py $log_path fig13.pdf"
fi
