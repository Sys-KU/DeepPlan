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
engine="deepcache"
workload="skew"
min_conc=50
max_conc=160
step_conc=10
SLO=100
rate=50

server_cmd="$build_path/server"

echo "Run Server"
$server_cmd & 1> /dev/null

echo "Wait 30 seconds for the server to be ready."
sleep 30

tmp_file="/tmp/deepcache_bursty_${model_name}_${engine}"

echo "Concurrency, Latency, Goodput, ColdStart" > $tmp_file

echo "Model Setup"
#client_cmd="$build_path/client -m $model_name -r $rate -c $max_conc -w $workload -e $engine -s $SLO"
#$client_cmd 1> /dev/null

echo "Start Experiment ($engine)"
for ((c=$min_conc; c<=$max_conc; c+=$step_conc)); do
	echo "== Concurrency $c =="
	client_cmd="$build_path/client -m $model_name -r $rate -c $c -w $workload -e $engine -s $SLO"
	output=`$client_cmd`

	latency=$(echo "$output" | awk '{if ($2 == "Latency:") { print $(NF-1)}}')
	goodput_rate=$(echo "$output" | awk '{if ($1 == "Goodput") { print $(NF-1)}}')
	cold_rate=$(echo "$output" | awk '{if ($1 == "Cold") { print $(NF-1)}}')
	echo "$output"
	printf "$c, $latency, $goodput_rate, $cold_rate" >> $tmp_file
	echo "" >> $tmp_file
done


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

tmp_file="/tmp/deepcache_bursty_${model_name}_${engine}"

output_file="$log_path/${model_name}_$engine.csv"

cp $tmp_file $output_file

server_pid=$(ps -ef | grep -v grep | grep "$server_cmd" | awk '{print $2}')
kill -s SIGINT $server_pid

echo "Closing Server"

wait
