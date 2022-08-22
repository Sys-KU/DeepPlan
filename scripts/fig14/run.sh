#!/bin/bash

PLAN_REPO=${PLAN_REPO}
if [[ -z "$PLAN_REPO" ]]; then
	echo "PLAN_REPO environment variable not set, please set this variable"
	return
fi

AZURE_TRACE_DIR=${AZURE_TRACE_DIR}
if [[ -z "$AZURE_TRACE_DIR" ]]; then
	echo "AZURE_TRACE_DIR environment variable not set, please set thie variable"
	return
fi

export PLAN_REPO=${PLAN_REPO}

script_path=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
build_path="$script_path/../../build"

model_list="bert_base bert_base bert_base bert_base bert_base roberta_base roberta_base roberta_base roberta_base"
conc=216
rate=100

engines=("deepplan+" "deepplan" "pipeline")

for engine in "${engines[@]}"; do
	server_cmd="$build_path/server"

	echo "Run Server"
	$server_cmd & 1> /dev/null

	echo "Wait 30 seconds for the server to be ready."
	sleep 30

	p_option=1

	_engine=$engine
	if [ "$engine" = "deepplan+" ]; then
		_engine="deepplan"
		p_option=2
	fi

	tmp_file="/tmp/deepplan_${engine}_fig14"
	printf "" > $tmp_file

	echo "Start Experiment ($engine)"
	client_cmd="$build_path/client -m $model_list -e $_engine -r $rate -c $conc -w azure -p $p_option"
	$client_cmd | tee -a $tmp_file

	server_pid=$(ps -ef | grep -v grep | grep "$server_cmd" | awk '{print $2}')
	kill -s SIGINT $server_pid

	echo "Closing Server"

	wait

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

for engine in "${engines[@]}"; do
	tmp_file="/tmp/deepplan_${engine}_fig14"

	output_file="$log_path/${engine}.csv"

	cp $tmp_file $output_file
done

#eval "python3 graph.py $log_path fig14.pdf"
