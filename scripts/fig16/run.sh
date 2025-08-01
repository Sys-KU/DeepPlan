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

model_list="bert_base bert_base bert_base bert_base \
            roberta_base roberta_base roberta_base roberta_base \
            resnet50 resnet50 resnet50 resnet50 \
            resnet101 resnet101 resnet101 resnet101 \
            bert_large bert_large bert_large bert_large \
            roberta_large roberta_large roberta_large roberta_large"
conc=120
rate=140

engines=("deepplan" "pipeline")
r_policies=("dynamic" "lru")

for engine in "${engines[@]}"; do
    for r_policy in "${r_policies[@]}"; do
        server_cmd="$build_path/server -w 0.8"

        echo "Run Server"
        $server_cmd & 1> /dev/null

        echo "Wait 30 seconds for the server to be ready."
        sleep 30

        _engine=$engine

        tmp_file="/tmp/deepplan_${engine}_${r_policy}_fig14"
        tmp_dump_file="/tmp/dump_${engine}_${r_policy}"
        printf "" > $tmp_file

        echo "Start Experiment ($engine)"
        client_cmd="$build_path/client -m $model_list -e $_engine -r $rate -c $conc -w azure --disable_prefetch --disable_timeout --r_policy $r_policy --dump $tmp_dump_file"
        echo "$client_cmd"
        stdbuf --output=L $client_cmd | tee -a $tmp_file

        server_pid=$(ps -ef | grep -v grep | grep "$server_cmd" | awk '{print $2}')
        kill -s SIGINT $server_pid

        echo "Closing Server"

        wait
    done
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
    for r_policy in "${r_policies[@]}"; do
        tmp_file="/tmp/deepplan_${engine}_${r_policy}_fig14"
        tmp_dump_file="/tmp/dump_${engine}_${r_policy}"

        output_file="$log_path/${engine}_${r_policy}.csv"
        dump_file="$log_path/dump_${engine}_${r_policy}.csv"

        awk '$1 ~ /^[0-9]*,/ { print $3 $4 $5 }' $tmp_file > $output_file
        cp $tmp_dump_file $dump_file

        echo "Created '$output_file' log file"
    done
done

output_file="$log_path/offered_load.csv"
awk '$1 ~ /^[0-9]*,/ { print $2 }' $tmp_file > "$log_path/offered_load.csv"
echo "Created '$output_file' log file"

is_installed=$(pip list | grep -F matplotlib)

if [ -z "$is_installed" ]; then
	echo "Matplotlib is not installed. So the graph can not be created."
else
	eval "python3 graph.py $log_path fig14.pdf"
fi
