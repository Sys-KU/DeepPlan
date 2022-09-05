#!/bin/bash

if [[ $UID != 0 ]]; then
	echo "Please run this script with sudo:"
	echo "sudo $0 $*"
	exit 1
fi

script_path=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
build_path="$script_path/../../../build"

args_list=(
						"-t linear -i 768 -o 768 -s 384 768"
						"-t linear -i 768 -o 3072 -s 384 768"
						"-t conv -i 64 -o 64 -k 1 -s 64 56 56"
						"-t conv -i 256 -o 256 -k 3 -s 256 14 14"
						"-t conv -i 512 -o 512 -k 3 -s 512 7 7"
						"-t emb -i 2 -o 768 -s 384"
						"-t emb -i 512 -o 768 -s 384"
						"-t emb -i 30522 -o 768 -s 384"
					)

TARGET="experiments/pcm_pcie"

output_file="$script_path/result.csv"
echo "Load numRdrCur, DHA numRdrCur, Load Time (ms), Exec Time (ms), DHA Exec Time (ms)" > $output_file

for args in "${args_list[@]}"; do
	cmd="$build_path/$TARGET $args"
	echo "Run $cmd"
	output=`$cmd 2> /dev/null`
	echo $output

	load_rdcur=$(echo "$output" | awk '{if (NR == 1) { print $NF }}')
	dha_rdcur=$(echo "$output" | awk '{if (NR == 2) { print $NF }}')
	load_time=$(echo "$output" | awk '{if (NR == 3) { print $(NF-1) }}')
	exec_time=$(echo "$output" | awk '{if (NR == 3) { print $(NF-1) }}')
	dha_time=$(echo "$output" | awk '{if (NR == 3) { print $(NF-1) }}')

	echo "$load_rdcur, $dha_rdcur, $load_time, $exec_time, $dha_time" >> $output_file
done

