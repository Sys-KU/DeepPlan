#!/bin/bash
script_path=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
output_dir="$script_path/azure-functions"

echo "Downloading Azure trace dataset"
wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz

mkdir -p $output_dir

echo "Extract azurefunctions-dataset2019.tar.xz"
tar -xvf azurefunctions-dataset2019.tar.xz -C $output_dir

echo "The Azure trace datasets are saved to '$outpu_dir'"
echo "To run azure experiments, follow the command below"
echo "export AZURE_TRACE_DIR=\"$output_dir\""

