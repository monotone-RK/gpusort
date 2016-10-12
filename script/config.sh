#!/bin/bash
#file include common variable

bin_test_dir=/path/to/bin/test
gen_data_dir=/path/to/data
result_dir=/path/to/log

stdout_file=$result_dir/stdout.txt
error_file=$result_dir/stderr.txt
result_file=$result_dir/status.txt

host_tag="-H"
np_on_each_node=4
RM="rm -r -f $gen_data_dir/*"
RM_="rm -r -f $result_dir/*"
type_size=mb
n_cores=24
list_hosts=("hostname01" "hostname02")

