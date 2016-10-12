#!/bin/bash
source config.sh
$RM_
success_number=0
failure_number=0
i=0
total_file=$(find . -name "hk_*" | wc -l)
name_of_algorithm="hyk_sort"

run_subscript=false
begin_script=""
end_script=""
foo=false

while getopts "a:db:e:" opt; do
  case $opt in
    a)
      name_of_algorithm=$OPTARG >&2
      ;;
    d)
      run_subscript=true >&2
      ;;
    b)
      begin_script=$OPTARG >&2
      ;;
    e)
      end_script=$OPTARG >&2
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done


for i in `seq 1 $total_file`
do
  if [ $i -lt 10 ]; then
    testcase[$i]="hk_0$i -a $name_of_algorithm"
  else
    testcase[$i]="hk_$i -a $name_of_algorithm"
  fi
done

echo Start testing script
sleep 2
echo ""
#if run all testcases
if [ "$run_subscript" = false ]; then
  j=1
  for i in "${testcase[@]}"
  do
    echo -en "Running $j/$total_file testcases  : Success $success_number/$[$j-1]  :  Failure $failure_number/$[$j-1]"
    ./$i
#    echo "$i"
    if [ $? == 0 ]; then
      let success_number++ 1
    else
      let failure_number++ 1
    fi
    echo -en "\r"
      let j++ 1
  done
  echo -en "run $[$j - 1]/$total_file testcases   :  Success   $success_number/$[$j - 1]  :  Failure $failure_number/$[$j - 1]"
  echo ""
#if run a block testcases
else
#check file input
  if ([ "$begin_script" = "" ]) || ([ "$end_script" = "" ]); then
  echo "No input file"
  else
    j=1
    for i in "${testcase[@]}"
    do
      if echo "$i" | grep -q "${begin_script}"; then
        foo=true
        begin_script=$j
#        echo $j
        break
      else
        let j++ 1
      fi
    done

    if [ "$foo" = true ]; then 
      foo=false
      j=1
      for i in "${testcase[@]}"
      do
        if echo "$i" | grep -q "${end_script}"; then
          foo=true
          end_script=$j
#          echo $j 
          break
        else
          let j++ 1
        fi
      done
    fi
#parameter is invalid
    if [ "$foo" = false ]; then
      echo "Invalid file"
      echo $begin_script
      echo $end_script
    else
#parameter is valid
      j=1
      if [ $begin_script -gt $end_script ]; then
        swap=$begin_script
        begin_script=$end_script
        end_script=$swap
      fi
      total_file=$[end_script - begin_script + 1]
      for k in `seq $begin_script $end_script`
      do
        echo -en "Running $j/$total_file testcases  : Success $success_number/$[$j-1]  :  Failure $failure_number/$[$j-1]"
        ./${testcase[$k]}
        #echo "${testcase[$k]}"
        if [ $? == 0 ]; then
          let success_number++ 1
        else
          let failure_number++ 1
        fi
          echo -en "\r"
          let j++ 1
      done
      echo -en "run $[$j - 1]/$total_file testcases   :  Success   $success_number/$[$j - 1]  :  Failure $failure_number/$[$j - 1]"
      echo ""
    fi
  fi
fi
