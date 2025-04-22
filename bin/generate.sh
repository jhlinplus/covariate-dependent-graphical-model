#!/bin/bash
data_str=${1-GGM0}

data_seeds=(0 1 2 3 4)

cd ..
pwd; hostname; date

echo "start data generation"
for seed in "${data_seeds[@]}";
do
    ########## generate data ################
    if [[ "$data_str" == GGM* ]]; then
        script_suffix="GGM"
    elif [[ "$data_str" == NPN* ]]; then
        script_suffix="NPN"
    elif [[ "$data_str" == DAG* ]]; then
        script_suffix="DAG"
    else
        echo "unsupported data_str=${data_str}"
        exit 1
    fi
    dataCMD="python -u generator/generate_${script_suffix}.py --data_str=${data_str} --data_seed=$seed"
    echo "dataCMD: ${dataCMD}"
    eval ${dataCMD} || { echo 'some error occurred; exit 1' ; exit 1; }
    date
done
echo "done data generation"
