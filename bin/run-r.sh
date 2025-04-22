#!/bin/bash
data_str=${1-GGM0}

data_seeds=(0 1 2 3 4)
train_size=10000

cd ..
pwd; hostname;

for seed in "${data_seeds[@]}";
do
    if [[ "$data_str" == "GGM1" ]] || [[ "$data_str" == "NPN1" ]] || [[ "$data_str" == DAG* ]]; then
        regimes=(1 2 3)
    elif [[ "$data_str" == "GGM2" ]] || [[ "$data_str" == "NPN2" ]]; then
        regimes=(1 2)
    else
        echo "unrecognized data_str=${data_str}"
        exit 1
    fi
    ########## perform estimation ################
    runCMD="Rscript --vanilla run_compete.R --data_str=${data_str} --data_seed=${seed} --train_size=${train_size}"
    echo "runCMD: ${runCMD}"
    eval ${runCMD} || { echo 'some error occurred; exit 1'; exit 1;}
    date
    
    ########## perform estimation by regime ################
    for regime in "${regimes[@]}";
    do
        runCMD="Rscript --vanilla run_compete.R --data_str=${data_str} --data_seed=${seed} --regime=${regime} --train_size=${train_size}"
        echo "runCMD: ${runCMD}"
        eval ${runCMD} || { echo 'some error occurred; exit 1'; exit 1;}
        date
    done
done
