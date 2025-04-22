#!/bin/bash
data_str=${1-GGM0}
gpu=${2-0}
version=${3}

data_seeds=(0 1 2 3 4)
train_size=10000

cd ..
pwd; hostname; date

echo "start estimation"
for seed in "${data_seeds[@]}";
do
    
    ########## perform estimation ################
    runCMD="python -u run_sim.py --data_str=${data_str} --train_size=${train_size} --data_seed=$seed --gpu=$gpu --version=$version --debug"

    echo "runCMD: ${runCMD}"
    eval ${runCMD} || { echo 'some error occurred; exit 1'; exit 1;}

    date
done
echo "done estimation"
