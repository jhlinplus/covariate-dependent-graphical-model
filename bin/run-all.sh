#!/bin/bash
data_str=${1-DAG1}
gpu=${2-0}

generate_data=true
run_r=true
data_seeds=(0 1 2 3 4)
train_size=10000

cd ..
pwd; hostname; date

eval "$(conda shell.bash hook)"
python --version

if [[ "$generate_data" == "true" ]]; then
    echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    echo "start data generation"
    echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
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
fi

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "start neural network estimation"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
for seed in "${data_seeds[@]}";
do
    ########## perform estimation ################
    runCMD="python -u run_sim.py --data_str=${data_str} --train_size=${train_size} --data_seed=$seed --gpu=$gpu --debug"
    echo "runCMD: ${runCMD}"
    eval ${runCMD} || { echo 'some error occurred; exit 1'; exit 1;}
    date
done
echo "done estimation"

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "start RegGMM (linear) estimation"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
for seed in "${data_seeds[@]}";
do
    ########## perform linear estimation ################
    runCMD="python -u run_sim.py --data_str=${data_str} --train_size=${train_size} --data_seed=$seed --gpu=$gpu --version=reggmm --debug"
    echo "runCMD: ${runCMD}"
    eval ${runCMD} || { echo 'some error occurred; exit 1'; exit 1;}
    date
done
echo "done RegRMM estimation"

conda deactivate

if [[ "$run_r" == "true" ]]; then
    conda activate rvenv
    echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    echo "start R (glasso/mb) estimation"
    echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    for seed in "${data_seeds[@]}";
    do
        ########## perform estimation ################
        runCMD="Rscript --vanilla run_compete.R --data_str=${data_str} --data_seed=${seed} --train_size=${train_size}"

        echo "runCMD: ${runCMD}"
        eval ${runCMD} || { echo 'some error occurred; exit 1'; exit 1;}

        date
    done
    echo "done estimation with R packages"
    conda deactivate
fi
date
