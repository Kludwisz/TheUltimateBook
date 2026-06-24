GPUS=$(nvidia-smi -L | wc -l)
for i in $GPUS; do 
    ./a.out --runs $1 --cutoff $2 --device $((i)) &
done
