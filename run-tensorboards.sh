port=$2

for std in 0.01 0.1 1.0
do
    for iw in none sqrt linear quadratic
    do
#       echo inspect/${1}-std-${std}/$iw
#       echo $port
        tensorboard --logdir=inspect/${1}-std-${std}/$iw --port=$port &
        port=$((port + 1))
    done
done
wait
