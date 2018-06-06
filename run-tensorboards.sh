port=$2

for y in ${1}/*
do
    tensorboard --logdir=$y --port=$port &
    port=$((port + 1))
done
wait

# for x in log/${1}/processed/*
# do
#     for y in ${x}/*
#     do
#         tensorboard --logdir=$y --port=$port &
#         port=$((port + 1))
#     done
# done
# wait
