'''
for batch_size in 250 500 1000; do
    gpu=0
    for lr in 0.1 0.01 0.001 0.0001; do
        python3 ce.py --batch-size $batch_size --gpu $gpu --lr $lr --w 0.95 &
        gpu=$((gpu+1))
    done
    wait
done
'''

for ges in True False; do
    for iw in none quadratic; do
        for lra in 0.001 0.0001; do
            for lrc in 0.001 0.0001; do
                for nic in 25 50; do
                    gpu=0
                    for np in 25 50; do
                        for sn in a p; do
                            for std in 1 0.1; do
                                python3 actor-critic.py --ges $ges --gpu $((gpu%4)) --iw $iw --lra $lra --lrc $lrc --nic $nic --np $np --sn $sn --std $std &
                                gpu=$((gpu+1))
                            done
                        done
                    done
                done
            done
        done
    done
done
