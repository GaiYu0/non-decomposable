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
