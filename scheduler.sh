# xyz_arr=(x y z)
# for xyz in "${xyz_arr[@]}"

gpu=0
for std in 0.1 0.5 2.5 7.5
do
    python3 parameter.py --gpu $gpu --std $std &
    gpu=$((gpu + 1))
done
wait
