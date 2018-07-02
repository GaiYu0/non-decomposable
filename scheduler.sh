# xyz_arr=(x y z)
# for xyz in "${xyz_arr[@]}"

gpu=0
for
do
    ipython3 &
    gpu=$((gpu + 1))
done
wait
