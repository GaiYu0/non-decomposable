iw_arr=("" sqrt linear quadratic)

gpu=0
for iw in "${iw_arr[@]}"
do
    ipython3 $1 -- --gpu=$gpu --iw=$iw --n-iterations-critic=1 &
    ipython3 $1 -- --gpu=$gpu --iw=$iw --n-iterations-critic=5 &
    gpu=$((gpu + 1))
done
wait

gpu=0
for iw in "${iw_arr[@]}"
do
    ipython3 $1 -- --gpu=$gpu --iw=$iw --n-iterations-critic=10 &
    ipython3 $1 -- --gpu=$gpu --iw=$iw --n-iterations-critic=15 &
    ipython3 $1 -- --gpu=$gpu --iw=$iw --n-iterations-critic=20 &
    gpu=$((gpu + 1))
done
wait
