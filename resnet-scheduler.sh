iw_arr=("" sqrt linear quadratic)

gpu=0
for iw in "${iw_arr[@]}"
do
    ipython3 $1 -- --gpu=$gpu --iw=$iw --n-iterations-critic=25 --n-perturbations=25 &
    gpu=$((gpu + 1))
done
wait

gpu=0
for iw in "${iw_arr[@]}"
do
    ipython3 $1 -- --gpu=$gpu --iw=$iw --n-iterations-critic=25 --n-perturbations=50 &
    ipython3 $1 -- --gpu=$gpu --iw=$iw --n-iterations-critic=50 --n-perturbations=25 &
    gpu=$((gpu + 1))
done
wait
