iw_arr=(none sqrt linear quadratic)

gpu=0
for iw in "${iw_arr[@]}"
do
    ipython3 $1 -- --gpu $gpu --iw $iw &
    gpu=$((gpu + 1))
done
wait
