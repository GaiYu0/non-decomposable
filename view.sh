# Assume that "./raw" does not contain a directory named "tmp".

field_arr=(batch_size_c batch_size_critic gpu iw n_iterations n_iterations_critic n_perturbations std tau topk)
declare -A field_map
for i in ${!field_arr[@]}; do
    field_map["${field_arr[i]}"]=$i
done

# for key in ${!field_map[@]}; do
#     echo $key ${field_map[$key]}
# done

view () {
    if [ $# -eq 0 ]; then
        return 0
    else
        mkdir tmp
        mv * tmp
        while true; do
            filename=$(ls tmp | head -n 1)
            if ! [[ -z $filename ]]; then
                field=${field_map[$1]}
                value=$(echo $filename | cut -d'-' -f $((2 * ${field} + 3)))
                folder=${1}-${value}
                mkdir $folder
                mv tmp/*${folder}* $folder
            else
                rm -r tmp
                shift
                for folder in *; do
                    cd $folder
                    view $@
                    cd ..
                done
                break
            fi
        done
    fi
}

mkdir view
ln raw/* view -s
cd view
view $@
