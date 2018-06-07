run () {
    if [ $1 -eq 0 ]; then
        for x in *; do
#           tensorboard --logdir=$x --port=$port &
            echo $x $port
            port=$((port + 1))
        done
        wait
        return
    else
        for x in *; do
            cd $x
            run $(($1 - 1))
            cd ..
        done
    fi
}

port=$2
run $1
