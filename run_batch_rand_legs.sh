#/usr/bin/bash
source ../.virtual/bin/activate
#python main.py --workers=3 --show=none --load=True --load-model-dir=./trained_models/ --scale-legs=0.5 --obstacle-prob=0.5


lens="0.5 1 1.5 2"
experiments="0 1 2 3 4"

for i in $experiments
do
   for l in $lens
    do
        echo $i
        python main.py --workers=2 --show=none --scale-legs=$l --obstacle-prob=0.0 --env=a3cwalker_$i &
    done
done