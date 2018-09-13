#/usr/bin/bash
#source ../.virtual/bin/activate
#python main.py --workers=3 --show=none --load=True --load-model-dir=./trained_models/ --scale-legs=0.5 --obstacle-prob=0.5


lens="0.8 0.9 1.1 1.2"
experiments="0 1 2 3 4"

for i in $experiments
do
   echo python main.py --workers=1 --show=none --load-model-dir=./trained_models/ --scale-legs=$l --obstacle-prob=0.0 --env=a2cwalker$i --seed=$i
done
