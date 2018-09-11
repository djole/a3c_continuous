#/usr/bin/bash
source ../.virtual/bin/activate
#python main.py --workers=3 --show=none --load=True --load-model-dir=./trained_models/ --scale-legs=0.5 --obstacle-prob=0.5
python main.py --workers=1 --show=none --load-model-dir=./trained_models/ --scale-legs=0.5 --obstacle-prob=0.5 --env=a3cwalker --gpu-ids=-1

