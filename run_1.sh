cd ./src/

python3  train.py --task hopper --device cuda:0
python3  train.py --task walker --device cuda:0
python3  train.py --task enduro --device cuda:0
