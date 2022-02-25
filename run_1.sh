cd ./src/

python3  test_ppo.py --task hopper --device cuda:0
python3  test_ppo.py --task walker --device cuda:0
python3  test_ppo.py --task enduro --device cuda:0
