python run.py --identifier test --task lenet --seed 0 \
              --gpu m1 --num-workers 2 \
              --loglevel DEBUG --eval-every 5 \
              --epochs 5 --optimizer adam --lr 1e-3 --batch-size 32 \
              --rho 0.1 --prune-threshold 0.005 --scaled-max adam \
              --decay-type l1 --decay-rate 0.1
