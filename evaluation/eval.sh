#!/bin/bash

python3 -u run_AA.py --local --gpu 0 --normalize --dataset $1 --model $2 --eps $3 --root $4 --exp_name $5 --csv $6 --distance $7