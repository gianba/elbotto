#!/bin/bash

# turn on bash's job control
set -m

# Start the primary process and put it in the background
tensorboard --logdir=../dqnagent/summary --bind_all &

# Start the helper process
python ./elbotto/launcher.py $@

# now we bring the primary process back into the foreground
# and leave it there
fg %1