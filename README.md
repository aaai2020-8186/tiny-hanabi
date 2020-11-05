# tiny-hanabi

The code in this directory was used to produce all Tiny Hanabi experiments in the submission.
The code runs in python3.
Install the dependencies with "conda env create -f environment.yml".
Activate the environment with "source activate tiny-hanabi".
Try "python run.py A decpomdp ql --ql_init_lr 0.1 --init_epsilon 0.1 --num_episodes 10_000 --plot && open figures/example.pdf" to examine the learning progress over one short run.
