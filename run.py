"""This file contains code for running tabular algorithms in
the tiny Hanabi suite with independent learners (irl), simplified action
decoding (sad), binary simplified action decoding (bsad), private simplified
action decoding (psad), additive value decomposition (avd), the public belief
MDP (pubmdp), the temporal belief MDP (tbmdp), and the vacuous belief mdp
(vbmdp). The `run` function is the only function that needs to be accessed to
run experiments and is the only function that enforces type constrains. This
file also has a command line interface by which `run` can be accessed."""

import argparse 
import itertools 
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import algorithms
import games

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

def run(game, learner):
	eval_episodes = set(schedule(learner.num_episodes, learner.num_evals))
	expected_returns = [evaluate(game, learner)]
	for t in range(1, learner.num_episodes + 1):
		run_episode(game, learner)
		if t in eval_episodes:
			expected_returns.append(evaluate(game, learner))
	return expected_returns

def evaluate(game, learner):
	returns = []
	for s0 in game.start_states():
		returns.append(run_episode(game, learner, train=False, s0=s0))
	return np.mean(returns)

def run_episode(game, learner, train=True, s0=None):
	game.reset(s0)
	while not game.is_terminal():
		a = learner.act(game.context(), game.num_legal_actions(), train)
		game.step(a)
	if train:
		learner.update_from_episode(game.episode())
		learner.update_rates()
	return game.payoff()


def schedule(num_episodes: int, num_evals: int):
	"""Construct policy evaluation schedule

	:param num_episodes: The number of episodes
	:param num_evals: The number of policy evaluations
	"""
	return [(t * num_episodes) // (num_evals - 1) for t in range(num_evals)]

GAMENAMES = ('A', 'B', 'C', 'D', 'E', 'F')
SETTINGS = ('decpomdp', 'pubmdp', 'tbmdp', 'vbmdp')
ALGORITHMS = ('ql', 'hql', 'reinforce', 'a2c')
SADS = ('sad', 'bsad', 'asad', 'psad')

def interface(gamename, 
			  setting, 
			  algorithm, 
			  pg_init_lr=None, 
			  ql_init_lr=None, 
			  ql_init_lr2=None,
			  init_epsilon=None,
			  avd=None,
			  central_critic=None,
			  sad=None,
			  num_episodes=None,
			  num_evals=None,
			  fn=None,
			  plot=None):
	game = games.get_game(gamename, setting)
	if setting == 'decpomdp':
		if algorithm == 'ql':
			learner = algorithms.make_iql(ql_init_lr, init_epsilon, sad, avd, num_episodes, num_evals)
		if algorithm == 'hql':
			learner = algorithms.make_ihql(ql_init_lr, ql_init_lr2, init_epsilon, sad, avd, num_episodes, num_evals)
		if algorithm == 'reinforce':
			learner = algorithms.make_ireinforce(pg_init_lr, sad, num_episodes, num_evals)
		if algorithm == 'a2c':
			learner = algorithms.make_ia2c(pg_init_lr, ql_init_lr, sad, central_critic, num_episodes, num_evals)
	else:
		if algorithm == 'ql':
			learner = algorithms.make_ql(ql_init_lr, init_epsilon, num_episodes, num_evals)
		if algorithm == 'hql':
			learner = algorithms.make_hql(ql_init_lr, ql_init_lr2, init_epsilon, num_episodes, num_evals)	
		if algorithm == 'reinforce':
			learner = algorithms.make_reinforce(pg_init_lr, num_episodes, num_evals)
		if algorithm == 'a2c':
			learner = algorithms.make_a2c(pg_init_lr, ql_init_lr, num_episodes, num_evals)
	expected_returns = run(game, learner)
	df = pd.DataFrame({'episode': schedule(num_episodes, num_evals),
					   'expected_return': expected_returns,
					   'optimal_return': num_evals * [game.optimal_return],
					   'gamename': num_evals * [gamename], 
					   'setting': num_evals * [setting], 
					   'algorithm': num_evals * [algorithm], 
					   'pg_init_lr': num_evals * [pg_init_lr],
					   'ql_init_lr': num_evals * [ql_init_lr], 
					   'ql_init_lr2': num_evals * [ql_init_lr2], 
					   'init_epsilon': num_evals * [init_epsilon],
					   'avd': num_evals * [avd],
					   'central_critic': num_evals * [central_critic],
					   'sad': num_evals * [sad]})
	df.to_pickle('results/' + fn + '.pkl')
	if plot:
		plt.axhline(y=game.optimal_return, color='gray', linestyle='-', linewidth=1)
		ax = sns.lineplot(data=df, x='episode', y='expected_return')
		if central_critic:
			algorithm = 'a2c2'
		title = f'Game {gamename}; {setting}; {algorithm};'
		if avd:
			title += ' avd;'
		if sad:
			title += f' {sad};'
		if pg_init_lr:
			title += f' pglr={pg_init_lr};'
		if ql_init_lr:
			title += f' qllr={ql_init_lr};'
		if init_epsilon:
			title += f' eps={init_epsilon};'
		plt.title(title[:-1])
		plt.savefig('figures/' + fn + '.pdf')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('gamename', choices=GAMENAMES)
	parser.add_argument('setting', choices=SETTINGS)
	parser.add_argument('algorithm', choices=ALGORITHMS)
	parser.add_argument('--pg_init_lr', type=float)
	parser.add_argument('--ql_init_lr', type=float)
	parser.add_argument('--ql_init_lr2', type=float)
	parser.add_argument('--init_epsilon', type=float)
	parser.add_argument('--avd', default=False, action='store_true')
	parser.add_argument('--central_critic', default=False, action='store_true')
	parser.add_argument('--sad', default=False, choices=SADS)
	parser.add_argument('--num_episodes', type=int, default=int(1e6))
	parser.add_argument('--num_evals', type=int, default=100)
	parser.add_argument('--fn', default='example')
	parser.add_argument('--plot', default=False, action='store_true')
	args = parser.parse_args()
	interface(**vars(args))
