"""This file contains six games dubbed the tiny Hanabi suite. The suite serves
as a toy testbed for algorithms for common-payoff games. Each game in the 
suite is structured as follows:
1) Each player is privately dealt one of `num_cards` cards at uniform random.
2) Player 1 selects an action, which is publicly observable.
3) Player 2 selects an action.
The payoff of the game is determined player 1's card, player 2's card, 
player 1's action, and player 2's action.
"""
from abc import ABC, abstractmethod
import itertools
from typing import Union

import numpy as np

GAMENAMES = ('A', 'B', 'C', 'D', 'E', 'F')

PAYOFFS = {'A': np.array([
							 [
								 [[0, 1], [0, 0]],
								 [[0, 1], [3, 2]],
							 ],
							 [
								 [[3, 3], [3, 2]],
								 [[2, 0], [3, 3]],
							 ],
						 ], dtype=np.float32),
		   'B': np.array([
							 [
								 [[1, 0], [1, 0]],
								 [[0, 1], [0, 1]],
							 ],
							 [
								 [[0, 1], [0, 0]],
								 [[1, 0], [1, 0]],
							 ],
						 ], dtype=np.float32),
		   'C': np.array([
							 [
								 [[3, 0], [0, 3]],
								 [[2, 0], [3, 3]],
							 ],
							 [
								 [[2, 2], [3, 0]],
								 [[0, 1], [0, 2]],
							 ],
						 ], dtype=np.float32),
		   'D': np.array([
							 [
								 [[3, 0], [1, 3]],
								 [[3, 0], [3, 0]],
							 ],
							 [
								 [[3, 2], [0, 2]],
								 [[0, 1], [0, 0]],
							 ],
						 ], dtype=np.float32),
		   'E': np.array([
					         [
					             [[10, 0, 0], [4, 8, 4], [10, 0, 0]],
					             [[0, 0, 10], [4, 8, 4], [0, 0, 10]],
					         ],
					         [
					             [[0, 0, 10], [4, 8, 4], [0, 0, 0]],
					             [[10, 0, 0], [4, 8, 4], [10, 0, 0]],
					         ],
					     ], dtype=np.float32),
		   'F': np.array([
					  	     [
					  		     [[0, 3], [3, 2]], 
					  		     [[0, 0], [0, 1]],
					  		     [[3, 1], [2, 1]],
					  	     ],
					  	     [
					  		     [[0, 2], [0, 1]],
					  		     [[1, 2], [1, 2]],
					  		     [[0, 1], [0, 3]],
					  	     ],
					  	     [
					  		     [[1, 3], [1, 2]],
					  		     [[0, 3], [2, 2]],
					  		     [[3, 1], [3, 0]],
					  	     ]
					     ], dtype=np.float32)
	      }

OPTIMAL_RETURNS = {'A': 2.25, 
				   'B': 1.00,
				   'C': 2.50,
				   'D': 2.50, 
				   'E': 10, 
				   'F': 2 + 1/3}


def normalize_payoffs(data: Union[float, np.ndarray], 
				 	  maximum: float, 
					  minimum: float) -> Union[float, np.ndarray]:
	"""Normalize data to [0, 1] from [minimum, maximum]"""
	return (data - minimum) / (maximum - minimum)

class Game(ABC):
	def __init__(self, payoffs: np.ndarray, optimal_return: float):
		self.num_cards = payoffs.shape[0]
		self.num_actions = payoffs.shape[-1]
		self.payoffs = payoffs
		self.optimal_return = optimal_return

	def random_start(self):
		start_states = self.start_states()
		self.history = start_states[np.random.choice(range(len(start_states)))]

	def is_terminal(self):
		return len(self.history) == self.horizon

	def payoff(self):
		return self.payoffs[tuple(self.history)] if self.is_terminal() else 0

	@abstractmethod
	def step(self, action):
		self.step_(action)

	@abstractmethod
	def num_legal_actions(self):
		pass

	@abstractmethod
	def context(self):
		pass

	@abstractmethod
	def episode(self):
		pass

	def cummulative_reward(self):
		return self.payoff()

	def reset(self, history=None):
		if history is None:
			self.random_start()
		else:
			self.history = history

class DecPOMDP(Game):
	def __init__(self, payoffs: np.ndarray, optimal_return: float):
		super().__init__(payoffs, optimal_return)
		self.num_players = 2
		self.horizon = 4

	def start_states(self):
		return tuple([i, j] for i in range(self.num_cards) for j in range(self.num_cards))

	def step(self, action):
		self.history.append(action)

	def num_legal_actions(self):
		return self.num_actions

	def context(self):
		return tuple(self.history)

	def episode(self):
		return self.history + [self.payoff()]

class PuBMDP(Game):
	def __init__(self, payoffs: np.ndarray, optimal_return: float):
		super().__init__(payoffs, optimal_return)
		self.num_players = 1
		self.horizon = 4
		self.build(payoffs)

	def start_states(self):
		return [[c] for c in range(self.num_cards)]

	def step(self, prescription):
		if len(self.history) == 1:
			prescription1_table = table_repr(prescription, self.num_cards, self.num_actions)
			action = np.argmax(prescription1_table[self.history[0]])
			belief = self.beliefs[prescription, action]
			self.history += [prescription, belief]
		elif len(self.history) == 3:
			self.history.append(prescription)

	def context(self):
		return None if len(self.history) == 1 else self.history[-1]

	def episode(self):
		return [(None, self.history[1], 0), (*self.history[2:], self.payoff())]

	def num_legal_actions(self):
		return self.num_prescriptions

	def build(self, payoffs):
		num_possible_info_state = self.num_cards
		self.num_prescriptions = self.num_actions ** num_possible_info_state
		self.beliefs = {}
		self.payoffs = {}
		for prescription1 in range(self.num_prescriptions):
			prescription1_table = table_repr(prescription1, self.num_cards, self.num_actions)
			for c1 in range(self.num_cards):
				a1 = np.argmax(prescription1_table[c1])
				possible_c1 = np.flatnonzero(prescription1_table[:, a1])
				b = (tuple(possible_c1), a1)
				self.beliefs[prescription1, a1] = b
				for prescription2 in range(self.num_prescriptions):
					prescription2_table = table_repr(prescription2, self.num_cards, self.num_actions)
					tmp = []
					for c1_ in possible_c1:
						for c2_ in range(self.num_cards):
							a2 = np.argmax(prescription2_table[c2_])
							tmp.append(payoffs[c1_, c2_, a1, a2])
					self.payoffs[c1, prescription1, b, prescription2] = np.mean(tmp)

class TBMDP(Game):
	def __init__(self, payoffs: np.ndarray, optimal_return: float):
		super().__init__(payoffs, optimal_return)
		self.num_players = 1
		self.horizon = 2
		self.build(payoffs)

	def start_states(self):
		return ([],)

	def step(self, action):
		self.history.append(action)

	def num_legal_actions(self):
		if len(self.history) == 0:
			return self.legal_actions[None]
		return self.legal_actions[self.history[0]]

	def context(self):
		if len(self.history) == 0:
			return None
		elif len(self.history) == 1:
			return self.history[0]

	def episode(self):
		return [(None, self.history[0], 0), (self.history[0], self.history[1], self.payoff())]

	def build(self, payoffs):
		self.payoffs = {}
		self.legal_actions = {}
		num_possible_info_state1 = self.num_cards
		num_prescriptions1 = self.num_actions ** num_possible_info_state1
		self.legal_actions[None] = num_prescriptions1
		for prescription1 in range(num_prescriptions1):
			prescription1_table = table_repr(prescription1, self.num_cards, self.num_actions)
			possible_a1 = np.flatnonzero(prescription1_table.max(axis=0))
			num_possible_info_state2 = self.num_cards * len(possible_a1)
			num_legal_actions2 = self.num_actions ** num_possible_info_state2 
			self.legal_actions[prescription1] = num_legal_actions2
			for prescription2 in range(num_legal_actions2):
				prescription2_table = table_repr(prescription2, num_possible_info_state2, self.num_actions)
				possible_a2 = np.flatnonzero(prescription2_table.max(axis=0))
				idx = {(c2_, a1_): i for i, (c2_, a1_) in enumerate(itertools.product(range(self.num_cards), possible_a1))}
				tmp = []
				for c1_ in range(self.num_cards):
					for c2_ in range(self.num_cards):
						a1_ = np.argmax(prescription1_table[c1_])
						a2_ = np.argmax(prescription2_table[idx[(c2_, a1_)]])
						tmp.append(payoffs[c1_, c2_, a1_, a2_])
				self.payoffs[prescription1, prescription2] = np.mean(tmp)


class VBMDP(Game):
	def __init__(self, payoffs: np.ndarray, optimal_return: float):
		super().__init__(payoffs, optimal_return)
		self.num_players = 1
		self.horizon = 1
		self.build(payoffs)

	def start_states(self):
		return ([],)

	def context(self):
		return None

	def step(self, action):
		self.history.append(action)

	def num_legal_actions(self):
		return self.num_action_profiles

	def episode(self):
		return [(None, self.history[0], self.payoff())]

	def build(self, payoffs):
		self.payoffs = {}
		num_cards = payoffs.shape[0]
		num_actions = payoffs.shape[-1]
		num_p1_info_states = num_cards
		num_p2_info_states = num_cards * num_actions
		num_info_states = num_p1_info_states + num_p2_info_states
		num_p1_action_profiles = num_actions ** num_p1_info_states
		num_p2_action_profiles = num_actions ** num_p2_info_states
		self.num_action_profiles = num_p1_action_profiles * num_p2_action_profiles
		for action_profile in range(self.num_action_profiles):
			action_profile_table = table_repr(action_profile, num_info_states, num_actions)
			tmp = []
			for c1 in range(num_cards):
				for c2 in range(num_cards):
					a1 = np.argmax(action_profile_table[c1])
					p2_info_state_idx = num_cards + a1 * num_cards + c2
					a2 = np.argmax(action_profile_table[p2_info_state_idx])
					tmp.append(payoffs[c1, c2, a1, a2])
			self.payoffs[(action_profile,)] = np.mean(tmp)


def table_repr(index: int, 
		  	  num_info_states: int, 
		  	  num_actions: int) -> np.ndarray:
	"""Express action with a table representation.

	:param index: Index of action
	:param num_info_states: Number of possible states in underlying game
	:param num_actions: Number of actions in underlying game
	:return: `num_states` by `num_actions` representation
	"""
	table = np.zeros((num_info_states, num_actions))
	for info_state in range(num_info_states):
		table[info_state, index % num_actions] = 1
		index = index // num_actions
	return table

def get_game(gamename: str, setting: str, normalize: bool=True) -> Game:
	"""Get game `gamename`.
	
	:param gamename: Name of the game
	:param normalize: Whether to normalize the payoffs of the game to [0, 1]
	:return: Possibly normalized game
	"""
	payoffs = PAYOFFS[gamename]
	optimal_return = OPTIMAL_RETURNS[gamename]
	if normalize:
		maximum, minimum = payoffs.max(), payoffs.min()
		payoffs = normalize_payoffs(payoffs, maximum, minimum)
		optimal_return = normalize_payoffs(optimal_return, maximum, minimum)
	if setting == 'decpomdp':
		return DecPOMDP(payoffs, optimal_return)
	if setting == 'pubmdp':
		return PuBMDP(payoffs, optimal_return)
	if setting == 'tbmdp':
		return TBMDP(payoffs, optimal_return)
	if setting == 'vbmdp':
		return VBMDP(payoffs, optimal_return)
