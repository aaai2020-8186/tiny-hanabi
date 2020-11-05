from abc import ABC, abstractmethod
from typing import Union

import numpy as np

def argmax(x: Union[list, np.ndarray]) -> int:
	"""Argmax with random tiebreaking."""
	if type(x) is list:
		x = np.array(x)
	return np.random.choice(np.flatnonzero(x == max(x)))

def softmax(logits: Union[list, np.ndarray]) -> np.ndarray:
	"""Softmax simplex projection."""
	if type(logits) is list:
		logits = np.array(logits)
	pref = np.exp(logits)
	return pref / pref.sum()

def gradient_update(value: float, lr: float, target: float) -> float:
	return (1 - lr) * value + lr * target

class BaseParameters(ABC):
	def __init__(self) -> None:
		self.vals = {}

	def add_state(self, state, num_legal_actions):
		if state not in self.vals:
			self.vals[state] = num_legal_actions * [0]

	@abstractmethod
	def update_learning_rate(self, denominator):
		pass	

	@abstractmethod
	def update_params(self, state, action, target):
		pass

class Parameters(BaseParameters):
	def __init__(self, init_lr: float) -> None:
		super().__init__()
		self.init_lr = init_lr
		self.lr = init_lr

	@abstractmethod
	def update_params(self, state, action, target):
		pass

	def update_learning_rate(self, denominator):
		self.lr -= self.init_lr / denominator

class DoubleLrParameters(BaseParameters):
	def __init__(self, init_lr1: float, init_lr2: float) -> None:
		super().__init__()
		self.init_lr1 = init_lr1
		self.init_lr2 = init_lr2
		self.lr1 = init_lr1
		self.lr2 = init_lr2

	@abstractmethod
	def update_params(self, state, action, target):
		pass

	def update_learning_rate(self, denominator):
		self.lr1 -= self.init_lr1 / denominator
		self.lr2 -= self.init_lr2 / denominator

class ValueParameters(Parameters):
	def __init__(self, init_lr : float) -> None:
		super().__init__(init_lr)

	def update_params(self, state, action, target):
		self.vals[state][action] = gradient_update(self.vals[state][action], self.lr, target)

class DoubleLrValueParameters(DoubleLrParameters):
	def __init__(self, init_lr1 : float, init_lr2 : float) -> None:
		super().__init__(init_lr1, init_lr2)

	def update_params(self, state, action, target):
		est = self.vals[state][action]
		lr = self.lr1 if target >= est else self.lr2
		self.vals[state][action] = gradient_update(est, lr, target)

class PolicyParameters(Parameters):
	def __init__(self, init_lr: float) -> None:
		super().__init__(init_lr)

	def update_params(self, state, action, target):
		logits = self.vals[state]
		probs = softmax(logits)
		for a_, p in enumerate(probs):
			grad_log_prob = int(a_ == action) - p
			self.vals[state][a_] = gradient_update(logits[a_], self.lr, grad_log_prob * target)

class Actor(ABC):
	def __init__(self, params):
		self.params = params

	def act(self, state, num_legal_actions, train):
		self.params.add_state(state, num_legal_actions)
		if train:
			return self.act_normally(state)
		return self.act_greedily(state)

	@abstractmethod
	def act_normally(self, state):
		pass

	def act_greedily(self, state):
		return np.argmax(self.params.vals[state])

	def update_exploration_rate(self, denominator):
		pass


class ValueActor(Actor):
	def __init__(self, params, init_epsilon):
		super().__init__(params)
		self.init_epsilon = init_epsilon
		self.epsilon = init_epsilon

	def act_normally(self, state):
		if np.random.random_sample() < self.epsilon:
			return np.random.choice(np.arange(len(self.params.vals[state])))
		return argmax(self.params.vals[state])

	def update_exploration_rate(self, denominator):
		self.epsilon -= self.init_epsilon / denominator

class PolicyActor(Actor):
	def __init__(self, params):
		super().__init__(params)

	def act_normally(self, state):
		probs = softmax(self.params.vals[state])
		return np.random.choice(range(len(probs)), p=probs)

class SingleAgentLearner(ABC):
	def __init__(self, actor, num_episodes, num_evals):
		self.actor = actor
		self.params = actor.params
		self.num_episodes = num_episodes
		self.num_evals = num_evals

	def act(self, state, num_legal_actions, train):
		return self.actor.act(state, num_legal_actions, train)

	def update_rates(self):
		self.actor.update_exploration_rate(self.num_episodes)
		self.params.update_learning_rate(self.num_episodes)

	@abstractmethod
	def update_from_episode(self, episode):
		pass

class QLearner(SingleAgentLearner):
	def __init__(self, actor, num_episodes, num_evals):
		assert type(actor) is ValueActor
		super().__init__(actor, num_episodes, num_evals)

	def update_from_episode(self, episode):
		q_update(self.params, episode)

class ReinforceLearner(SingleAgentLearner):

	def update_from_episode(self, episode):
		states, actions, rewards = zip(*episode)
		payoff = sum(rewards)
		for s, a in zip(states, actions):
			self.params.update_params(s, a, payoff)

class A2CLearner(SingleAgentLearner):
	def __init__(self, actor, critic, num_episodes, num_evals):
		super().__init__(actor, num_episodes, num_evals)
		self.critic = critic

	def act(self, state, num_legal_actions, train):
		return super().act(state, num_legal_actions, train)

	def update_from_episode(self, episode):
		for s, a, _ in episode:
			ac_update(self.actor, self.critic, s, s, a)
		q_update(self.critic, episode)


class MultiAgentLearner:
	def __init__(self, alice, bob, sad):
		self.alice = alice
		self.bob = bob
		self.sad = sad
		assert alice.num_episodes == bob.num_episodes
		self.num_episodes = alice.num_episodes
		assert alice.num_evals == bob.num_evals
		self.num_evals = alice.num_evals

	def act(self, context, num_legal_actions, train):
		if len(context) == 2:
			info_state = context[0]
			return self.alice.act(info_state, num_legal_actions, train)
		if len(context) == 3:
			info_state = sad_transform(self.alice, context, self.sad)
			return self.bob.act(info_state, num_legal_actions, train)

	def update_from_episode(self, episode):
		c1, c2, a1, a2, payoff = episode
		p2_info_state = sad_transform(self.alice, episode, self.sad)
		self.alice.update_from_episode([(c1, a1, payoff)])
		self.bob.update_from_episode([(p2_info_state, a2, payoff)])

	def update_rates(self):
		self.alice.update_rates()
		self.bob.update_rates()


class IndependentQLearner(MultiAgentLearner):
	def __init__(self, alice, bob, sad, avd):
		super().__init__(alice, bob, sad)
		self.avd = avd

	def update_from_episode(self, episode):
		if self.avd:
			c1, c2, a1, a2, payoff = episode
			p1_is2 = (c1, a1)
			p2_is2 = sad_transform(self.alice, episode, self.sad)
			avd_update(self.alice, self.bob, c1, c2, a1, 0, 0, p1_is2, p2_is2, False)
			avd_update(self.alice, self.bob, p1_is2, p2_is2, 0, a2, payoff, None, None, True)
		else:
			super().update_from_episode(episode)


class IndependentReinforceLearner(MultiAgentLearner):
	def __init__(self, alice, bob, sad):
		super().__init__(alice, bob, sad)


class IndependentA2CLearner(MultiAgentLearner):
	def __init__(self, alice, bob, sad, use_central_critic):
		super().__init__(alice, bob, sad)
		self.use_central_critic = use_central_critic
		if use_central_critic:
			assert alice.critic is bob.critic
			self.critic = alice.critic

	def update_from_episode(self, episode):
		if self.use_central_critic:
			c1, c2, a1, a2, payoff = episode
			p1_info_state = c1
			p2_info_state = sad_transform(self.alice, episode, self.sad)
			full_info1 = (c1, c2)
			full_info2 = (c1, c2, a1)
			ac_update(self.alice.actor, self.critic, p1_info_state, full_info1, a1)
			ac_update(self.bob.actor, self.critic, p2_info_state, full_info2, a2)
			full_info_episode = [(full_info1, a1, 0), (full_info2, a2, payoff)]
			q_update(self.critic, full_info_episode)
		else:
			super().update_from_episode(episode)

def avd_update(alice, bob, p1_is, p2_is, a1, a2, r, p1_is_, p2_is_, is_done):
	l1q1 = alice.params[p1_is][a1]
	l2q1 = bob.params[p2_is][a2]
	if is_done:
		q_next = 0
	else:
		q_next = max(alice.params[p1_is_]) + max(bob.params[p2_is_])
	alice.params.update_params(p1_is, a1, r + q_next - l2q1)
	bob.params.supdate_params(p2_is, a2, r + q_next - l1q1)

def q_update(params, episode):
	states, actions, rewards = zip(*episode)
	values = []
	for s, a, r, s_ in zip(states, actions, rewards, states[1:]):
		params.update_params(s, a, r + max(params.vals[s_]))
	params.update_params(states[-1], actions[-1], rewards[-1])

def ac_update(actor, critic, a_state, c_state, action):
	critic.add_state(c_state, len(actor.params.vals[a_state]))
	policy = softmax(actor.params.vals[a_state])
	q_vals = critic.vals[c_state]
	actor.params.update_params(a_state, action, q_vals[action] - sum(q * p for q, p in zip(q_vals, policy)))

def sad_transform(alice, context, sad):
	c1, c2, a1 = context[:3]
	info_state = (c2, a1)
	if not sad:
		return info_state
	greedy_a1 = alice.actor.act_greedily(c1)
	if sad == 'sad':
		signal = greedy_a1
	elif sad == 'bsad':
		signal = a1 == greedy_a1
	elif sad == 'asad':
		signal = None if a1 == greedy_a1 else greedy_a1
	elif sad == 'psad':
		signal = None if a1 == greedy_a1 else c1
	return (*info_state, signal)

def make_ql(init_lr, init_epsilon, num_episodes, num_evals):
	return QLearner(ValueActor(ValueParameters(init_lr), init_epsilon), num_episodes, num_evals)

def make_hql(init_lr1, init_lr2, init_epsilon, num_episodes, num_evals):
	return QLearner(ValueActor(DoubleLrValueParameters(init_lr1, init_lr2), init_epsilon), num_episodes, num_evals)

def make_reinforce(init_lr, num_episodes, num_evals):
	return ReinforceLearner(PolicyActor(PolicyParameters(init_lr)), num_episodes, num_evals)

def make_a2c(pg_init_lr, ql_init_lr, num_episodes, num_evals):
	return A2CLearner(PolicyActor(PolicyParameters(pg_init_lr)), ValueParameters(ql_init_lr), num_episodes, num_evals)

def make_iql(init_lr, init_epsilon, sad, avd, num_episodes, num_evals):
	alice = make_ql(init_lr, init_epsilon, num_episodes, num_evals)
	bob = make_ql(init_lr, init_epsilon, num_episodes, num_evals)
	return IndependentQLearner(alice, bob, sad, avd)

def make_ihql(init_lr1, init_lr2, init_epsilon, sad, avd, num_episodes, num_evals):
	alice = make_hql(init_lr1, init_lr2, init_epsilon, num_episodes, num_evals)
	bob = make_hql(init_lr1, init_lr2, init_epsilon, num_episodes, num_evals)
	return IndependentQLearner(alice, bob, sad, avd)

def make_ireinforce(init_lr, sad, num_episodes, num_evals):
	alice = make_reinforce(init_lr, num_episodes, num_evals)
	bob = make_reinforce(init_lr, num_episodes, num_evals)
	return IndependentReinforceLearner(alice, bob, sad)

def make_ia2c(pg_init_lr, ql_init_lr, sad, central_critic, num_episodes, num_evals):
	alice = make_a2c(pg_init_lr, ql_init_lr, num_episodes, num_evals)
	bob = make_a2c(pg_init_lr, ql_init_lr, num_episodes, num_evals)
	if central_critic:
		alice.critic = bob.critic
	return IndependentA2CLearner(alice, bob, sad, central_critic)