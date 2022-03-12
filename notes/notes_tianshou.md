# Study Notes about Tianshou

[toc]

# Policy 

1. General:
   1. `gamma`: discount factor, should be in [0, 1]. Default to 0.99
   2. `gae_lambda`: the parameter for GAE, should be in [0, 1], default to 0.95
2. `BasePolicy`
   1. `__init__`: 
      1. `observation_space`
      2. `action_space`
      3. `action_scaling`:  whether to map actions from range [-1, 1] to range [action_spaces.low, action_spaces.high]. Default to True.
      4. `action_bound_method`: method to bound action to range [-1, 1], can be either "clip" (for simply clipping the action), "tanh" (for applying tanh squashing) for now, or empty string for no bounding. Default to "clip".
   2. `update`: `process_fn` -> `learn` -> `post_process_fn`
   3. input: `obs` , `state`, `info`
   4. output: next `state`
   5. `compute_episodic_return`: use generalized advantage estimator to calculate unnormalized_returns and advantages
   6. `compute_nstep_return`:  $G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +\gamma^n (1 - d_{t + n}) Q_{\mathrm{target}}(s_{t + n})$
   7. `_gae_return(v_s, v_s_, rew, end_flag, gamma, gae_lambda)`: 
      1. $\delta = r + V_{t+1}(s\prime) * \gamma - V_t(s)$
      2. $discount = (1-\text{end flag}) * (\gamma * \lambda_{gae})$
      3. for $i$ in range(len(rew) -1, -1, -1):
         1. $gae = \delta_i + discount[i] * gae$
         2. $returns[i] = gae$
      4. return $returns$
   8. `_n_step_return(rew, end_flag, target_q, indices, gamma, n_step)`:

## Model Free

### Gradient Based

1. `PGPolicy(BasePolicy)`: REINFORCE Algo
   1. `__init__`: 
      1. `model` (s->logits)  (**actor**)
      2. `optim`
      3. `dist_fn` :distribution class for computing the action
      4. `reward_normalization`: normalize estimated values to have std close to 1. Default to False.
      5. `lr_scheduler`
      6. `deterministic_eval`: whether to use deterministic action instead of stochastic action sampled by the policy. Default to False.
   2. `process_fnc`: Compute the discounted return for each transition $G_t = \sum_{i=t}^T \gamma^{i-t}r_i$
   3. `forward`: compute action over the given batch data
   4. `learn`: https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63 
2. `A2CPolicy(PGPolicy)`
   1. `__init__`:  Synchronous Advantage Actor-Critic
      1. `actor`: s-> logits
      2. `critic`: s -> V(s)
      3. `vf_coef`: weight for value loss, default to 0.5
      4. `ent_coef`: weight for entropy loss, default to 0.01
      5. `max_grad_norm`: clipping gradient in back propagation. Default to None.
      6. `gae_lamba`: param for GAE 
      7. `max_batchsize`: the maximum size of the batch when computing GAE, depends on the size of available memory and the memory cost of the model; should be as large as possible within the memory constraint. Default to 256.
   2. `process_fn`: uses `_compute_returns` method
   3. `_compute_returns`: compute returns and advantages.
   4. `learn`: get actor loss, critic loss, entropy loss (for exploration)
3. `NPGPolicy(A2CPolicy)`: Natural Policy Gradient
   1. `__init__`
      1. `advantage_normalization`: whether to do per mini-batch advantage normalization. Default to True
      2. `optim_critic_iters`: Number of times to optimize critic network per update. Default to 5.
      3. `actor_step_size`: ? . defaul to 0.5 .
   2. `process_fn`: 
      1. use `A2CPolicy`'s `process_fn` function first to add reward and advantage
      2. calculate the log probability of old action
   3. `learn`: get actor loss, critic loss and KL
   4. `_get_flat_grad`: get gradient in flat version
   5. `_get_from_flat_params(self, model, flat_params)`: set the model parameter as flat_params
   6. `_conjugate_gradients(self, minibatch, flat_kl_grad, nsteps=10, residual_tol=1e-10)`
   7. `_MVP(self, v, flat_kl_grad)`: Matrix vector product 
4. `TRPOPolicy(NPGPolicy)`: Trust Region Policy Optimization
   1. `__init__`:
      1. `max_kl`: max kl-divergence used to constrain each actor network update. Default to 0.01.
      2. `backtrack_coeff`: Coefficient to be multiplied by step size when constraints are not met. Default to 0.8.
      3. `max_backtracks`:  Max number of backtracking times in linesearch. Default to 10.
   2. `learn`
      1. similar to `NPGPolicy`'s `learn` method but with line search added.
5. `PPOPolicy(A2CPolicy)`
   1. `__init__`
      1. `eps_clip`: $\epsilon$ in $L_{CLIP}$ in the original paper. Default to 0.2.
      2. `dual_clip`: a parameter c mentioned in arXiv:1912.09729 Equ. 5, where c > 1 is a constant indicating the lower bound. Default to 5.0 (set None if you do not want to use it).
      3. `value_clip`:  a parameter mentioned in arXiv:1811.02553v3 Sec. 4.1. Default to True.
      4. `recompute_advantage`:  whether to recompute advantage every update repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5. Default to False.
   2. `learn`
6. `DiscreteSACPolicy(SACPolicy)`

  

### Value Based

1. `DQNPolicy(BasePolicy)`

2. `QRDQNPolicy(DQNPolicy)`

3. `FQFPolicy(QRDQNPolicy)`

4. `IQNPolicy(QRDQNPolicy)`

   

5. `DDPGPolicy(BasePolicy)`

6. `SACPolicy(DDPGPolicy)`

7. `TD3Policy(DDPGPolicy)`

   

8. ` C51Policy(DQNPolicy)`

9. `RainbowPolicy(C51Policy)`



## Model Based

1. `ICMPolicy(BasePolicy)`
2. `PSRLModel(object)`

## Imitation

1. `ImitationPolicy(BasePolicy)`
2. `BCQPolicy(BasePolicy)`
3. `CQLPolicy(SACPolicy)`
4. `DiscreteBCQPolicy(DQNPolicy)`
5. `DiscreteCQLPolicy(QRDQNPolicy)`
6. `DiscreteCRRPolicy(PGPolicy)`

## MultiAgent

1. `MultiAgentPolicyManager(BasePolicy)`



# Trainer

1. `onpolicy_trainer`

   1. parameters:

      1. `policy`
      2. `train_collector`
      3. `test_collector`
      4. `max_epoch`
      5. `step_per_epoch`: the number of transitions collected per epoch
      6. `repeat_per_collect`: the number of repeat time for policy learning. Eg. set it to 2 means the policy needs to learn each given batch data twice
      7. `episode_per_test`
      8. `batch_size`: the batch size of sample data, which is going to feed in the policy network
      9. `step_per_collect`: the number of transitions the collector would collect before the network update. Only either one of step_per_collect and episode_per_collect can be specified.
      10. `episode_per_collect`: the number of episodes the collector would collect before the network update. Only either one of step_per_collect and episode_per_collect can be specified.
      11. `train_fn`: a hook called at the beginning of training in each epoch.  It can be used to perform custom additional operations, with the signature ``f(num_epoch: int, step_idx: int) -> None``.
      12. `test_fn`
      13. `save_fn`: a hook called when the undiscounted average mean reward in evaluation phase gets better, with the signature `f(policy: BasePolicy) -> None`
      14. `save_checkpoint_fn`: a function to save training process, with the signature ``f(epoch: int, env_step: int, gradient_step: int) -> None``; you can save whatever you want.
      15. `resume_from_log`:  resume env_step/gradient_step and other metadata from existing tensorboard log. Default to False.
      16. `stop_fn`:  a function with signature ``f(mean_rewards: float) -> bool``, receives the average undiscounted returns of the testing result, returns a boolean which indicates whether reaching the goal.
      17. `reward_metric`:  a function with signature ``f(rewards: np.ndarray with shape (num_episode, agent_num)) -> np.ndarray with shape (num_episode,)``, used in multi-agent RL. We need to return a single scalar for each episode's result to monitor training in the multi-agent RL setting. This function specifies what is the desired metric, e.g., the reward of agent 1 or the average reward over all agents.
      18. `logger`
      19. `verbose`
      20. `test_in_train`: whether to test in the training phase. Default to True.

   2. key code:

      1. ```                python
         losses = policy.update(
             sample_size=0,
             buffer=train_collector.buffer,
             batch_size=batch_size,
             repeat=repeat_per_collect
         )
         ```

2. `offpolicy_trainer`

   1. parameters:

      1. no `episode_per_collect`
      2. `update_per_step` : the number of times the policy network would be updated per transition after (step_per_collect) transitions are collected, e.g., if update_per_step set to 0.3, and step_per_collect is 256, policy will be updated round(256 * 0.3 = 76.8) = 77 times after 256 transitions are collected by the collector. Default to 1.

   2. key code:

      1. ```                python
         losses = policy.update(
             sample_size=batch_size,
             buffer=train_collector.buffer
         )
         ```

3. `offline_trainer`

   1. parameter:
      1. `update_per_epoch`:  the number of policy network updates, so-called gradient steps, per epoch.

