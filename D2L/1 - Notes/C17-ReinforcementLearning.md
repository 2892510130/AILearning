## Chapter 17 : Reinforcement Learning
- Markov Decision Process
  - $\mathcal S, \mathcal A$ are states and actions a robot can take, when taking an action, states after may not
be deterministic, it has a probability. We use a transition function $T: \mathcal S \times \mathcal A \times \mathcal S \rightarrow [0, 1]$ to denote this, $T(s, a, s') = P(s' | s,a)$ is the probability of reaching s' given $s$ and $a$. For $\forall s \in \mathcal S$ and $\forall a \in \mathcal A$, $\sum_{s^{'}\in S}T(s, a, s^{'}) = 1$. Reward $r:\mathcal S \times \mathcal A \rightarrow \mathbb R$, $r(s,a)$.
  - This can build a MDP, $(\mathcal S, \mathcal A, \mathcal T, r)$, and a trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, s_2, a_2, r_2, \ldots)$. The *return* of a trajectory is the total reword $R(\tau) = \sum_t r_t$, the goal of reinforcement learning is to find a trajectory that has the largest return. The trajectory might be infinite, so in order for a meaningful formular of its return, we introduce a discount factor $\gamma < 1$, $R(\tau) = \sum_{t=0}^{\infty}\gamma^tr_t$. For large $\gamma$, the robot is encouraged to explore, for small one to take a short trajetory to goal.
  - Markov system only depend on current state and action, not the history one (but we can always augment the system).
    
- Value Iteration
  - *Value function* is the value of a state, from that state, the expected sum reward (return). *Action Value function* is the value of an action at state s, from that state, take that action, the expected sum reward (return). And $V^{\pi}(s) = \sum_{a \in \mathcal A}\pi(a\mid s) Q^{\pi}(s,a)$.
  - Policy $\pi(s) = \pi(a\mid s)$, and $\sum_a \pi(a \mid s) = 1$. Given a policy, we may get a lot of trajectories, the average return of those are $$V^{\pi}(s_0) = E_{a_t \sim \pi(s_t)}[R(\tau)] = E_{a_t \sim \pi(s_t)}\left[ \sum_{t=0}^{\infty}\gamma^tr(s_t, a_t) \right]$$ this is the *value function* of policy $\pi$. Given two probability (one for choose the action, one for take the action), we can write the probability of a certain trajectory as $$P(\tau) = \pi(a_0\mid s_0) \cdot P(s_1 \mid s_0, a_0) \cdot \pi(a_1\mid s_1) \cdot P(s_2 \mid s_1, a_1) \cdots$$ So if we divide the trajectory into 2 parts, $s_0$ and $\tau^{'}$, we get $$R(\tau) = r(s_0, a_0) + \gamma \sum_{t=1}^{\infty}\gamma^{t-1}r(s_t, a_t) = r(s_0, a_0) + \gamma R(\tau^{'})$$
And the expectaion of this is (the optimal policy is the argmax of this):
$$
\begin{array}{lll}
V^{\pi}(s_0) &=& E_{a_t \sim \pi(s_t)}[r(s_0, a_0) + \gamma R(\tau^{'})] \\
             &=& E_{a_0 \sim \pi(s_0)}[r(s_0, a_0)] + E_{a_0 \sim \pi(s_0)}\left[E_{s_1 \sim P(s_1 \mid a_0, s_0)}[E_{a_t \sim \pi(s_t)}[\gamma R(\tau^{'})]]\right] \\ 
             &=& E_{a_0 \sim \pi(s_0)}[r(s_0, a_0)] + E_{a_0 \sim \pi(s_0)}\left[\gamma E_{s_1 \sim P(s_1 \mid a_0, s_0)}[V^{\pi}(s_1)]\right] \\
             &\rightarrow & \\
V^{\pi}(s)   &=& \sum_{a \in \mathcal A}\pi(a\mid s)\left[r(s,a) + \gamma \sum_{s^{'}}P(s^{'} \mid s, a)V^{\pi}(s^{'})\right], \forall s \in S \\
             &\rightarrow & r(s,a) = \sum_r p(r \mid s,a)r = \sum_r\sum_{s^{'}}p(r,s^{'} \mid s,a)r =  \sum_{s^{'}}p(s^{'} \mid s,a)r(s^{'})\\
             &=& \sum_{a \in \mathcal A}\pi(a\mid s)\sum_{s^{'} \in \mathcal S}P(s^{'} \mid s, a)\left[r(s^{'}) + \gamma V^{\pi}(s^{'})\right]
\end{array}$$
    Last arrow please refer to *Math of Reinforcement Learning*, if we assume reward only depend on next state. Note that here maybe we should write $\mathcal A$ as $\mathcal A(s)$, and $r$ depends on $s, a$, but for simplicity (and without error) we drop them.
  - *Action Value* (start at $s_0$ but action is fixed to $a_0$):
$$
\begin{array}{lll}
Q^{\pi}(s_0, a_0) &=& r(s_0, a_0) + E_{a_t \sim \pi(s_t)} \left[ \sum_{t=1}^{\infty}\gamma^{t}r(s_t, a_t) \right] \\
                  &\rightarrow & \\
Q^{\pi}(s, a)     &=& r(s,a) + \gamma \sum_{s^{'}}P(s^{'} \mid s, a)\sum_{a^{'}}\pi(s^{'} \mid a^{'})Q^{\pi}(s^{'}, a^{'}), \forall s \in S, a \in \mathcal A \\
                  &=& r(s,a) + \gamma \sum_{s^{'}}P(s^{'} \mid s, a)V^{\pi}(s^{'}), \forall s \in S, a \in \mathcal A
\end{array}
$$
We can also write it into the form of $r(s^{'})$, which is the form Gym takes.
  - Value Iteration: initialize the value function to arbitrary values $V_0(s), \forall s \in S$. If the policy is deterministic, then at $k^{th}$ iteration (max will return one action, which means the action policy is deterministic): $$V_{k+1}(s) = \max_{a \in \mathcal A} \left \{ r(s,a) + \gamma \sum_{s^{'}}P(s^{'} \mid s, a)V_{k}(s^{'}) \right \}, \forall s \in \mathcal S$$ And $V^*(s) = \lim_{k \rightarrow \infty} V_k(s)$. The same iteration can be written with action value: $$Q_{k+1}(s,a) = r(s,a) + \gamma \max_{a^{'} \in A}\sum_{s^{'}}P(s^{'} \mid s, a)Q_k(s^{'}, a^{'})$$
  - Policy Evaluation: if the policy is not deterministic, it is stochastic, we can also get:
$$V_{k+1}^{\pi}(s) = \sum_{a \in \mathcal A}\pi(a \mid s) \left [ r(s,a) + \gamma \sum_{s^{'} \in \mathcal S}P(s^{'} \mid s, a)V_{k}^{\pi}(s^{'}) \right ], \forall s \in \mathcal S$$

- Q-Learning
  - Using the robot's policy $\pi_e(a \mid s)$, we can approximate the value iteration:
    $$\hat Q = \min_Q \frac{1}{nT}\sum_{i=1}^n\sum_{t=0}^T\left(Q(s_t^i, a_t^i)-r(s_t^i, a_t^i)-\gamma \max_{a^{'}}Q(s_{t+1}^i, a^{'})\right)^2 = \min_Q \mathscr l(Q)$$
    Using gradient descent:
    $$
    \begin{array}{lll}
    Q(s_t^i, a_t^i) &\leftarrow& Q(s_t^i, a_t^i) + \alpha \nabla_{Q(s_t^i, a_t^i)} \mathscr l (Q) \\
                    &=         & (1-\alpha)Q(s_t^i, a_t^i) +\alpha\left(r(s_t^i, a_t^i) + \gamma \max_{a^{'}}Q(s_{t+1}^i, a^{'})\right)
    \end{array}
    $$
    In order to stop at terminal state, we update the formular:
    $$Q(s_t^i, a_t^i) = (1-\alpha)Q(s_t^i, a_t^i) +\alpha\left(r(s_t^i, a_t^i) + \gamma (1 - \mathbb 1_{s_{t+1}^i \text{is terminal}}) \max_{a^{'}}Q(s_{t+1}^i, a^{'})\right)$$
    Then we get $\hat \pi(s) = \arg \max_a \hat Q(s,a)$.
  - In order to explore well, with current estimate of Q, set the policy to (*$\epsilon$-greedy exploration policy*):
    $$ \pi_e(a \mid s) = \left \{ 
    \begin{array}{lll}
    \arg \max_{a^{'}} \hat Q(s, a^{'}) & \text{with prob.} ~1 - \epsilon\\
    \text{uniform}(\mathcal A)         & \text{with prob.} ~\epsilon
    \end{array}
    \right .$$
    Other choise is *softmax exploration policy*, $T$ is called *temperature*:
    $$\pi_e(a \mid s) = \frac{e^{\hat Q(s,a)/T}}{\sum_{a^{'}}e^{\hat Q(s,a^{'})/T}}$$
