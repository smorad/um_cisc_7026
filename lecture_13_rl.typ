#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#import "@preview/touying:0.5.3": *
#import themes.university: *
#import "common_touying.typ": *
#import "@preview/cetz:0.3.1"
#import "@preview/fletcher:0.5.1" as fletcher: node, edge

#set math.vec(delim: "[")
#set math.mat(delim: "[")

// TODO: Use R(s, a) so terminal states make sense
// TODO: Fix proof from infinite return to one-step return so it is more clear

// Lecture 1
#let rl_summary = [An *agent* shall *observe* and *act* within an *environment*, according to a *policy*. In RL, we aim to learn a *policy* that maximizes a cumulative *reward*.]

#let rl_terms = [#text(size: 20pt)[
  - *Agent* is a being with the capability to observe, think, and act in an environment (i.e., eyes, a brain, and a body) #pause

  - *Environment* is where the agent lives. It is governed by hidden rules that determine how the environment evolves. #pause

  - *Observations* ($o$) are sensory information that the agent receives from the environment #pause
    
  - *Actions* ($a$) produced by the agent change the environment #pause

  - *Policy* ($pi$) is the "brain". It decides which actions to take based on the observations. #pause
  
  - *State* ($s$) is the description of the environment at a specific timestep #pause
  
  - *Reward* ($r$) is an objective that we define. It determines what policy the agent will learn.
  ]
]

#let simple_agent_env_diagram = canvas(length: 0.8cm, {
    import draw: *
    rect((0,0), (4,2), name: "policy")
    rect((rel: (-2,-2), to: "policy.west"),(rel: (2,2), to: "policy.east"), name: "agent")
    content((rel: (0,1), to: "agent.north"), [Agent])

    rect((rel: (-3, -2), to: "agent.south"), (rel: (3, -4), to: "agent.south"), name: "trans")
    rect((rel: (-4, -2), to: "trans.center"), (rel: (4, 2), to: "trans.center"), name: "env")
    content((rel: (0,-1), to: "env.south"), [Environment])

    line((rel: (0, 0), to: "trans.west"), (rel: (-2, 0), to: "trans.west"))
    line((rel: (-2, 0), to: "trans.west"), (rel: (-1, 0), to: "agent.west"))
    content((rel: (-3,3), to: "trans.west"), [$s_t$])
    line((rel: (-1, 0), to: "agent.west"), (rel: (0, 0), to: "policy.west"), mark: (end: ">", size: 0.3))

    line((rel: (0, 0), to: "trans.west"), (rel: (-3, 0), to: "trans.west"), mark: (end: ">", size: 0.3))
    content((rel: (-4,0), to: "trans.west"), [$r_t$])

    line((rel: (0, 0), to: "policy.east"), (rel: (3, 0), to: "policy.east"))
    line((rel: (3, 0), to: "policy.east"), (rel: (2, 0), to: "trans.east"))
    line((rel: (2, 0), to: "trans.east"), (rel: (0, 0), to: "trans.east"), mark: (end: ">", size: 0.3))
    content((rel: (6,1), to: "env.north"), [$a_t$])
    
    content((rel: (0,-5), to: "trans.south"), [ 
      #text(size: 20pt)[
        
        $s_t$: state, 
        $a_t$: action,
        $r_t$: reward
    ]
    ])

  })

#let env_diagram = canvas(length: 0.8cm, {
  import draw: *
  rect((0,0), (4,2), name: "policy")
  content((rel: (0,0), to: "policy.center"), [$pi$])
  rect((rel: (-2,-2), to: "policy.west"),(rel: (2,2), to: "policy.east"), name: "agent")
  content((rel: (0,1), to: "agent.north"), [Agent])

  rect((rel: (-3, -2), to: "agent.south"), (rel: (3, -4), to: "agent.south"), name: "trans")
  content((rel: (0,0), to: "trans.center"), [$T$])
  rect((rel: (-4, -2), to: "trans.center"), (rel: (4, 2), to: "trans.center"), name: "env")
  content((rel: (0,-1), to: "env.south"), [Environment])

  line((rel: (0, 0), to: "trans.west"), (rel: (-2, 0), to: "trans.west"))
  line((rel: (-2, 0), to: "trans.west"), (rel: (-1, 0), to: "agent.west"))
  content((rel: (-3,3), to: "trans.west"), [$s_t$])
  line((rel: (-1, 0), to: "agent.west"), (rel: (0, 0), to: "policy.west"), mark: (end: ">", size: 0.3))

  line((rel: (0, 0), to: "trans.west"), (rel: (-3, 0), to: "trans.west"), mark: (end: ">", size: 0.3))
  content((rel: (-4,0), to: "trans.west"), [$r_t$])

  line((rel: (0, 0), to: "policy.east"), (rel: (3, 0), to: "policy.east"))
  line((rel: (3, 0), to: "policy.east"), (rel: (2, 0), to: "trans.east"))
  line((rel: (2, 0), to: "trans.east"), (rel: (0, 0), to: "trans.east"), mark: (end: ">", size: 0.3))
  content((rel: (6,1), to: "env.north"), [$a_t$])
  
  content((rel: (0,-5), to: "trans.south"), [ 
    #text(size: 20pt)[
      
      $s_t$: state, 
      $a_t$: action, \
      $r_t$: reward,
      $pi$: policy, \
      $T$: transition fn
  ]
  ])
})

#let agent_env_diagram = canvas(length: 0.8cm, {
    import draw: *
    rect((0,0), (4,2), name: "policy")
    content((rel: (0,0), to: "policy.center"), [$pi$])
    rect((rel: (-2,-2), to: "policy.west"),(rel: (2,2), to: "policy.east"), name: "agent")
    content((rel: (0,1), to: "agent.north"), [Agent])

    rect((rel: (-3, -2), to: "agent.south"), (rel: (3, -4), to: "agent.south"), name: "trans")
    content((rel: (0,0), to: "trans.center"), [$T$])
    rect((rel: (-4, -2), to: "trans.center"), (rel: (4, 2), to: "trans.center"), name: "env")
    content((rel: (0,-1), to: "env.south"), [Environment])

    line((rel: (0, 0), to: "trans.west"), (rel: (-2, 0), to: "trans.west"))
    line((rel: (-2, 0), to: "trans.west"), (rel: (-1, 0), to: "agent.west"))
    content((rel: (-3,3), to: "trans.west"), [$s_t$])
    line((rel: (-1, 0), to: "agent.west"), (rel: (0, 0), to: "policy.west"), mark: (end: ">", size: 0.3))

    line((rel: (0, 0), to: "trans.west"), (rel: (-3, 0), to: "trans.west"), mark: (end: ">", size: 0.3))
    content((rel: (-4,0), to: "trans.west"), [$r_t$])

    line((rel: (0, 0), to: "policy.east"), (rel: (3, 0), to: "policy.east"))
    line((rel: (3, 0), to: "policy.east"), (rel: (2, 0), to: "trans.east"))
    line((rel: (2, 0), to: "trans.east"), (rel: (0, 0), to: "trans.east"), mark: (end: ">", size: 0.3))
    content((rel: (6,1), to: "env.north"), [$a_t$])
    
    content((rel: (0,-5), to: "trans.south"), [ 
      #text(size: 20pt)[
        
        $s_t$: state, 
        $a_t$: action, \
        $r_t$: reward,
        $pi$: policy, \
        $T$: transition fn
    ]
    ])

  })

#let trans_diagram = canvas(length: 0.8cm, {
  import draw: *
  circle((6, -2), name: "bottom")
  content((rel: (0,0), to: "bottom"), $s_(1)$)

  circle((0, 0), name: "start")
  content((rel: (0,0), to: "start"), $s_(0)$)

  circle((6, 2), name: "top")
  content((rel: (0,0), to: "top"), $s_(1)$)


  circle((12, -2), name: "bottom2")
  content((rel: (0,0), to: "bottom2"), $s_(2)$)
  
  circle((12, 2), name: "top2")
  content((rel: (0,0), to: "top2"), $s_(2)$)

  content((rel: (3,0), to: "top2"), $dots$)
  content((rel: (3,0), to: "bottom2"), $dots$)


  line((rel: (0, 0), to: "start.east"), (rel: (0, 0), to: "bottom.west"), mark: (end: ">", size: 0.3))
  content((rel: (2,2), to: "start"), $a_0 = 1$)

  line((rel: (0, 0), to: "start.east"), (rel: (0, 0), to: "top.west"), mark: (end: ">", size: 0.3))
  content((rel: (2,-2), to: "start"), $a_0 = 2$)


  line((rel: (0, 0), to: "bottom.east"), (rel: (0, 0), to: "bottom2.west"), mark: (end: ">", size: 0.3))
  content((rel: (3,-1), to: "bottom"), $a_(1)$)

  line((rel: (0, 0), to: "top.east"), (rel: (0, 0), to: "top2.west"), mark: (end: ">", size: 0.3))
  content((rel: (3,1), to: "top"), $a_(1)$)


  line((rel: (0, 0), to: "bottom2.east"), (rel: (1, 0), to: "bottom2.east"), mark: (end: ">", size: 0.3))
  line((rel: (0, 0), to: "top2.east"), (rel: (1, 0), to: "top2.east"), mark: (end: ">", size: 0.3))
})


#let simple_agent_env_diagram = canvas(length: 0.8cm, {
    import draw: *
    rect((0,0), (4,2), name: "policy")
    rect((rel: (-2,-2), to: "policy.west"),(rel: (2,2), to: "policy.east"), name: "agent")
    content((rel: (0,1), to: "agent.north"), [Agent])

    rect((rel: (-3, -2), to: "agent.south"), (rel: (3, -4), to: "agent.south"), name: "trans")
    rect((rel: (-4, -2), to: "trans.center"), (rel: (4, 2), to: "trans.center"), name: "env")
    content((rel: (0,-1), to: "env.south"), [Environment])

    line((rel: (0, 0), to: "trans.west"), (rel: (-2, 0), to: "trans.west"))
    line((rel: (-2, 0), to: "trans.west"), (rel: (-1, 0), to: "agent.west"))
    content((rel: (-3,3), to: "trans.west"), [$s_t$])
    line((rel: (-1, 0), to: "agent.west"), (rel: (0, 0), to: "policy.west"), mark: (end: ">", size: 0.3))

    line((rel: (0, 0), to: "trans.west"), (rel: (-3, 0), to: "trans.west"), mark: (end: ">", size: 0.3))
    content((rel: (-4,0), to: "trans.west"), [$r_t$])

    line((rel: (0, 0), to: "policy.east"), (rel: (3, 0), to: "policy.east"))
    line((rel: (3, 0), to: "policy.east"), (rel: (2, 0), to: "trans.east"))
    line((rel: (2, 0), to: "trans.east"), (rel: (0, 0), to: "trans.east"), mark: (end: ">", size: 0.3))
    content((rel: (6,1), to: "env.north"), [$a_t$])
    
    content((rel: (0,-5), to: "trans.south"), [ 
      #text(size: 20pt)[
        
        $s_t$: state, 
        $a_t$: action,
        $r_t$: reward
    ]
    ])
})

#let q_function_return = $ Q_pi (s_0, a_0) = bb(E) [r_0 mid(|) a_0] + bb(E)[sum_(t=1)^oo gamma^(t) r_t mid(|) a_t tilde pi(s_t)] $
#let qstar_function_return = $ Q_(pi_*) (s_0, a_0) = bb(E) [r_0 mid(|) a_0] + bb(E)[sum_(t=1)^oo gamma^(t) r_t mid(|) a_t tilde pi_(*)(s_t)] $
#let qstar_function_return2 = $ Q_* (s_0, a_0) = bb(E) [r_0 mid(|) a_0] + bb(E)[sum_(t=1)^oo gamma^(t) r_t mid(|) a_t tilde pi_(*)(s_t)] $
#let qstar_action_policy = $ pi_* (s) = op("argmax", limits: #true)_(a in A) Q_* (s, a) $
#let qstar_argmax_return = $ Q_(*) (s_0, a_0) = bb(E) [r_0 mid(|) a_0] + bb(E)[sum_(t=1)^oo gamma^(t) r_t mid(|) a_t tilde pi_* (s_t)] $
#let qstar_argmax_intractable = $ Q_(*) (s_0, a_0) = bb(E) [r_0 mid(|) a_0] + bb(E)[underbrace(sum_(t=1)^oo gamma^(t) r_t, "Very annoying") mid(|) a_t tilde pi_* (s_t)] $
#let standard_q = $ Q (s, a) = r + gamma dot max_{a' in A} Q (s', a') $
#let reward_dist = $ R(s_(t+1)) T(s_(t+1) mid(|) s_(t),a_(t)) pi(a_(t) mid(|) s_(t)) $
#let reward_dist_labeled = $ underbrace(R(s_(t+1)), "reward fn") overbrace(T(s_(t+1) mid(|) s_(t),a_(t)), "state trans. probs") underbrace(pi(a_(t) mid(|) s_(t)), "action probs") $

#let ml_train_loop = ```python
dataset = load_dataset()
model = nn.Module(dataset.x.size, dataset.y.size)
theta = model.init(seed=0) # Functional

for update in range(num_updates):
  train_data = dataset.sample()
  theta = train(model, theta, train_data) # Functional
  metrics = evaluate(model, theta, dataset.val_set)
```

#let train_loop = ```python
env = LunarLander()
Q = nn.Module(env.state_space, env.action_space)
theta = Q.init(seed=0)
pi = policy(Q, theta)

for update in range(num_updates):
  collected_data = collect_training_data(env, pi)
  dataset += collected_data
  train_data = dataset.sample()
  theta = train(Q, theta, train_data)
  metrics = evaluate(env, pi)
```

#let train_loop_pi = ```python
env = LunarLander()
Q = nn.Module(env.state_space, env.action_space)
theta = Q.init(seed=0)
pi, pi_e = max_q(Q, theta), e_greedy(Q, theta)

for update in range(num_updates):
  collected_data = collect_training_data(env, pi_e)
  dataset += collected_data
  train_data = dataset.sample()
  theta = train(Q, theta, train_data)
  metrics = evaluate(env, pi)
```

#let train_loop_target = ```python
env = LunarLander()
Q = nn.Module(env.state_space, env.action_space)
theta, psi = Q.init(seed=0), Q.init(seed=0)
pi, pi_e = max_q(Q, theta), e_greedy(Q, theta)

for update in range(num_updates):
  collected_data = collect_training_data(env, pi_e)
  dataset += collected_data
  train_data = dataset.sample()
  theta = train(Q, theta, psi, train_data)
  metrics = evaluate(env, pi)
```

#show: university-theme.with(
  aspect-ratio: "16-9",
  // config-common(handout: true),
  config-info(
    title: [Reinforcement Learning],
    subtitle: [CISC 7026 - Introduction to Deep Learning],
    author: [Steven Morad],
    //date: datetime.today(),
    institution: [University of Macau],
    logo: image("figures/common/bolt-logo.png", width: 4cm)
  ),
  header-right: none,
  header: self => utils.display-current-heading(level: 1)
)

#title-slide()

== Outline <touying:hidden>

#components.adaptive-columns(
    outline(title: none, indent: 1em, depth: 1)
)
  
==
  *Lecture Goal*: Provide a proper understanding of the theoretical foundations of reinforcement learning
==
  #strike[*Lecture Goal*: Provide a proper understanding of the theoretical foundations of reinforcement learning] #pause
  
  *Lecture Goal*: Give you enough information to begin learning RL on your own

= What is RL?
  /*+ What is RL?
  + Markov decision processes
  + Agents/policies
  + Derive and define Q learning
  + Resources*/

==
How does reinforcement learning (RL) differ from supervised or unsupervised learning? #pause

In supervised and unsupervised learning, we know the answer #pause

$ f(bold(x), bold(theta)) = bold(y) $ #pause

$ f^(-1)(f(bold(x), bold(theta)), bold(theta)) = bold(x) $ #pause

In reinforcement learning, we do not know the answer! #pause

$ f(bold(x), bold(theta)) = ? $ #pause

What does this mean?

==
*Example:* You train a model $f$ to play chess #pause

$ f: X times Theta |-> Y $ #pause

$ X in "Position of pieces on the board" $ #pause

$ Y in "Where to put piece" $ 

==
#side-by-side[$ X in "Position of pieces on the board" $][
    $ Y in "Where to put piece" $ 
] #pause

#cimage("figures/lecture_13/chess.png", height: 85%)

==
#cimage("figures/lecture_13/chess.png", height: 85%) #pause

#side-by-side[What is the correct answer? #pause][We do not know the answer]//[But RL can tell us!]

==
#cimage("figures/lecture_13/chess.png", height: 85%) #pause

#side-by-side[No answer, no supervised learning#pause][RL can train without the answer!]//[But RL can tell us!]

==
#cimage("figures/lecture_13/chess.png", height: 85%) #pause

#side-by-side[An answer gives us just one move #pause][We need many moves to win]

==
RL gives us the best *sequence* of moves to achieve a result #pause

- Win a game of chess #pause
- Drive a customer to the store #pause
- Cook a tasty meal #pause
- Treat a sick patient #pause
- Prevent climate change #pause
- Reduce human suffering #pause
- Find your own purpose (achieve conciousness)

==
Real applications of RL: #pause
#grid(
columns: 2,
column-gutter: 1em,
row-gutter: 2em,
link("https://www.youtube.com/watch?v=Zeyv1bN9v4A"), "GT",
link("https://www.youtube.com/watch?v=kopoLzvh5jY&t=1s"), "H&S",
link("https://www.youtube.com/watch?v=eHipy_j29Xw"), "DoTA"
)


==
  Other real applications of RL: #pause

  - Autonomous vehicles #pause
  - Video game NPCs #pause
  - Behavior modeling in psychology/ecology/biology #pause
  - Material and drug design #pause
  - Finance #pause
  - Alignment in large language models #pause
    - Artificial General Intelligence? #pause
  - Anywhere with cause and effect
    - Where you *change* the world by *interacting* with it

== 
RL is more complex than supervised learning #pause

Instead of a model and dataset, we have an *agent* and *environment* #pause

#align(center + horizon, 
grid(
    columns: 2,
    column-gutter: 0.1fr,
    cimage("figures/lecture_13/person.png", height: 70%),
    cimage("figures/lecture_13/earth.png", height: 60%),
    "Agent",
    "Environment"
)
)

==
#side-by-side(align: left)[
    The agent receives a positive reward for doing good #pause

    And a negative reward for doing bad #pause
][
    #cimage("figures/lecture_13/dog.jpg", height: 85%) #pause

]

Eventually, the agent only does good behaviors

==
Humans learn by reinforcement learning too #pause

#side-by-side(align: left)[
    #cimage("figures/lecture_13/baby.jpg", height: 60%) #pause
][
    When the baby cries, they will receive hugs (reward) #pause

    So the baby will learn to cry to get more hugs! #pause

]

Note that "good" behavior is subjective! #pause

Enough about the agent, let us talk about the environment

==
#side-by-side(align: left)[#cimage("figures/lecture_13/chess.png", height: 80%) #pause][
The environment is the world that the agent lives in #pause

The environment is a collection of rules #pause

For example, each piece can only move in certain ways #pause

If two pieces touch, then one piece dies
]

==
For you, your environment is Macau! #pause

There are a set of rules that govern what you can do #pause
- You follow the rules of physics (you cannot fly) #pause
- You follow the laws (cannot steal) #pause
- You come to this specific location to attend lecture #pause
- You get good grades (your parents make rules too) #pause

==
The *state* describes the agent in the environment #pause

If you are the agent, maybe your state contains: #pause
- Your physical location (x, y, z coordinates)
- The time #pause
- Who is in the room with you #pause
- If you are hungry or thirsty #pause

Now that you understand the agent, rewards, and environment, we will get more technical

==
  #side-by-side(align:left)[  
    #simple_agent_env_diagram #pause
  ][
    - The agent takes *actions* in the environment #pause
    - Actions change the environment *state*, producing an new state and *reward* #pause
    - The cycle continues for $t=0, 1, dots$ #pause
    - Goal is to maximize the *cumulative reward*
      - Sum of rewards over *all* timestep
  ]

==
  //The environment defines the task we are trying to solve. #pause

  By definition, RL solves *Markov Decision Processes (MDPs)* #pause

  To solve a problem, we must convert it into an MDP #pause

  We call the MDP the environment #pause
  
  How you structure your problem is *critical* -- more important than which algorithms you use, how much compute you have, etc. #pause

  Let us formally introduce the MDP


= Markov Decision Processes

==
  *Definition:* An MDP is a 5-tuple $(S, A, T, R, gamma)$ #pause
  
  $S$ is the set of states known as the *state space*. #pause

  $A$ is the set of actions known as the *action space* #pause

  $T: S times A arrow Delta S space$ The *state transition function*. #pause //What is the next state, given the previous state and action? This can be probabilistic. #pause

  $R: S arrow bb(R)$ is the *reward function*. #pause //Each time we change states, the MDP emits a scalar reward. #pause

  $gamma in [0, 1]$ is the *discount factor* #pause

  Let us briefly explain these terms.

==
    $S$ is the set of states known as the *state space*. #pause

    Recall that the environment is always changing #pause

    We need a way to describe what state the environment is in #pause

    If the environment is a table, the state space might describe the positions of all objects on the table 
    $ bold(s) = vec(delim: "[", x_1, y_1, x_2, y_2, dots.v) $


==
    $A$ is the set of actions known as the *action space* #pause

    What capabilities does the agent have? #pause

    For the table example, I can apply a force to a specific object on the table 
    
    $ bold(a) = vec(delim: "[", F_x, F_y, i) $

==
    #text(size:24pt)[
    $T: S times A arrow Delta S space$ The *state transition function*. #pause

    Sometimes called "transition dynamics", "state transition matrix", etc #pause

    "Rules" of the environment, determine the (stochastic) evolution of the environment #pause

    #side-by-side(align: left)[$ T( underbrace( 
    vec(delim: "[", x_1, y_1, x_2, y_2, dots.v), "state"
    ), space 
    underbrace(
      vec(delim: "[", F_x, F_y, i), "action"
    )
  ) =  
    underbrace(Delta vec(delim: "[", x_1, y_1, x_2, y_2, dots.v), "next state dist."
    ) $#pause][

    *Markov* decision process because transition dynamics are *conditionally independent* of past states and actions

    $ T(s_t, a_t | s_(t-1), a_(t-1), dots, s_(0), a_(0)) = T(s_t, a_t) $
    ]
    ]

// 15 mins?
==
    $R: S arrow bb(R)$ is the *reward function*. #pause

    Produces reward based on the state #pause

    Reward function determines agent behavior #pause

    +100 for pushing objects onto the floor, or +100 for pushing objects to the centre
==

  *Definition:* An MDP is a 5-tuple $(S, A, T, R, gamma)$ #pause
  
  $S$ is the set of states known as the *state space*. #pause

  $A$ is the set of actions known as the *action space* #pause

  $T: S times A arrow Delta S space$ The *state transition function*. #pause //What is the next state, given the previous state and action? This can be probabilistic. #pause

  $R: S arrow bb(R)$ is the *reward function*. #pause //Each time we change states, the MDP emits a scalar reward. #pause

  $gamma in [0, 1]$ is the *discount factor*

==
    #side-by-side(align: left)[
    #cimage("figures/lecture_13/mario.png", width: 100%)
  ][
    #text(size: 25pt)[
    Super Mario Bros. is a video game about Mario, an Italian plumber #pause

    Mario can move and jump #pause

    Touching a goomba kills Mario #pause

    Mario can squish Goombas #pause

    ? blocks give you mushrooms #pause

    You collect coins and have a time limit and score #pause

    *Task:* Define Super Mario MDP
  ]
  ]


==
  #side-by-side(align: left)[
    #cimage("figures/lecture_13/mario.png", width: 100%)
  ][
    #text(size: 24pt)[
    *State Space ($S$)*? #pause

      - Mario position/velocity $(bold(r), dot(bold(r)))$ #pause
      - Score #pause
      - Number of coins collected #pause
      - The time remaining #pause
      - Which question blocks we opened #pause
      - Goomba position/velocity and squished/not squished
      #pause

      $S = {bb(R)^4, bb(Z)_+, bb(Z)_+, bb(Z)_+, {0,1}^m, bb(R)^(4 times k), {0,1}^k}$]
  ]
==

  #side-by-side()[
    #cimage("figures/lecture_13/mario.png", width: 100%)
  ][
    *State Space ($S$)*? #pause
    $[0, 1]^(2 times 256 times 240 times 3)$ #pause
    
    $ mat(
      vec(255, 0, 0), vec(170, 10, 50), dots; 
      vec(10, 100, 235), vec(200, 200, 35), dots;
      dots.v, , dots.down
  ), mat(
      vec(255, 0, 0), vec(170, 10, 50), dots; 
      vec(10, 100, 235), vec(200, 200, 35), dots;
      dots.v, , dots.down
  ) $ #pause

  Two images necessary to compute velocities! #pause

  $S = bb(Z)_(<255)^(2 times 256 times 240 times 3)$
  ]


// 24 mins?

==
  #side-by-side(align:top)[
    #cimage("figures/lecture_13/mario.png", width: 100%)
  ][
    *Action Space ($A$)*? #pause
      - Acceleration of Mario $dot.double(bold(r))$ #pause
        - But when playing Mario, we cannot explicitly set $dot.double(bold(r))$
  ]

==
  #side-by-side(align: left)[
    #cimage("figures/lecture_13/mario.png", width: 100%)
  ][
    *Action Space ($A$)*? #pause
      - The Nintendo controller has $A, B, arrow.t, arrow.b, arrow.l, arrow.r$ buttons #pause
        - $A = {
          A, B, arrow.t, arrow.b, arrow.l, arrow.r
        }$ #pause
          - Cannot represent multiple buttons at once #pause
        - $A = {0, 1}^6$ #pause
        - #text(size: 22pt)[${underbrace({0, 1, 2, 3, 4}, emptyset",direction") times underbrace({0, 1, 2, 3}, emptyset",a,b,a+b" )}$]
  ]

==
  #side-by-side(align: top + left)[
    #cimage("figures/lecture_13/mario.png", width: 100%)
  ][
      *Transition Function ($T$)*? #pause
        - $T(bold(s)_"pixel", arrow.r)$ #pause
          - Move the Mario pixels right, unless a wall #pause
          - Difficult to write down #pause
          - Deterministic

  ]

==
  #side-by-side(align: top + left)[
    #cimage("figures/lecture_13/mario.png", width: 100%)
  ][
      *Transition Function ($T$)*? #pause
        - $T(bold(s)_"r", arrow.r)$ #pause
          - Changes Mario's $(bold(r), dot(bold(r)))$ in game memory #pause
          - Human understandable, easier to implement for game developers
  ]


==
    #side-by-side(align: left)[
    #cimage("figures/lecture_13/mario.png", width: 100%)
  ][
  *Question:* In Mario, a single image frame is not a Markov state. How come? #pause
  
  *Answer:* Cannot measure velocity.
  ]


==
    #side-by-side(align: left)[
    #cimage("figures/lecture_13/mario.png", width: 100%)
  ][
  #text(size: 24pt)[
    *Question:* Why do we need velocity in the state? #pause
  
    *Answer:* If we don't have it, Markov property is violated
  
    $T(s_t, a_t)$: Mario is moving $arrow.t, arrow.b, arrow.l, arrow.r$ 
  
    $T(s_t, a_t | s_(t-1))$: Mario is moving $arrow.r$ at 1 m/s #pause
  
    Not conditionally independent! $T(s_t, a_t | s_(t-1), a_(t-1), dots, s_(0), a_(0)) != T(s_t, a_t) $
  ]
  ]


// 15 jul, w video 43min

==
  #side-by-side(align: left)[
    #cimage("figures/lecture_13/mario.png", width: 100%)
  ][
    *Reward ($R$)*? #pause
      - 1 for beating the level and 0 otherwise #pause
      - Total score #pause
      - 1 for beating the level + $0.01 dot "score"$ 
  ]


// Say somethign about we have all the pieces to start talking about learning
// The objective we are trying to solve

==
  - $S checkmark$ #pause
  - $A checkmark$ #pause
  - $T checkmark$ #pause
  - $R checkmark$ #pause
  - $gamma ?$

// 34 mins


==
  Agent goal in RL is to maximize the *cumulative* reward #pause
  
  The cumulative reward is called the *return ($G$)*
  $ G = sum_(t=0)^oo R(s_(t+1)) = sum_(t=0)^oo r_t $ #pause
  Note that we care about all future rewards, not just the current reward!


==
  Do humans maximize the return? #pause
  
  *Experiment:* one cookie now, or two cookies in a year? #pause

  Usually, humans and animals prefer rewards sooner. #pause

  The *discount factor $gamma$* injects this preference into RL. Results in the *discounted return (G)* #pause
  
  $ G = sum_(t=0)^oo gamma^t r_t = r_0 + gamma r_1 + gamma^2 r_2 + dots \
   0 <= gamma <= 1
  $ #pause

  Where have we seen this before?


==
  $ G = sum_(t=0)^oo gamma^t r_t = r_0 + gamma r_1 + gamma^2 r_2 + dots \
   0 <= gamma <= 1
  $ #pause
  With a reward of 1 at each timestep and $gamma = 0.9$
  $ 
   G = sum_(t=0)^oo gamma^t r_t = 1 + 0.9 + 0.81 + dots = 1 / (1 - gamma) = 10 \
  $ #pause

  We almost always choose to maximize the #underline[discounted] return


// 15 jul, w video 47min


==
  *Exercise:* Reinforcement learning also describes human and animal behaviors. How can you describe your behavior using reinforcement learning? #pause

  *Environment:* City of Cambridge #pause
  
  *State:* My position, motivation, weather, and current activity #pause
  
  *Action Space:* Either go to the Cambridge Blue or go home #pause

  *Reward Function:* $100 dot "drink"_"beer" - 1/m w_"rain"$ #pause

  This happens internally when I decide to go to the pub after work



= Agents and Policies

==
  #side-by-side(align: left)[  
    #env_diagram 
  ][
    - We have defined the environment #pause
    - Now let us define the agent
  ]



==
  The agent acts following a *policy* $pi$. #pause
  
  $pi: S arrow Delta A$ is a mapping from states to actions (or action probabilities), determining agent behavior in the MDP. #pause

  #side-by-side[$ a_t tilde pi(s_t) $][Sample action from the policy] #pause

  #side-by-side[$ pi(a_t | s_t) $][Probability of taking each action]

==

  #text(size: 24pt)[
  The policy that maximizes the discounted return ($G$) is called an *optimal policy ($pi_*$)* #pause

  $ pi_* = max_(pi) sum_(t=0)^oo gamma^t r_t  $ #pause

  Reward function $R$ is deterministic #pause

  State transition function and policy are stochastic, we must consider this! #pause

  $ r_t tilde [ #reward_dist_labeled ]  $
  ]


==
  $ r_t tilde [ #reward_dist_labeled ]  $ #pause

  I think many of you might be afraid of distributions #pause

  (Many) RL researchers are also afraid of distributions #pause

  Often, we get rid of the distribution using the *expectation* #pause

  $ bb(E)[r_t] = integral_S integral_A #reward_dist_labeled dif a_(t) dif s_(t+1) $

==

  $ pi_* = max_(pi) sum_(t=0)^oo gamma^t r_t; quad bb(E)[r_t] = integral_S integral_A R(s_(t+1)) T(s_(t+1) | s_(t),a_(t)) pi(a_(t) | s_(t)) dif a_(t) dif s_(t+1) $

  *In English:* The optimal must consider the action distribution combined and state transition distribution to compute the reward/return #pause

  We write the return as the expectation given our policy actions
  
  $ pi_* = max_(pi) space bb(E) [ sum_(t=0)^oo gamma^t r_t mid(|) a_t tilde pi (s_t)] $ #pause

  Now, our policy is truly optimal
  

// Draw PGM-esque thingy

// 15 jul, w video 55min

= Q Learning

==
  We use *algorithms* to search for the optimal policy $pi_*$ #pause
  
  Virtually all algorithms are based on either *Q Learning (QL)*, *Policy Gradient (PG)*, or both #pause
  
  Popular algorithms:
    - Deep Q Networks (DQN) #pause
    - Proximal Policy Optimization (PPO) #pause
    - Deep Deterministic Policy Gradient (DDPG) #pause
    - Twin Deep Deterministic Policy Gradient (TD3) #pause
    - Soft Actor Critic (SAC) #pause
    - Advantage Weighted Regression (AWR) #pause
    - Asynchronous Actor Critic (A2C)

==

  *DQN:* *Q* learning using a deep neural network #pause
  
  *PPO:* Policy gradient with update clipping and *Q/V* function #pause

  *DDPG:* *Q* learning with continuous actions via learned $op("argmax")$ #pause

  *TD3:* DDPG with action noise and a double *Q* trick #pause
  
  *SAC:* TD3 with entropy bonuses #pause

  *AWR:* Offline policy gradient with *Q/V* function #pause

  *A2C:* Policy gradient with *Q/V* function


==

  //Q learning is state of the art in 2024 (see REDQ, DroQ, TQC) #pause

  A theoretical understanding of Q learning is necessary, because algorithms build on top of Q learning #pause

  We will derive and define Q learning #pause

  *The Plan:* #pause
    + Derive the value function $V$ #pause
    + Derive Q function from $V$ #pause
    + Derive an optimal policy from $Q$ #pause
    + Learn to train $Q$

==
  Recall the discounted return for a specific policy $pi$
  
  $ G_pi = bb(E) [sum_(t=0)^oo gamma^t r_t mid(|) a_t tilde pi(s_t)] $ #pause

  *In English:* At each timestep, we take an action $a_t tilde pi(s_t)$ #pause

  follow the state transition function $s_(t+1) tilde T(s_t, a_t)$ #pause

  and get a reward $r_t = R(s_(t+1))$ 


==
  $ G_pi = bb(E) [sum_(t=0)^oo gamma^t r_t mid(|) a_t tilde pi(s_t)] $ #pause

  *Question:* Where does $s_0$ come from? #pause

  *Answer:* Some higher power #pause

  That is not a good answer


==
  $ G_pi = bb(E) [sum_(t=0)^oo gamma^t r_t mid(|) a_t tilde pi(s_t)] $

  What if we defined the return starting from a specific state $s_0$? #pause

  $ V_pi (s_0) = bb(E) [sum_(t=0)^oo gamma^(t) r_t mid(|) a_t tilde pi(s_t)] $ #pause

  Measures the *value* of a state (how good is it to be in this state?), for a given policy $pi$

  We call this the *Value Function ($V_pi$)*   $quad V_pi: S arrow bb(R)$


==
  *The Plan:*
    + Derive the value function $V$
    + *Derive Q function from $V$*
    + Derive an optimal policy from $Q$
    + Learn to train $Q$

  
==
$ V_pi (s_0) = bb(E) [sum_(t=0)^oo gamma^(t) r_t mid(|) a_t tilde pi(s_t)] $ #pause
  
Let's go one step further. What if we parameterize the value function with an initial action? #pause

Pull the first term out of the sum #pause

$ V_pi (s_0) = bb(E) [r_0 mid(|) a_0 tilde pi(s_0)] + bb(E)[sum_(t=1)^oo gamma^(t) r_t mid(|) a_t tilde pi(s_t)] $


==
  #text(size: 24pt)[
    $ V_pi (s_0) = bb(E) [r_0 mid(|) a_0 tilde pi(s_0)] + bb(E)[sum_(t=1)^oo gamma^(t) r_t mid(|) a_t tilde pi(s_t)] $
    
    Now, rewrite $V_(pi)$ as a function of an action $a_0$ #pause
    
    $ V_pi (s_0, a_0) = bb(E) [r_0 mid(|) a_0] + bb(E)[sum_(t=1)^oo gamma^(t) r_t mid(|) a_t tilde pi(s_t)] $ #pause
    
    When $V$ depends on a specific action, we call it the *Q function*: 
    
    #q_function_return
  ]


// 58 mins

==

  #q_function_return #pause

  The Q function might appear simple but it is very powerful #pause

  $a_0$ affects your next state $s_(1)$, which affects the future 
  

==
    #q_function_return

  *Example:* You have PhD offers from Cambridge and Oxford #pause

  $ a_0 = {"Oxford", "Cambridge"} $ #pause 


  Q function gives you a number denoting how much better your life will be for attending Cambridge (based on your behavior $pi$). Takes into account reward (based on income, friend group, experiences, etc).


==
  #q_function_return

  $ 
    Q(s_0, "Cambridge") &= f("friends" + "experiences" + "income") = 1200 \
    Q(s_0, "Oxford") &= f("friends" + "experiences" + "income") = 900
  $ #pause

  #align(center)[#text(size:20pt)[#trans_diagram]]


==
  *The Plan:*
    + Derive the value function $V$
    + Derive Q function from $V$
    + *Derive an optimal policy from $Q$*
    + Learn to train $Q$


==
  #q_function_return

  $ 
    Q(s_0, "Cambridge") &= f("friends" + "experiences" + "income") = 1200 \
    Q(s_0, "Oxford") &= f("friends" + "experiences" + "income") = 900
  $ #pause

  *Question:* Given the Q function, what would your policy be?

==
  #q_function_return

  $ 
    bold(Q(s_0, "Cambridge") &= f("friends" + "experiences" + "income") = 1200) \
    Q(s_0, "Oxford") &= f("friends" + "experiences" + "income") = 900
  $

  *Question:* Given the Q function, what would your policy be?


==
  Write this more formally

  $ pi(s) = op("argmax", limits: #true)_(a in A) Q(s, a) $ #pause

  We said that $pi(s)$ is a distribution  #pause

  We can rewrite it as a degenerate distribution #pause

  $ pi(s) = "Deg"[ op("argmax", limits: #true)_(a in A) Q(s, a) ] $ #pause

  We call this the *greedy policy*


==
  $ pi(s) = "Deg"[ op("argmax", limits: #true)_(a in A) Q(s, a) ] $ #pause

  The greedy policy is optimal (see Bellman Optimality Equation) #pause

  $ pi_* (s) = "Deg"[ op("argmax", limits: #true)_(a in A) Q_* (s, a) ] $ #pause

  #qstar_function_return2




==
  $ pi_* (s) = "Deg"[ op("argmax", limits: #true)_(a in A) Q_* (s, a) ] $ #pause


  $ 
    bold(Q_*(s_0, "Cambridge") &= f("friends" + "experiences" + "income") = 1200) \
    Q_*(s_0, "Oxford") &= f("friends" + "experiences" + "income") = 900
  $ #pause

  #align(center)[#text(size:20pt)[#trans_diagram]]


// 1h5m

==
  *The Plan:*
    + Derive the value function $V$
    + Derive Q function from $V$
    + Derive an optimal policy from $Q$
    + *Learn to train $Q$*

// Abote 1h, without videos

==
  #qstar_argmax_return


==
  #qstar_argmax_intractable #pause

  We need infinitely many rewards to approximate $Q_*$ #pause

  Can we get rid of the infinite sum?

  //After infinite time, we will have one datapoint for training. Can we get rid of the infinite sum?


==
    #qstar_argmax_return #pause

    //, by factoring out $gamma$ and changing $r_t$ to $r_(t+1)$ 

    
    Shift the sum so it starts at $t=0$
    
    $ Q_(*) (s_0, a_0) = bb(E) [r_0 | a_0] + bb(E)[sum_(t=0)^oo gamma^(t+1) r_(t+1) mid(|) a_(t+1) tilde pi_* (s_(t+1)) ] $ #pause

    Factor out $gamma$

    $ Q_(*) (s_0, a_0) = bb(E) [r_0 | a_0] + gamma dot bb(E)[sum_(t=0)^oo gamma^t r_(t+1) mid(|) a_(t+1) tilde pi_* (s_(t+1)) ] $

==
#only((1, "3-"))[$ Q_(*) (s_0, a_0) = bb(E) [r_0 | a_0] + gamma dot bb(E)[sum_(t=0)^oo gamma^t r_(t+1) mid(|) a_(t+1) tilde pi_* (s_(t+1)) ] $] #pause

#only((2))[      
  $ Q_(*) (s_0, a_0) = bb(E) [r_0 | a_0] + gamma dot underbrace(bb(E)[sum_(t=0)^oo gamma^t r_(t+1) | a_(t+1) tilde pi_* (s_(t+1)) ], #cimage("figures/lecture_13/mr-krabs-tired.gif", height: 6em)) $
] #pause

This is the value function for the policy $pi_*$ starting at $s_1$ #pause

$ Q_(*) (s_0, a_0) = bb(E) [r_0 | a_0] + gamma dot V_(*)(s_1) $  #pause
==
This is the value function for the policy $pi_*$ starting at $s_1$

$ Q_(*) (s_0, a_0) = bb(E) [r_0 | a_0] + gamma dot V_(*)(s_1) $  #pause

  We can rewrite $V$ as $Q$, removing the dependence on $V$ #pause

  $ Q_(*) (s_0, a_0) = bb(E) [r_0 | a_0] + gamma dot Q_(*)(s_1, pi_* (s_1)) $ #pause

  The policy $pi_*$ takes the $op("argmax")$ over Q, which reduces to 
  
  $ Q_*(s_0, a_0) = bb(E) [r_0 | a_0] + gamma dot max_{a in A} Q_*(s_1, a) $

==
  $ Q_*(s_0, a_0) = bb(E) [r_0 | a_0] + gamma dot max_{a in A} Q_*(s_1, a) $ #pause

  Literature often drops the expectation and time subscripts

  $ Q_*(s, a) = r + gamma dot max_{a' in A} Q_*(s', a') $ #pause

  With the infinite sum gone, this is much easier to compute! #pause

  All we need is: 
  
  $ (s, a, r, gamma, s') $




==
  We can parameterize $Q$ with parameters $theta$ and try to approximate $Q_*$

  $ Q(s, a, theta) = r + gamma dot max_{a' in A} Q(s', a', theta) $ #pause

  Let us turn this into an optimization objective #pause

  $ Q(s, a, theta) - (r + gamma dot max_{a' in A} Q(s', a', theta)) = 0 $ #pause

  $ argmin_theta (Q(s, a, theta) - (r + gamma dot max_{a' in A} Q(s', a', theta)))^2 $

==
$ argmin_theta cal(L)(s, a, r, s', theta) = argmin_theta (Q(s, a, theta) - (r + gamma dot max_{a' in A} Q(s', a', theta)))^2 $ #pause


At the start of lecture, I said we do not know the answer in RL #pause

However, we can define this objective/loss function #pause

If we optimize this objective, we will find the optimal Q function #pause

$ lim_(cal(L) -> 0) Q(s, a, theta) = Q_* (s, a) $ 

==
If we optimize this objective, we will find the optimal Q function 

$ lim_(cal(L) -> 0) Q(s, a, theta) = Q_* (s, a) $ #pause

What does this mean? #pause

Given enough time and data, we can learn the best possible policy #pause
- Best chess player #pause
- Best driver #pause
- Best chef #pause
- Best surgeon

==
  #text(size: 25pt)[
  We defined the $Q$ function

  $ Q (s, a, theta) = r + gamma dot max_{a' in A} Q (s', a', theta) $ #pause

  We defined the optimal policy given the $Q$ function

  $ pi (s) = "Deg"[op("argmax", limits: #true)_(a in A) Q (s, a, theta)] $ #pause

  We defined the Q function training objective

  $ min_theta (Q (s, a, theta) - (r + gamma dot max_{a' in A} Q (s', a', theta)))^2 $
]


==
  Q learning learns superhuman policies on many video games #pause

  #side-by-side[#link("https://www.youtube.com/watch?v=O2QaSh4tNVw")][SMB]

  #side-by-side[#link("https://youtu.be/VIwGxOdXGfw?si=A-CVLI6vEJHOxrvx&t=478")][MK]


= Resources

==
  + Reinforcement Learning, an Introduction (2018, Sutton and Barto) 
    - Available for free online (legal)
    - All the RL theory you will ever need #pause
    
  + David Silver’s slides for his RL course at UCL
    - Builds good intuition #pause
    
  + OpenAI Spinning Up
    - Mixes theory with implementation #pause
    
  + CleanRL
    - Verified, single-file implementations of many RL algorithms #pause

  + Special Topics in AI (Winter/Spring 2025)

// 1h15m without videos
// with videos, maybe 1h25?