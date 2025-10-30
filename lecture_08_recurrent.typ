#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.0"
#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "common.typ": *
#import "plots.typ": *
#import "@preview/algorithmic:1.0.5"
#import algorithmic: style-algorithm, algorithm-figure, algorithm
#import "@preview/mannot:0.3.0": *

#let handout = true

#set math.vec(delim: "[")
#set math.mat(delim: "[")

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(handout: handout),
  config-info(
    title: [Recurrence],
    subtitle: [CISC 7026 - Introduction to Deep Learning],
    author: [Steven Morad],
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

// TODO: Should X represent the sequence and X_i represent just one element of the sequence?


// Trace theory input only
// Composite memory (no neurons)
// Neural networks
// TODO training over sequence
// Linear recurrence IS convolution


= Admin
==
Almost done with coursework! #pause
- Last homework due 11.06 #pause
- Final project plan and group due 11.08 #pause
- Last exam planned for 11.21

==
For final project, you must form groups of *4-5 people* #pause
  - No more, no less #pause
  - *You cannot turn in project plan without a group* #pause
  - https://ummoodle.um.edu.mo/mod/choicegroup/view.php?id=809019

*Steps:* #pause
+ Submit "Final Project Group" on Moodle #pause
+ Group submit "Final Project Plan" on Moodle #pause
+ Group submit "Final Project" on Moodle

==
How was exam 2? #pause

You like it more or less than exam 1? #pause

Do you feel like you are learning some deep learning theory? #pause
- Reason for exams is to make you learn theory, not just `import torch` #pause
  - "What is shape?" questions make sure you (not LLM) did homework #pause
- LLMs can write `torch`, writing `torch` is not enough for good job #pause

To do what LLMs cannot, you must understand theory to: #pause
+ Uncover and fix issues when nn does not learn #pause
+ Discover new algorithms and models #pause


I am working with TAs upload grades sometime next week

==
Teaching schedules for next term released (winter 2026) #pause

Next term, I teach #underline[CISC7404 Special Topics in Artificial Intelligence] #pause
- *Topic:* Decision Making and Deep Reinforcement Learning #pause
  - Q learning, policy gradient, offline RL, imitation learning, RLHF #pause
- RL is my primary research focus #pause

Similar teaching style to this course #pause
- Theory-focused, uses statistics (random variables, expectations, etc) #pause
- 2 Assignments, 3 exams (best 2 exams scored) #pause
- Group final project (show last year projects) #pause

Enroll if you are interested!

==
We had to cover basics (backpropagation, classification, etc) #pause

Now, I think the lectures become more interesting #pause

Focus on modern methods that solve hard problems #pause
- Recurrent networks and long-term memory
- Autoencoders
- Probabilistic generative models
- Attention and transformers
- LLMs and reinforcement learning

= Review
==
// TODO Review
  In perceptrons, each neuron in a layer is independent #pause

  #align(center, cetz.canvas({
    import cetz.draw: *


  let image_values = (
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
  )
  draw_filter(0, 0, image_values)
  content((2, 2), image("figures/lecture_7/ghost_dog_bw.svg", width: 4cm))
  content((6, 6), image("figures/lecture_7/ghost_dog_bw.svg", width: 4cm))
  })) 

==
  We assume no relationship or ordering for input elements or parameter #pause

  $ sigma(theta_1 x_1 + theta_2 x_2) => sigma(theta_2 x_2 + theta_1 x_1) $

  #cimage("figures/lecture_7/permute.jpg") #pause

  These images are equivalent to a perceptron! 

==
  A *signal* represents information as a function of time, space or some other variable #pause

  $ x(t) = 2 t + 1 $ #pause

  $ x(u, v) = u^2 / v - 3 $ #pause

  $x(t), x(u, v)$ represent physical processes that we may or may not know #pause

  In *signal processing*, we analyze the meaning of signals

==
  $ x(t) = "stock price" $ #pause

  #align(center, stonks) #pause

  There is an underlying structure to $x(t)$ #pause

  *Structure:* Tomorrow's stock price will be close to today's stock price

==
  $ x(t) = "audio" $ #pause

  #align(center, waveform) #pause

  *Structure:* Nearby waves form syllables #pause

  *Structure:* Nearby syllables combine to create meaning 

==
  $ x(u, v) = "image" $ #pause
  
  #align(center, implot) #pause

  *Structure:* Repeated components (circles, symmetry, eyes, nostrils, etc)

==
    Two common properties of signals: #pause

    *Locality:* Information concentrated over small regions of space/time #pause

    *Translation Equivariance:* Shift in signal results in shift in output

==
  A more realistic scenario of locality and translation equivariance 

  #align(center, cetz.canvas({
    import cetz.draw: *


  let image_values = (
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
  )
  let image_colors = (
    (white, white, white, white, white, white, white, white),
    (white, white, white, white, white, none, none, none),
    (white, white, white, white, white, none, none, none),
    (white, white, white, white, white, none, none, none),
    (white, white, none, none, white, white, white, white),
    (white, white, none, none, white, white, white, white),
    (white, white, none, none, white, white, white, white),
    (white, white, white, white, white, white, white, white),
  )
  content((4, 4), image("figures/lecture_7/flowers.jpg", width: 8cm))
  draw_filter(0, 0, image_values, colors: image_colors)
  })) 

==
  We use convolution to turn signals into useful signals

  #side-by-side[#waveform_left][#hello] #pause

  Convolution is translation equivariant and local

==
  Convolution is the sum of products of a signal $x(t)$ and a *filter* $g(t)$ #pause

  If the t is continuous in $x(t)$

  $ 
  x(t) * g(t) &= integral_(-oo)^(oo) x(t + tau) g(tau) d tau \ 
  
  x(t) * g(t) &= sum_(tau=-oo)^(oo) x(t + tau) g(tau)
  $ #pause

  We slide the filter $g(t)$ across the signal $x(t)$ #pause

  *Note:* Implemented as cross-correlation, not classical convolution

==
  #conv_signal_plot #pause

  #conv_filter_plot #pause

  #conv_result_plot

==
  We can write both a perceptron and convolution in vector form

  #side-by-side[$ f(x(t), bold(theta)) = underbrace(sigma(bold(theta)^top vec(1, x(0.1), x(0.2), dots.v)), "Perceptron") $ #pause][
  $ f(x(t), bold(theta)) = underbrace(vec(
    sigma(bold(theta)^top vec(1, x(0.1), x(0.2))),
    sigma(bold(theta)^top vec(1, x(0.2), x(0.3))),
    dots.v
  ), "Convolution layer") $
  ]

  A convolution layer applies a "mini" perceptron to every few timesteps #pause

  The output size depends on the signal length

==
  If we want a single output, we should use *pooling* #pause

  $ z(t) = f(x(t), bold(theta)) = mat(
    sigma(bold(theta)^top vec(1, x(0.1), x(0.2))),
    sigma(bold(theta)^top vec(1, x(0.2), x(0.3))),
    dots
  )^top $ #pause 

  $ "SumPool"(z(t)) = sigma(bold(theta)^top vec(1, x(0.1), x(0.2))) + sigma(bold(theta)^top vec(1, x(0.2), x(0.3))) + dots
  $ #pause

  $ "MeanPool"(z(t)) = "SumPool"(z(t)) / (T - k + 1) ; quad "MaxPool"(z(t)) = max(z(t))
  $
==

  Our examples considered: #pause
  - 1 dimensional variable $t$ #pause
  - 1 dimensional output/channel $x(t)$ #pause
  - 1 filter #pause

  Can expand to multiple dimensions, the idea is exactly the same

==
  #let c = cetz.canvas({
    import cetz.draw: *

    content((4, 4), image("figures/lecture_7/dog_r.png", width: 8cm))
    
    draw_filter(
      0, 0,
      (
        (0, 1, 1, 1, 1, 1, 0, 1),
        (0, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (1, 1, 0, 0, 0, 1, 1, 1),
        (1, 1, 0, 0, 0, 1, 1, 1),
        (0, 1, 0, 0, 0, 1, 1, 0),
        (1, 0, 1, 1, 1, 1, 1, 1),
      )
    )

    content((13, 4), image("figures/lecture_7/dog_g.png", width: 8cm))
    draw_filter(
      9, 0,
      (
        (0, 1, 1, 1, 1, 1, 0, 1),
        (0, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (1, 1, 0, 0, 0, 1, 1, 1),
        (1, 1, 0, 0, 0, 1, 1, 1),
        (0, 1, 0, 0, 0, 1, 1, 0),
        (1, 0, 1, 1, 1, 1, 1, 1),
      )
    )  

    content((22, 4), image("figures/lecture_7/dog_b.png", width: 8cm))
    draw_filter(
      18, 0,
      (
        (0, 1, 1, 1, 1, 1, 0, 1),
        (0, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (1, 1, 0, 0, 0, 1, 1, 1),
        (1, 1, 0, 0, 0, 1, 1, 1),
        (0, 1, 0, 0, 0, 1, 1, 0),
        (1, 0, 1, 1, 1, 1, 1, 1),
      )
    )

  content((4, -1), text(size: 40pt)[$*$])

  draw_filter(
    3, -4,
    (
      (2, 0),
      (0, 1)
    )
  )
  
  content((8.5, -3), text(size: 40pt)[$+$])
  content((13, -1), text(size: 40pt)[$*$])

  draw_filter(
    12, -4,
    (
      (1, 1),
      (0, 1)
    )
  )

  content((17.5, -3), text(size: 40pt)[$+$])
  content((22, -1), text(size: 40pt)[$*$])
  draw_filter(
    21, -4,
    (
      (1, 0),
      (0, 0)
    )
  )
  })
  #c

==
   #side-by-side(align: left)[
    *Stride* allows you to "skip" cells during convolution #pause
    - Decrease the size of image without pooling #pause

    *Padding* adds zero pixels to the image to increase the output size
  ][
  #let c = cetz.canvas({
    import cetz.draw: *

    content((4, 4), image("figures/lecture_7/ghost_dog_bw.svg", width: 8cm))
    
    draw_filter(
      0, 0,
      (
        (0, 1, 1, 1, 1, 1, 0, 1),
        (0, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (1, 1, 0, 0, 0, 1, 1, 1),
        (1, 1, 0, 0, 0, 1, 1, 1),
        (0, 1, 0, 0, 0, 1, 1, 0),
        (1, 0, 1, 1, 1, 1, 1, 1),
      )
    )

  content((4, -1), text(size: 40pt)[$*$])

  draw_filter(
    3, -4,
    (
      (1, 0),
      (0, 1)
    )
  )
  
  })
  #c
  ]

= Sequence Modeling

==
    We previously used convolution to model signals #pause

    Many interesting signals depend on *time* #pause
    - Stock market #pause
    - Audio transcription #pause
    - Video classification #pause


    We also call these time-varying signals *sequences*

    $ x(1), x(2), dots $ #pause

    Today, we examine a new way to model time-varying signals/sequences 

==

    *Convolution:* Electrical engineering approach to sequence modeling #pause

    *Recurrent Models:* Neuroscience approach to sequence modeling #pause

    You can use convolution or recurrent models to model sequences #pause
    - Stock market
    - Audio transcription
    - Video classification #pause

    So what is the difference between convolution and recurrent models?

==
    Convolution works over signals of any domain (time, space, graph, etc) #pause

    Recurrent neural networks only work with time #pause

    Convolution relies on locality and translation equivariance properties #pause

    Recurrent models do not assume locality or equivariance #pause

    Equivariance and locality make learning more efficient, but not all problems have this structure #pause
    - Best solution depends on your problem structure #pause

    Examine some real life time-varying signals, and see if these properties hold

==
    *Example 1:* You like dinosaurs as a child, you grow up and study dinosaurs for work #pause

    *Question:* Is this local? #pause

    *Answer:* No, two related events separated by 20 years 

==
    *Example 2:* Your parent changes your diaper #pause

    *Question:* Translation equivariant? #pause

    *Answer:* No! Ok if you are a baby, different meaning if you are an adult! 

==
    #side-by-side(align: left)[*Example 3:* You hear a gunshot then see runners][
        #cimage("figures/lecture_8/running.jpg", height: 50%) #pause
    ]
    *Question:* Translation equivariant? #pause

    *Answer:* No! (1) gunshot, (2) see runners, enjoy the race. (1) see runners, (2) hear gunshot, you start running too!

==
    Problems without locality and translation equivariance are difficult to solve with convolution #pause

    For these problems, we need something else! #pause

    Can use humans as inspiration, life is a sequence/time-varying signal #pause

    How do humans experience time and process temporal data? #pause

    Can we design a neural network based on human perceptions of time?

= Composite Memory
==
    How do humans process temporal data/time-varying signals? #pause

    We only perceive the present #pause

    See dog $->$ photoreceptors fire $->$ neurons fire in the brain #pause

    No dog $->$ no photoreceptors fire $->$ no neurons fire #pause

    This process only considers the present #pause

    We know there was a dog, even if we no longer see it #pause

    We can reason over time by recording information as *memories* #pause

    Humans process temporal data by storing and recalling memories

==
    #side-by-side(align: left)[
        #cimage("figures/lecture_8/locke.jpeg")
    ][
        John Locke (1690) believed that conciousness and identity arise from memories #pause

        If all your memories were erased, you would be a different person #pause

        Without the ability to reason over memories, we would only react to stimuli like bacteria
    ]

==
    So how do we model memory in humans? #pause

    Can we create a similar model for neural networks?

==
    #side-by-side[
        Francis Galton (1822-1911) \ photo composite memory

        #cimage("figures/lecture_8/galton.jpg", height: 70%) #pause
    ][
        Composite photo of members of a party 

        #cimage("figures/lecture_8/composite_memory.jpg", height: 70%)
    ]

==
    *Task:* Find a mathematical model of composite memory #pause

    $ X: &bb(R)^(h times w) quad "Person you see at the party" \ #pause

    H: &bb(R)^(h times w) quad "The memories in your mind" $ #pause

    $ f: X^T times Theta |-> H $ #pause

    Composite photography/memory uses a weighted sum #pause

    $ f(bold(x), bold(theta)) = sum_(i=1)^T bold(theta)^top overline(bold(x))_i $

==
    #side-by-side[
      Memories from the party 
    ][
      $ bold(h)_T = sum_(i=1)^T bold(theta)^top overline(bold(x))_i $ #pause
    ]


    #side-by-side[What if we see someone at $T+1$? #pause][
        $ bold(h)_(T+1) = (sum_(i=1)^T bold(theta)^top overline(bold(x))_i) + bold(theta)^top overline(bold(x))_(T+1) $ #pause
    ]



    #side-by-side[And another new person? #pause][
        $ bold(h)_(T+1) = (sum_(i=1)^T bold(theta)^top overline(bold(x))_i) + bold(theta)^top overline(bold(x))_(T+1) + bold(theta)^top overline(bold(x))_(T+2)  $ #pause
    ]

    We repeat the same process for each new face #pause

    We can rewrite $f$ as a *recurrent function*


= Linear Recurrence
==
    Let us rewrite composite memory as a recurrent function #pause

    $ f(bold(x), bold(theta)) = underbrace((sum_(i=1)^T bold(theta)^top overline(bold(x))_i), bold(h)) + bold(theta)^top overline(bold(x))_"new" $ #pause

    $ f(bold(h), bold(x), bold(theta)) = bold(h) + bold(theta)^top overline(bold(x)) $ #pause

    $ bold(x) in bb(R)^(d_x), quad bold(h) in bb(R)^(d_h) $

==
    #side-by-side[$  bold(x) in bb(R)^(d_x), quad bold(h) in bb(R)^(d_h) $][
    $ f(bold(h), bold(x), bold(theta)) = bold(h) + bold(theta)^top overline(bold(x)) $] #pause

    $ #redm[$bold(h)_1$] = f(bold(0), bold(x)_1, bold(theta)) = bold(0) + bold(theta)^top overline(bold(x))_1 $ #pause

    $ #greenm[$bold(h)_2$] = f(#redm[$bold(h)_1$], bold(x)_2, bold(theta)) = bold(h)_1 + bold(theta)^top overline(bold(x))_2 $ #pause

    $ bold(h)_3 = f(#greenm[$bold(h)_2$], bold(x)_3, bold(theta)) = bold(h)_2 + bold(theta)^top overline(bold(x))_3 $ #pause

    $ dots.v $

    $ bold(h)_T = f(bold(h)_(T-1), bold(x)_T, bold(theta)) = bold(h)_(T-1) + bold(theta)^top overline(bold(x))_T $ 

    //We *scan* through the inputs $bold(x)_1, bold(x)_2, dots, bold(x)_T$

==
    *Question:* What is the meaning of $bold(h)$ in humans? #pause

    #cimage("figures/lecture_8/insideout.jpg", height: 85%)
    // TODO: inside out

==
    Right now, our model remembers everything #pause

    But $bold(h)$ is a fixed size, what if $T$ is very large? #pause

    If we keep adding and adding $bold(x)$ into $bold(h)$, we will run out of space #pause

    We need to create space for new memories #pause

    https://www.youtube.com/watch?v=IQ8Aak-k5Yc #pause

    Humans make space for new memories by forgetting old memories

==
    #side-by-side[Murdock (1982) #cimage("figures/lecture_8/murdock.jpg", height: 80%) #pause ][
        
        $ f(bold(h), bold(x), bold(theta)) = #redm[$gamma$] bold(h) + bold(theta)^top overline(bold(x)); quad 0 < gamma < 1 $ #pause

        #side-by-side[*Key Idea:*][$ lim_(T -> oo) gamma^T = 0 $ #pause ]

        Let $gamma = 0.9$ #pause

        $ 0.9 dot bold(h) = 0.9 bold(h) $ #pause
        $ 0.9 dot 0.9 dot bold(h) = 0.81 bold(h) $ #pause
        $ 0.9 dot 0.9 dot 0.9 dot dots dot bold(h) = 0 $ #pause

        Let us work out the math
        ] 

==
    $ f(bold(h), bold(x), bold(theta)) = gamma bold(h) + bold(theta)^top overline(bold(x)); quad 0 < gamma < 1 $ #pause


    $ bold(h)_T = gamma bold(h)_(T - 1) + bold(theta)^top overline(bold(x))_T $ #pause

    *Question:* What is $bold(h)_(T - 1)$ in terms of $bold(h)_(T - 2)$? #pause
    
    $ bold(h)_(T - 1) = #redm[$gamma bold(h)_(T-2) + bold(theta)^top overline(bold(x))_(T - 1)$] $ #pause

    $ bold(h)_T = gamma (#redm[$gamma bold(h)_(T - 2) + bold(theta)^top overline(bold(x))_(T - 1)$]) +  bold(theta)^top overline(bold(x))_T $ #pause

    $ bold(h)_T = gamma (#redm[$gamma$] (#greenm[$gamma bold(h)_(T - 3) + bold(theta)^top overline(bold(x))_(T - 2)$] ) #redm[$+ bold(theta)^top overline(bold(x))_(T - 1)$]) +  bold(theta)^top overline(bold(x))_T $ #pause

    $ bold(h)_T = gamma^3 bold(h)_(T - 3) + gamma^2 bold(theta)^top overline(bold(x))_(T - 2) + gamma bold(theta)^top overline(bold(x))_(T - 1) + bold(theta)^top overline(bold(x))_T $

    //$ bold(h)_T = sum_(i=1)^T gamma^(T - i - 1) bold(theta)^top overline(bold(x))_i $

==
    $ bold(h)_T = gamma^3 bold(h)_(T - 3) + gamma^2 bold(theta)^top overline(bold(x))_(T - 2) + gamma bold(theta)^top overline(bold(x))_(T - 1) + bold(theta)^top overline(bold(x))_T $ #pause

    $ bold(h)_T = gamma^T bold(theta)^top overline(bold(x))_1 + gamma^(T - 1) bold(theta)^top overline(bold(x))_2 + dots + gamma^2 bold(theta)^top overline(bold(x))_(T - 2) + gamma bold(theta)^top overline(bold(x))_(T - 1) + bold(theta)^top overline(bold(x))_T $ #pause

    //$ bold(h)_T = sum_(i=1)^T gamma^(T - i - 1) bold(theta)^top bold(x)_i $ #pause


    #align(center)[#forgetting]

==
    As $T$ increases, we add new information $bold(x)_T$ #pause

    As $T$ increases, we slowly forget old information #pause

    The memory decay is smooth and differentiable #pause

    We can learn the parameters $gamma, bold(theta)$ using gradient descent

==
    Morad et al., _Reinforcement Learning with Fast and Forgetful Memory_. Neural Information Processing Systems. (2024). #pause
    
    $ bold(H)_T = bold(gamma) bold(H)_(T - 1) + g(bold(x)_T) $ #pause

    #side-by-side[#cimage("figures/lecture_8/blind-minesweeper.png")][#cimage("figures/lecture_8/blind-maze.png")]

    https://www.youtube.com/watch?v=0ey63XPB-4U&t=85s
==

= Scans
==
    $ #redm[$bold(h)_1$] = f(bold(0), bold(x)_1, bold(theta)) = gamma bold(0) + bold(theta)^top overline(bold(x))_1 $ 

    $ #greenm[$bold(h)_2$] = f(#redm[$bold(h)_1$], bold(x)_2, bold(theta)) = gamma bold(h)_1 + bold(theta)^top overline(bold(x))_2 $ 

    $ bold(h)_3 = f(#greenm[$bold(h)_2$], bold(x)_3, bold(theta)) = gamma bold(h)_2 + bold(theta)^top overline(bold(x))_3 $ 

    $ dots.v $

    $ bold(h)_T = f(bold(h)_(T-1), bold(x)_T, bold(theta)) = gamma bold(h)_(T-1) + bold(theta)^top overline(bold(x))_T $ #pause

    How do we compute $bold(h)_1, bold(h)_2, dots, bold(h)_T$ on a computer? #pause

    We use an algebraic operation called a *scan*

==
    Our function $f$ is just defined for a single $X$ #pause

    $ f: H times X times Theta |-> H $ #pause

    But we need to process a sequence $X^T$! #pause

    To extend $f$ to sequences, we scan $f$ over the inputs

    $ scan(f): H times X^T times Theta |-> H^T $ #pause

    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(bold(h)_1, bold(h)_2, dots.v, bold(h)_T) $ 

==
    $ f: H times X times Theta |-> H, quad scan(f): H times X^T times Theta |-> H^T $ #pause

    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(h_1, h_2, dots.v, h_T) $ #pause
    
    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(f(bold(h)_0, bold(x)_1, bold(theta)), f(bold(h)_1, bold(x)_2, bold(theta)), dots.v, f(bold(h)_(T-1), bold(x)_T, bold(theta))) = vec(f(bold(h)_0, x_1), f(f(bold(h)_0, bold(x)_1), bold(x)_2), dots.v, f( dots f(bold(h)_0, bold(x)_1) dots, bold(x)_T)) $

==
    ```python
    import jax
    import jax.numpy as jnp

    T, d_x, d_h = 10, 2, 4
    xs, h0 = jnp.ones((T, d_x)), jnp.zeros((d_h,))
    theta = [jnp.ones((d_h,)), jnp.ones((d_x, d_h))] # (b, W)

    def f(h, x):
        b, W = theta
        result = h + (W.T @ x + b)
        return result, result # Must return tuple

    hT, hs = jax.lax.scan(f, init=h0, xs=xs) # Scan f over x
    ```

==
    `torch` does NOT have built-in scans, and is very slow compared to `jax` #pause

    We will write our own scan #pause

    ```python
    def scan(f, h, xs):
        # h shape is (d_h,)
        # xs shape is (T, d_x)
        hs = []
        for x in xs:
            h = f(h, x, theta) 
            hs.append(h)
        # output shape is (T, d_h)
        return torch.stack(hs)
    ```

==
    ```python
    import torch
    T, d_x, d_h = 10, 2, 4

    xs, h0 = torch.ones((T, d_x)), torch.zeros((d_h,))
    theta = (torch.ones((d_h,)), torch.ones((d_x, d_h))) 

    def f(h, x):
        b, W = theta
        result = h + (W.T @ x + b)
        return result # h

    hs = scan(f, h0, xs)
```

==
    Many deep learning courses do not teach scans #pause

    I teach scans because they are an important part of future LLMs #pause

    Companies are experimenting with LLMs using *associative scans* #pause
      - Qwen3-Next, Google RecurrentGemma, Griffin #pause

    #side-by-side(align: left)[Standard LLM: \~1M words][Associative LLM: 100M words] #pause

    An associative scan is a very fast scan we use when $f$ obeys the associative property $f(f(h_1, x_2), x_3) = f(h_1, f(x_2, x_3))$ #pause

    *Question:* Does $f(bold(h), bold(x), bold(theta)) = gamma bold(h) + bold(theta)^top overline(bold(x))$ obey the associative property? #pause

    *Answer:* Yes, linear operations obey the associative property

= Output Modeling
==
    We are almost done defining recurrent models #pause

    There is one more step we must consider, *memory recall* #pause

    $bold(h)$ stores all memories, but humans only access a few memories at once #pause

    *Example:* I ask you your favorite ice cream flavor #pause

    You recall previous times you ate ice cream, but not your phone number #pause

    We will model this recall of memories using a function $g$

==
    Let $g$ define our memory recall function #pause 

    $ g: H times X times Theta |-> Y $ #pause

    $g$ searches your memories $h$ using the input $x$, to produce output $y$ #pause

    $bold(x):$ "What is your favorite ice cream flavor?" #pause 

    $bold(h):$ Everything you remember (hometown, birthday, etc) #pause

    $g:$ Searches your memories for ice cream information, and responds "chocolate" #pause

    Now, we will combine $f$ and $g$

==
    *Step 1:* Perform scan to find recurrent states #pause

    $ vec(bold(h)_1, dots.v, bold(h)_T) = scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_f) $ #pause

    *Step 2:* Perform recall on recurrent states #pause

    $ vec(hat(bold(y))_1, dots.v, hat(bold(y))_T) = vec(
        g(bold(h)_1, bold(x)_1, bold(theta)_g),
        dots.v,
        g(bold(h)_T, bold(x)_T, bold(theta)_g),
    ) $ #pause

    Questions? This is on the homework

==
    To summarize, we defined: #pause
    - Recurrent function $f$ #pause
    - Scanned recurrence $scan(f)$ #pause
    - Output function $g$ #pause

    To run our model: #pause
    - Execute $scan(f)$ over inputs to make recurrent states #pause
    - Execute $g$ over recurrent states to make outputs

= Recurrent Loss Functions

// We built the recurrent model to model human mind
// But we can also use it to solve useful tasks
// Introduce y
// How to train recurrent models
// Use same error functions as standard NNs
// Examples of classification/regression tasks


==
    Let us examine some example tasks: #pause
    - Clock #pause
    - Explaining a video

==
    *Task:* Robot navigation #pause

    Input is robot velocity, output is robot position #pause

    $ X in bb(R)^(2), quad Y in bb(R)^2 $ 

==
    #side-by-side[Example input sequence:][
        $ vec(bold(x)_1, dots.v, bold(x)_T) = mat(dot(u)_1, dot(v)_1; dots.v, dots.v; dot(u)_T, dot(v)_T) $
    ] #pause
        

    #side-by-side[Desired output sequence][
       $ vec(bold(y)_1, dots.v, bold(y)_T) = mat(u_1, v_1; dots.v, dots.v; u_T, v_T) $
    ] #pause

    We have a corresponding label $bold(y)$ for each input $x$ #pause

    This is one datapoint (single datapoint has $T$ elements) #pause
    - Dataset contains $n times T$ positions and velocities

==
    Regression task, can use square error #pause

    First, scan $f$ over the inputs to find $h$

    $ vec(bold(h)_([i], 1), dots.v, bold(h)_([i], T)) = scan(f)(bold(h)_0, vec(bold(x)_([i], 1), dots.v, bold(x)_([i], T)), bold(theta)_f) $ #pause

    $ cal(L)(bold(X), bold(Y), bold(theta)) = sum_(i=1)^n sum_(j=1)^T sum_(k=1)^d_y [g(bold(h)_([i], j), bold(x)_([i], j), bold(theta)_g)_k - y_([i], j, k)]^2 $ #pause

    Onto the next task

==
    *Task:* Watch a video, then explain it 

    $ X in bb(Z)^(3 times 32 times 32), quad Y in {"comedy show", "action movie", ...} $ #pause

    #side-by-side[Example input sequence:][ $ vec(bold(X)_1, bold(X)_2, dots.v, bold(X)_T) $ ] #pause

    #side-by-side[Example output:]["dancing dog"] #pause

    Unlike before, we have many inputs but just one output!

==
    We will use the classification loss #pause

    We scan $f$ over the sequence, then compute $g$ for the final timestep #pause

    $ vec(bold(h)_([i], 1), dots.v, bold(h)_([i], T)) = scan(f)(bold(h)_0, vec(bold(x)_([i], 1), dots.v, bold(x)_([i], T)), bold(theta)_f) $ #pause

    $ cal(L)(bold(X), bold(Y), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_y) y_([i], j) log g(bold(h)_([i], #redm[$T$]), bold(x)_([i], #redm[$T$]), bold(theta)_g)_j $ #pause

    We only care about the $bold(h)_T$

==
    To summarize, we use standard losses for recurrent loss functions #pause

    $ sum_(i=1)^n #redm[$sum_(j=1)^T$] sum_(k=1)^(d_y) dots $

    Just be careful -- we often sum over an additional axis

= Backpropagation Through Time
==
    + We created the model #pause
    + We found the loss function #pause
    + Now we need to find the parameters! #pause

    Just like other models, we train recurrent models using gradient descent #pause
    - State $bold(h)$ updates over time, we must *backpropagate through time* #pause

    *Step 1:* Compute gradient #pause

    *Step 2:* Update parameters using gradient #pause

    How do we compute gradients for recurrent functions?

==
    First, compute gradient of $f$
    
    $ f(bold(h), bold(x), bold(theta)) = gamma bold(h) + bold(theta)^top overline(bold(x)) $ #pause

    *Question:* What is $(gradient_bold(theta) f)(bold(h), bold(x), bold(theta))$? #pause

    $ gradient_bold(theta) f(bold(h), bold(x), bold(theta)) = overline(bold(x))
    $ #pause

    Too easy, now let us find the gradient of $scan(f)$

==
    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(f(bold(h)_0, bold(x)_1, bold(theta)), f(bold(h)_1, bold(x)_2, bold(theta)), dots.v, f(bold(h)_(T-1), bold(x)_T, bold(theta))) = vec(f(bold(h)_0, bold(x)_1), f(f(bold(h)_0, bold(x)_1), bold(x)_2), dots.v, f( dots f(bold(h)_0, bold(x)_1) dots, bold(x)_T)) $ #pause

    #side-by-side[*Question:* What is $gradient_bold(theta) scan(f)$?][*Hint:* Chain rule] #pause

    $ (gradient_bold(theta) scan(f))(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(
        (gradient_bold(theta) f)(bold(h)_0, bold(x)_1), 
        (gradient_bold(theta) f)(f(bold(h)_0, bold(x)_1), bold(x)_2) (gradient f)(bold(h)_0, bold(x)_1), 
        dots.v, 
        (gradient_bold(theta) f)(dots f(bold(h)_0, bold(x)_1) dots, bold(x)_T) dots (gradient f)(bold(h)_0, bold(x)_1)
    ) $

==
    $ (gradient_bold(theta) scan(f))(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(
        gradient_bold(theta) f,
        (gradient_bold(theta) f) space (gradient f),
        (gradient_bold(theta) f) space (gradient f) space (gradient f),
        dots.v
    ) $ #pause

    *Question:* Any issues with this? #pause

    *Hint:* What if $gradient_bold(theta) f$ is $ << 1$ or $ >> 1$ ? #pause $(gradient_bold(theta) f)^T$ #pause

    #side-by-side[
      $ lim_(t->oo) 0.99^T = 0 $
    ][
      $ lim_(t->oo) 1.01^T = oo $
    ]

    *Read:* _Untersuchungen zu dynamischen neuronalen Netzen_ or \ _Learning Long-Term Dependencies with Gradient Descent is Difficult_

= Recurrent Neural Networks
==
    Until now, $f$ was a linear function #pause
    
    If we make $f$ a neural network, then we have a *recurrent neural network* (RNN)

==
Here is our linear recurrence 

$ f(bold(h), bold(x), bold(theta)) = bold(h) + bold(theta)^top overline(bold(x)) $ #pause

*Question:* How do we make it a neural network? #pause

#side-by-side[
  $ sigma(bold(h) + bold(theta)^top overline(bold(x))) $ #pause
][
  $ bold(h) + sigma(bold(theta)^top overline(bold(x))) $ #pause
][
  $ sigma(bold(h)) + bold(theta)^top overline(bold(x)) $ #pause
]

All work, and all are recurrent neural networks! #pause
- In practice, we often use $sigma(bold(h) + bold(theta)^top overline(bold(x)))$ form

==
    The simplest common recurrent neural network is the *Elman Network* #pause

   $ f(bold(h), bold(x), bold(theta)) = sigma(bold(theta)^top_1 bold(h) + bold(theta)^top_2 overline(bold(x))) $ #pause

    $ g(bold(h), bold(x), bold(theta)) = 
        bold(theta)^top_3 bold(h)
    $ #pause

    Apply sigmoid activation to $bold(h)$ to keep the recurrent state bounded #pause
    - Otherwise it can become very large over long sequences #pause

    /*
    $bold(h)$ grows large and causes exploding gradients, $sigma$ should be sigmoid! #pause

    #side-by-side(align: left)[*Question:* Co-domain of $f$? #pause][*Answer:* $[0, 1]$, bounded $bold(h)$] #pause
    */

    *Question:* Anything missing from our linear model? #pause

    *Answer:* Forgetting!

==
    Add forgetting 

    $ 
    f (bold(h), bold(x), bold(theta)) = 
    sigma(
        bold(theta)_1^top bold(h) #redm[$dot.circle f_"forget" (bold(h), bold(x), bold(theta))$] + bold(theta)_2^top overline(bold(x)) 
    )
    $ #pause


    $ 
    f_"forget" (bold(h), bold(x), bold(theta)) = sigma(
        bold(theta)_1^top overline(bold(x)) +  bold(theta)_2^top bold(h)
    )
    $ #pause

    *Question:* $sigma$ is sigmoid. What is co-domain of $f_"forget"$? #pause

    *Answer:* $[0, 1]$ #pause
    - When $f_"forget" = 1$, we forget nothing! 
    - When $f_"forget" = 0$, we forget everything! 
    - When $0 < f_"forget" < 1$, we forget some things! #pause

    Neural network learns which memories to forget with gradient descent

==
    *Minimal gated unit* (MGU) is a modern RNN #pause
    
    MGU defines two helper functions #pause

    $ 
    f_"forget" (bold(h), bold(x), bold(theta)) = sigma(
        bold(theta)_1^top overline(bold(x)) +  bold(theta)_2^top bold(h)
    ) 
    $ #pause

    $ 
    f_2(bold(h), bold(x), bold(theta)) = underbrace(sigma(
        bold(theta)_3^top overline(bold(x)) + bold(theta)_4^top 
            f_"forget" (bold(h), bold(x), bold(theta)) dot.circle bold(h)
    ), "Make new memories") 
    $ #pause

    $
    f(bold(h), bold(x), bold(theta)) = 
        underbrace(
          f_"forget" (bold(h), bold(x), bold(theta)) dot.circle bold(h)
        , "Forget old memories") + 
        underbrace((1 - f_"forget" (bold(h), bold(x), bold(theta))), "Replace") dot.circle underbrace(f_2(bold(h), bold(x), bold(theta)), "New memories")
    $ #pause

    Only forget when we have new information to remember

==
    There are even more complicated models #pause
    - Long Short-Term Memory (LSTM) #pause
    - Gated Recurrent Unit (GRU) #pause

    LSTM has 6 different functions! Too complicated to review #pause

    MGU is simpler and performs similarly to LSTM and GRU

==
    Recall the gradient for the linear recurrence #pause
    $ gradient_bold(theta) scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(
        gradient_bold(theta) f,
        (gradient_bold(theta) f) (gradient f),
        (gradient_bold(theta) f) (gradient f) (gradient f),
        dots.v
    ) $ #pause        

    #side-by-side[Elman network $f$:][$ f(bold(h), bold(x), bold(theta)) = sigma(bold(theta)^top_1 bold(h) + bold(theta)^top_2 overline(bold(x))) $] #pause

    *Question:* What is the gradient for $scan(f)$ of Elman network? 

==
    #side-by-side[Elman network $f$:][$ f(bold(h), bold(x), bold(theta)) = sigma(bold(theta)^top_1 bold(h) + bold(theta)^top_2 overline(bold(x))) $] #pause

    $ (gradient_bold(theta) scan(f))(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(
        gradient_bold(theta) f,
        (gradient_bold(theta) f) (gradient_bold(theta) f),
        (gradient_bold(theta) f) (gradient_bold(theta) f) (gradient_bold(theta) f),
        dots.v
    ) $ #pause        

    $ (gradient_bold(theta) scan(f))(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(
        gradient sigma,
        (gradient sigma) (gradient sigma),
        (gradient sigma) (gradient sigma) (gradient sigma),
        dots.v
    ) $ 

==
    $ (gradient_bold(theta) scan(f))(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(
        gradient sigma,
        (gradient sigma) (gradient sigma),
        (gradient sigma) (gradient sigma) (gradient sigma),
        dots.v
    ) $ 

    *Question:* What's the problem? #pause

    #side-by-side[*Answer:* Vanishing gradient][$ (gradient sigma) dot (gradient sigma) dot dots = 0 $] #pause

==
  All modern RNNs suffer from either exploding or vanishing gradient #pause
  - Very active area of research #pause
  - Could replace attention and yield much more efficient LLMs #pause

  Usually, we choose vanishing gradient #pause
  - Vanishing gradient prevents learning interactions over long $T$ #pause
  - Exploding gradient produces `NaN` and crashes training #pause

  Other works select eigenvalues near 1 to bound gradients #pause
  
  $ nabla sigma(bold(theta)^top bold(x)) = lambda bold(x); quad lambda approx 1 $ #pause

  But this restricts the power of the neural network


= Coding
==
    Jax RNN https://colab.research.google.com/drive/147z7FNGyERV8oQ_4gZmxDVdeoNt0hKta#scrollTo=TUMonlJ1u8Va

    Homework https://colab.research.google.com/drive/1CNaDxx1yJ4-phyMvgbxECL8ydZYBGQGt?usp=sharing

= Find a Group
==
All students *must stay 10 more minutes today* #pause
- Talk to each other to find a group of 4-5 people #pause
  - Discuss your project ideas, skills, etc #pause
- Cannot do final project alone #pause

Groups with 5 members can leave now (Groups S, P, A)
