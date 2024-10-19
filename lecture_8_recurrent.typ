#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2"
#import "common.typ": *
#import "conv_renders.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

#set math.vec(delim: "[")
#set math.mat(delim: "[")

// #enable-handout-mode(true)

// TODO: Should X represent the sequence and X_i represent just one element of the sequence?


#let forgetting = { 
    set text(size: 22pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 10,
    y-tick-step: 0.5,
    y-min: 0,
    y-max: 1,
    x-label: [Time],
    y-label: [],
    {
      plot.add(
        domain: (0, 40), 
        label: [Memory Strength],
        style: (stroke: (thickness: 5pt, paint: red)),
        t => calc.pow(0.9, t)
      )
    })
})}



// Trace theory input only
// Composite memory (no neurons)
// Neural networks
// TODO training over sequence
// Linear recurrence IS convolution

#let ag = (
  [Review],
  [Sequence Modeling],
  [Composite Memory],
  [Linear Recurrence],
  [Scans],
  [Output Modeling],
  [Recurrent Loss Functions],
  [Backpropagation through Time],
  [Recurrent Neural Networks],
  [Coding]
)

#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 8: RNNs"
)

#title-slide(
  title: [Recurrent Neural Networks],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
)

#slide(title: [Admin])[
    Makeup lecture Saturday October 26, 13:00-16:00 
]


#aslide(ag, none)
#aslide(ag, 0)
// TODO Review
#sslide[
  In perceptrons, each pixel is an independent neuron #pause

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
]

#sslide[
  These images are equivalent to a neural network

  #cimage("figures/lecture_7/permute.jpg") #pause

  It is a miracle that our neural networks could classify clothing!
]

#sslide[
  A *signal* represents information as a function of time, space or some other variable #pause

  $ x(t) = 2 t + 1 $ #pause

  $ x(u, v) = u^2 / v - 3 $ #pause

  $x(t), x(u, v)$ represent physical processes that we may or may not know #pause

  In *signal processing*, we analyze the meaning of signals

]

#sslide[
  $ x(t) = "stock price" $ #pause

  #align(center, stonks) #pause

  There is an underlying structure to $x(t)$ #pause

  *Structure:* Tomorrow's stock price will be close to today's stock price
]

#sslide[
  $ x(t) = "audio" $ #pause

  #align(center, waveform) #pause

  *Structure:* Nearby waves form syllables #pause

  *Structure:* Nearby syllables combine to create meaning 
]

#sslide[
  $ x(u, v) = "image" $ #pause
  
  #align(center, implot) #pause

  *Structure:* Repeated components (circles, symmetry, eyes, nostrils, etc)
]

#sslide[
    *Locality:* Information concentrated over small regions of space/time #pause

    *Translation Equivariance:* Shift in signal results in shift in output
]

#sslide[
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
]

#slide(title: [Convolution])[
  We use convolution to turn signals into useful signals

  #side-by-side[#waveform_left][#hello] #pause

  Convolution is translation equivariant and local
]

#slide(title: [Convolution])[
  Convolution is the sum of products of a signal $x(t)$ and a *filter* $g(t)$ #pause

  If the t is continuous in $x(t)$

  $ x(t) * g(t) = integral_(-oo)^(oo) x(t - tau) g(tau) d tau $ #pause

  If the t is discrete in $x(t)$

  $ x(t) * g(t) = sum_(tau=-oo)^(oo) x(t - tau) g(tau) $ #pause

  We slide the filter $g(t)$ across the signal $x(t)$
]

#slide(title: [Convolution])[

  #conv_signal_plot #pause

  #conv_filter_plot #pause

  #conv_result_plot
]

#slide(title: [Convolution])[
  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    1, 2, 3, 4, 5;
    2, 1;
    space ; 
  )
  $
]

#slide(title: [Convolution])[
  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    #redm[$1$], #redm[$2$], 3, 4, 5;
    #redm[$2$], #redm[$1$] ;
    #redm[$4$]; 
  )
  $
]

#slide(title: [Convolution])[
  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    1, #redm[$2$], #redm[$3$], 4, 5;
    , #redm[$2$], #redm[$1$] ;
    4, #redm[$7$]; 
  )
  $
]

#slide(title: [Convolution])[
  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    1, 2, #redm[$3$], #redm[$4$], 5;
    , , #redm[$2$], #redm[$1$] ;
    4, 5, #redm[$10$]; 
  )
  $
]

#slide(title: [Convolution])[
  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    1, 2, 3, #redm[$4$], #redm[$5$] ;
    , , , #redm[$2$], #redm[$1$] ;
    4, 5, 10, #redm[$13$]; 
  )
  $
]


#slide(title: [Convolution])[
  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    1, 2, 3, 4, 5 ;
    2, 1 ;
    4, 5, 10, 13; 
  )
  $ #pause

  To make a convolution layer, we make the filter with trainable parameters #pause

  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    1, 2, 3, 4, 5 ;
    theta_2, theta_1 ;
    #hide[1]; 
  )
  $ 
]

#sslide[
  We can write both a perceptron and convolution in vector form

  #side-by-side[$ f(x(t), bold(theta)) = sigma(bold(theta)^top vec(1, x(0.1), x(0.2), dots.v)) $ #pause][
  $ f(x(t), bold(theta)) = vec(
    sigma(bold(theta)^top vec(1, x(0.1), x(0.2))),
    sigma(bold(theta)^top vec(1, x(0.2), x(0.3))),
    dots.v
  ) $
  ]

  A convolution layer applies a "mini" perceptron to every few timesteps #pause

  The output size depends on the signal length
]

#sslide[
  If we want a single output, we should *pool* #pause

  $ z(t) = f(x(t), bold(theta)) = mat(
    sigma(bold(theta)^top vec(1, x(0.1), x(0.2))),
    sigma(bold(theta)^top vec(1, x(0.2), x(0.3))),
    dots
  )^top $ #pause 

  $ "SumPool"(z(t)) = sigma(bold(theta)^top vec(1, x(0.1), x(0.2))) + sigma(bold(theta)^top vec(1, x(0.2), x(0.3))) + dots
  $ #pause

  $ "MeanPool"(z(t)) = 1 / (T - k + 1) "SumPool"(z(t)); quad "MaxPool"(z(t)) = max(z(t))
  $
]

#sslide[
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
]

#sslide[
  #side-by-side[
    One last thing, *stride* allows you to "skip" cells during convolution #pause

    This can decrease the size of image without pooling #pause

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
]

#aslide(ag, 0)
#aslide(ag, 1)

/*
#sslide[
    - Conv is good at some Things
    - Generalizes to n dimensions, etc
    - Shortcomings - locality not good in all cases
    - Equivariance not good in all cases
    - Imagine memories created over a lifetime
        - Targeted remembering and forgetting of important events
    - RNNs developed specifically for temporal problems
    - Are there other ways to process sequences
    Convolution is not the only way to process sequential data
]
*/

#sslide[
    We previously used convolution to model signals #pause

    Convolution is an electrical engineering approach to modeling sequences #pause

    Now, we will discuss a psychological approach to sequence modeling #pause

    We call these models *recurrent neural networks* (RNNs) #pause

    You can solve temporal tasks using either convolution or RNNs
]

#sslide[
    Convolution works over inputs of any variables (time, space, etc) #pause

    Recurrent neural networks only work with time #pause

    Convolution makes use of locality and translation equivariance properties #pause

    Recurrent models do not assume locality or equivariance #pause

    Equivariance and locality make learning more efficient, but not all problems have this structure 
]

#sslide[
    *Example 1:* You like dinosaurs as a child, you grow up and study dinosaurs for work #pause

    *Question:* Is this local? #pause

    *Answer:* No, two events separated by 20 years #pause
]

#sslide[
    *Example 2:* Your parent changes your diaper #pause

    *Question:* Translation equivariant? #pause

    No! Ok if you are a baby, different meaning if you are an adult! #pause
]

#sslide[
    #side-by-side[*Example 3:* You hear a gunshot then see runners][
        #cimage("figures/lecture_8/running.jpg", height: 50%) #pause
    ]
    *Question:* Translation equivariant? #pause

    *Answer:* No! (1) gunshot, (2) see runners, enjoy the race. (1) see runners, (2) hear gunshot, you start running too! #pause

    *Question:* Any other examples? 
]

#sslide[
    Problems without locality and translation equivariance are difficult to solve with convolution #pause

    For these problems, we need something else! #pause

    Humans experience time and process temporal data #pause

    Can we design a neural network based on human perceptions of time?
]

#aslide(ag, 1)
#aslide(ag, 2)

#sslide[
    How do humans experience time? #pause

    Humans create memories #pause

    We experience time when we reason over our memories
]

#sslide[
    #side-by-side[
        #cimage("figures/lecture_8/locke.jpeg")
    ][
        John Locke (1690) believed that conciousness and identity arise from memories #pause

        If all your memories were erased, you would be a different person #pause

        Without the ability to reason over memories, we would simply react to stimuli like bacteria
    ]
]

#sslide[
    So how do we model memory in humans? 
]

#sslide[
    #side-by-side[
        Francis Galton (1822-1911), composite memory

        #cimage("figures/lecture_8/galton.jpg", height: 70%) #pause
    ][
        Composite photo of members of a party 

        #cimage("figures/lecture_8/composite_memory.jpg", height: 70%)
    ]
]

#sslide[
    *Task:* Model how our mind represents memories #pause

    $ X: bb(R)^(T times h times w) quad "Faces you see at the club" $ #pause

    $ Y: bb(R)^(h times w) quad "The image in your mind" $

    $ f: X times Theta |-> Y $ #pause

    $ f(bold(x), bold(theta)) = sum_(i=1)^T bold(theta)^top overline(bold(x))_i $
]

#sslide[
    $ f(bold(x), bold(theta)) = sum_(i=1)^T bold(theta)^top overline(bold(x))_i $ #pause

    *Question:* What if we see a new face? #pause

    $ f(bold(x), bold(theta)) = (sum_(i=1)^T bold(theta)^top overline(bold(x))_i) + bold(theta)^top overline(bold(x))_"new" $ #pause


    *Question:* And another new face? #pause

    $ f(bold(x), bold(theta)) = (sum_(i=1)^T bold(theta)^top overline(bold(x))_i) + bold(theta)^top overline(bold(x))_"new" + bold(theta)^top overline(bold(x))_"newnew"  $
]

#aslide(ag, 2)
#aslide(ag, 3)

#sslide[
    We can rewrite this function *recurrently* #pause

    $ f(bold(x), bold(theta)) = underbrace((sum_(i=1)^T bold(theta)^top overline(bold(x))_i), bold(h)) + bold(theta)^top overline(bold(x))_"new" $ #pause

    $ f(bold(h), bold(x), bold(theta)) = bold(h) + bold(theta)^top overline(bold(x)) $

    $ bold(x) in bb(R)^(d_x), quad bold(h) in bb(R)^(d_h) $
]

#sslide[
    #side-by-side[$  bold(x) in bb(R)^(d_x), quad bold(h) in bb(R)^(d_h) $][
    $ f(bold(h), bold(x), bold(theta)) = bold(h) + bold(theta)^top overline(bold(x)) $] #pause

    $ #redm[$bold(h)_1$] = f(bold(0), bold(x)_1, bold(theta)) = bold(0) + bold(theta)^top overline(bold(x))_1 $ #pause

    $ #greenm[$bold(h)_2$] = f(#redm[$bold(h)_1$], bold(x)_2, bold(theta)) = bold(h)_1 + bold(theta)^top overline(bold(x))_2 $ #pause

    $ bold(h)_3 = f(#greenm[$bold(h)_2$], bold(x)_3, bold(theta)) = bold(h)_2 + bold(theta)^top overline(bold(x))_3 $ #pause

    $ dots.v $

    $ bold(y) = bold(h)_T = f(bold(h)_(T-1), bold(x)_T, bold(theta)) = bold(h)_(T-1) + bold(theta)^top overline(bold(x))_T $ #pause

    //We *scan* through the inputs $bold(x)_1, bold(x)_2, dots, bold(x)_T$
]

#sslide[
    *Question:* What is the meaning of $bold(h)$ in humans? #pause

    #cimage("figures/lecture_8/insideout.jpg", height: 85%)
    // TODO: inside out
]


#sslide[
    Right now, our model remembers everything #pause

    https://www.youtube.com/watch?v=IQ8Aak-k5Yc #pause

    Humans cannot remember everything! #pause
]

#sslide[
    #side-by-side[Murdock (1982) #cimage("figures/lecture_8/murdock.jpg", height: 80%) #pause ][
        
        $ f(bold(h), bold(x), bold(theta)) = #redm[$gamma$] bold(h) + bold(theta)^top overline(bold(x)); quad 0 < gamma < 1 $ #pause

        *Key Idea:* $lim_(T -> oo) gamma^T = 0$ #pause 

        Let $gamma = 0.9$ #pause

        $ 0.9 dot 0.9 dot bold(h) = 0.81 bold(h) $ #pause
        $ 0.9 dot 0.9 dot 0.9 dot bold(h) = 0.72 bold(h) $ #pause
        $ 0.9 dot 0.9 dot 0.9 dot dots dot bold(h) = 0 $ 
        ] 
]

#sslide[
    $ f(bold(h), bold(x), bold(theta)) = gamma bold(h) + bold(theta)^top overline(bold(x)); quad 0 < gamma < 1 $ #pause


    $ bold(h)_T = gamma bold(h)_(T - 1) + bold(theta)^top overline(bold(x))_T $ #pause

    *Question:* What is $bold(h)_(T - 1)$ in terms of $bold(h)_(T - 2)$? #pause
    
    $ #redm[$gamma bold(h)_(T-2) + bold(theta)^top overline(bold(x))_(T - 1)$] $ #pause

    $ bold(h)_T = gamma (#redm[$gamma bold(h)_(T - 2) + bold(theta)^top overline(bold(x))_(T - 1)$]) +  bold(theta)^top overline(bold(x))_T $ #pause

    $ bold(h)_T = gamma (#redm[$gamma$] (#greenm[$gamma bold(h)_(T - 3) + bold(theta)^top overline(bold(x))_(T - 2)$] ) #redm[$+ bold(theta)^top overline(bold(x))_(T - 1)$]) +  bold(theta)^top overline(bold(x))_T $ #pause

    $ bold(h)_T = gamma^3 bold(h)_(T - 3) + gamma^2 bold(theta)^top overline(bold(x))_(T - 2) + gamma bold(theta)^top overline(bold(x))_(T - 1) + bold(theta)^top overline(bold(x))_T $

    //$ bold(h)_T = sum_(i=1)^T gamma^(T - i - 1) bold(theta)^top overline(bold(x))_i $
]

#sslide[
    $ bold(h)_T = gamma^3 bold(h)_(T - 3) + gamma^2 bold(theta)^top overline(bold(x))_(T - 2) + gamma bold(theta)^top overline(bold(x))_(T - 1) + bold(theta)^top overline(bold(x))_T $ #pause

    $ bold(h)_T = sum_(i=1)^T gamma^(T - i - 1) bold(theta)^top bold(x)_i $ #pause


    #align(center)[#forgetting]
]

#sslide[
    As $T$ increases, we add new information #pause

    As $T$ increases, we slowly forget old information #pause

    The memory decay is smooth and differentiable #pause

    We can learn the parameters using gradient descent
]

#sslide[
    Morad et al., _Reinforcement Learning with Fast and Forgetful Memory_. Neural Information Processing Systems. (2024). #pause
    
    $ bold(H)_t = bold(gamma) dot.circle bold(H)_(t - 1) + g(bold(x)_t) $ #pause

    Our models learn to play board games and computer games #pause

    Outperforms other recurrent models (LSTM, GRU, etc)
]


#aslide(ag, 3)
#aslide(ag, 4)

#sslide[
    $ #redm[$bold(h)_1$] = f(bold(0), bold(x)_1, bold(theta)) = gamma bold(0) + bold(theta)^top overline(bold(x))_1 $ 

    $ #greenm[$bold(h)_2$] = f(#redm[$bold(h)_1$], bold(x)_2, bold(theta)) = gamma bold(h)_1 + bold(theta)^top overline(bold(x))_2 $ 

    $ bold(h)_3 = f(#greenm[$bold(h)_2$], bold(x)_3, bold(theta)) = gamma bold(h)_2 + bold(theta)^top overline(bold(x))_3 $ 

    $ dots.v $

    $ bold(y) = bold(h)_T = f(bold(h)_(T-1), bold(x)_T, bold(theta)) = gamma bold(h)_(T-1) + bold(theta)^top overline(bold(x))_T $ #pause

    How do we compute $bold(h)_1, bold(h)_2, dots, bold(h)_T$ on a computer? #pause

    We use an algebraic operation called a *scan*
]

#sslide[
    Our function $f$ is just defined for a single $X$ #pause

    $ f: H times X times Theta |-> H $ #pause

    To extend $f$ to sequences, we scan $f$ over the inputs

    $ scan(f): H times X^T times Theta |-> H^T $ #pause

    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(h_1, h_2, dots.v, h_T) $ 
]

#sslide[
    $ f: H times X times Theta |-> H, quad scan(f): H times X^T times Theta |-> H^T $ #pause

    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(h_1, h_2, dots.v, h_T) $ #pause
    
    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(f(h_0, x_1, bold(theta)), f(h_1, x_2, bold(theta)), dots.v, f(h_(T-1), x_T, bold(theta))) = vec(f(h_0, x_1), f(f(h_0, x_1), x_2), dots.v, f( dots f(h_0, x_1) dots, x_T)) $

]

#sslide[
    ```python
    import jax
    import jax.numpy as jnp

    T, d_h = 10, 4
    xs, h0 = jnp.ones((T, d_h)), jnp.zeros((d_h,))
    theta = (jnp.ones((d_h,)), jnp.ones((d_h, d_h))) # (b, W)

    def f(h, x):
        b, W = theta
        result = h + W.T @ x + b
        return result, result # return one, return all

    _, hs = jax.lax.scan(f, init=h0, xs=xs) # Scan f over x
    ```
]

#sslide[
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
        return hs
    ```

]

#sslide[
    ```python
    import torch
    T, d_h = 10, 4
    xs, h0 = torch.ones((T, d_h)), torch.zeros((d_h,))
    theta = (torch.ones((d_h,)), torch.ones((d_h, d_h))) 
    def f(h, x):
        b, W = theta
        result = h + W.T @ x + b
        return result # h

    hs = scan(f, h0, xs)
```
]

#sslide[
    Many deep learning courses do not teach scans #pause

    I teach scans because they are an important part of future LLMs #pause

    OpenAI, Google, etc are currently experimenting with LLMs that use *associative scans* #pause

    Standard LLM: \~8000 words, LLM + associative scan: 1M+ words #pause

    An associative scan is a very fast scan we use when $f$ obeys the associative property #pause

    *Question:* Does $f(bold(h), bold(x), bold(theta)) = gamma bold(h) + bold(theta)^top overline(bold(x))$ obey the associative property? #pause

    *Answer:* Yes, linear operations obey the associative property
]

#aslide(ag, 4)
#aslide(ag, 5)

#sslide[
    We are almost done defining recurrent models #pause

    There is one more step we must consider, *memory recall* #pause

    $bold(h)$ represents all memories, but humans only access a few memories at once #pause

    *Example:* I ask you your favorite ice cream flavor #pause

    You recall previous times you ate ice cream, but not your phone number #pause
]

#sslide[
    We model recall using a function $g$ #pause

    $ g: H times X times Theta |-> Y $ #pause

    $g$ searches your memories $h$ using the input $x$, to produce output $y$ #pause

    $bold(x):$ What is your favorite ice cream flavor? #pause 

    $bold(h):$ Everything you remember (hometown, birthday, etc) #pause

    $g:$ Searches your memories for ice cream memories, and responds "chocolate" #pause

    Now, combine $f$ and $g$
]

#sslide[
    *Step 1:* Perform scan to find recurrent states #pause

    $ vec(bold(h)_1, dots.v, bold(h)_T) = scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_f) $ #pause

    *Step 2:* Perform recall on recurrent states #pause

    $ vec(bold(y)_1, dots.v, bold(y)_T) = vec(
        g(bold(h)_1, bold(x)_1, bold(theta)_g),
        dots.v,
        g(bold(h)_T, bold(x)_T, bold(theta)_g),
    ) $ #pause

    Questions? This is on the homework
]

#sslide[
    We defined: 
    - Recurrent function $f$ #pause
    - Scanned recurrence $scan(f)$ #pause
    - Output function $g$

    To run: #pause
    - Execute $scan(f)$ over inputs to make recurrent states #pause
    - Execute $g$ over recurrent states to make outputs
]

#aslide(ag, 5)
#aslide(ag, 6)

// We built the recurrent model to model human mind
// But we can also use it to solve useful tasks
// Introduce y
// How to train recurrent models
// Use same error functions as standard NNs
// Examples of classification/regression tasks


#sslide[
    Let us examine some example tasks: #pause
    - Clock #pause
    - Explaining a video
]

#sslide[
    *Task:* Clock -- keep track of time #pause

    Every second, the second hand ticks #pause

    Every minute, the minute hand ticks #pause

    Add up the ticks to know the time #pause

    $ X in {0, 1}^2, quad Y in bb(R)^2 $ 
]

#sslide[
    #side-by-side[Example input sequence:][
        $ mat(0, 1; 0, 1; dots.v, dots.v; 1, 1) $
    ] #pause
        

    #side-by-side[Desired output sequence][
        $  mat(0, 1; 0, 2; dots.v, dots.v; m, s) $
    ]
]

#sslide[
    We have a ground truth for each input $y_i$ #pause

    Can use square error #pause

    First, scan $f$ over the inputs to find $h$

    $ bold(h)_([i], j) = scan(f)(bold(h)_0, bold(x)_[i], bold(theta)_f) $ #pause

    $ cal(L)(bold(X), bold(Y), bold(theta)) = sum_(i=1)^n sum_(j=1)^T [g(bold(h)_([i], j), bold(x)_([i], j), bold(theta)_g) - bold(y)_([i], j)^2]^2 $ #pause

    Onto the next task
]

#sslide[
    *Task:* Watch a video, then explain it to me

    $ X in bb(Z)^(3 times 32 times 32), quad Y in {"comedy show", "action movie", ...} $ #pause

    #side-by-side[Example input sequence:][ $ vec(I_1, I_2, dots.v, I_T) $ ] 

    #side-by-side[Example output:]["dancing dog"] #pause

    Unlike before, we have many inputs but just one output!
]

#sslide[
    We will use the classification loss #pause

    We scan $f$ over the sequence, the compute $g$ for the final timestep #pause

    $ h_([i],j) = scan(f)(bold(h)_0, bold(x)_[i], bold(theta)_f) $ #pause

    $ cal(L)(bold(X), bold(Y), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_y) y_([i], j) log g(bold(h)_([i], T), bold(x)_([i], T))_j $ #pause

    We only care about the $bold(h)_T$
]

#aslide(ag, 6)
#aslide(ag, 7)

#sslide[
    + We created the model #pause
    + We found the loss function #pause
    + Now we need to find the parameters! #pause

    Just like all other neural networks, we train recurrent models using gradient descent #pause

    *Step 1:* Compute gradient #pause

    *Step 2:* Update parameters
]

#sslide[
    $ f(bold(h), bold(x), bold(theta)) = gamma bold(h) + bold(theta)^top overline(bold(x)) $ #pause

    *Question:* What is $gradient_bold(theta) f$? #pause

    $ gradient_bold(theta) f(bold(h), bold(x), bold(theta)) = overline(bold(x))^top
    $
]
#sslide[
    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(f(h_0, x_1, bold(theta)), f(h_1, x_2, bold(theta)), dots.v, f(h_(T-1), x_T, bold(theta))) = vec(f(h_0, x_1), f(f(h_0, x_1), x_2), dots.v, f( dots f(h_0, x_1) dots, x_T)) $

    #side-by-side[*Question:* What is $gradient_bold(theta) scan(f)$?][*Hint:* Chain rule] #pause

    $ gradient_bold(theta) scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(
        gradient_bold(theta)[f](h_0, x_1), 
        gradient_bold(theta)[f](f(h_0, x_1), x_2) dot gradient_bold(theta) [f](h_0, x_1), 
        dots.v, 
        gradient_bold(theta)[f]( dots f(h_0, x_1) dots, x_T) dot dots dot gradient_bold(theta)[f](h_0, x_1)
    ) $
]

#aslide(ag, 7)
#aslide(ag, 8)

#sslide[
    If $f$ is a neural network, then we have a recurrent neural network
]

#sslide[
    The first recurrent neural network was the *Elman Network* #pause

    $ f(bold(h), bold(x), bold(theta)) = sigma(bold(theta)^top_1 overline(bold(h)) + bold(theta)^top_2 overline(bold(x))) $ #pause

    $ g(bold(h), bold(x), bold(theta)) = 
        bold(theta)^top_3 overline(bold(h))
    $ #pause

    $bold(h)$ grows large and causes exploding gradients, $sigma$ should be sigmoid! #pause

    #side-by-side[*Question:* Max value of $sigma(bold(h))$? #pause][1!] #pause

    *Question:* Anything missing from our linear model? #pause

    *Answer:* Forgetting!
]


#sslide[
    Add forgetting 
    $ 
    f_"forget" (bold(h), bold(x), bold(theta)) = sigma(
        bold(theta)_1^top overline(bold(x)) +  bold(theta)_2^top overline(bold(h))
    )
    $ #pause

    *Question:* $sigma$ is sigmoid. What is range/codomain of $f_"forget"$? #pause

    *Answer:* $[0, 1]$ #pause

    $ 
    f (bold(h), bold(x), bold(theta)) = 
    sigma(
        bold(theta)_3^top overline(bold(x)) + bold(theta)_4^top overline(bold(h)) dot.circle f_"forget" (bold(h), bold(x), bold(theta))
    )
    $ #pause

    When $f_"forget < 1"$, we forget! 

]

#sslide[
    *Minimal gated unit* (MGU) #pause

    $ 
    f_"forget" (bold(h), bold(x), bold(theta)) = sigma(
        bold(theta)_1^top overline(bold(x)) +  bold(theta)_2^top overline(bold(h))
    ) 
    $ #pause

    $ 
    f_2(bold(h), bold(x), bold(theta)) = sigma(
        bold(theta)_3^top overline(bold(x)) + bold(theta)_4^top (
            f_"forget" (bold(h), bold(x), bold(theta)) dot.circle bold(h)
        )
    ) 
    $

    $
    f(bold(h), bold(x), bold(theta)) = 
        f_"forget" (bold(h), bold(x), bold(theta)) dot.circle bold(h) + (1 - f_"forget" (bold(h), bold(x), bold(theta))) dot.circle f_2(bold(h), bold(x), bold(theta))
    $ #pause

    Left term forgets old, right term replaces forgotten memories
]

#sslide[
    There are even more complicated models #pause
    - Long Short-Term Memory (LSTM) #pause
    - Gated Recurrent Unit (GRU) #pause

    LSTM has 6 different functions! Too complicated to review
]

#sslide[
    #side-by-side[Elman network $f$:][$ f(bold(h), bold(x), bold(theta)) = sigma(bold(theta)^top_1 overline(bold(h)) + bold(theta)^top_2 overline(bold(x))) $] #pause

    *Question:* What is the gradient for $scan(f)$? #pause

    $ gradient_bold(theta) scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(
        gradient_bold(theta)[f](h_0, x_1), 
        gradient_bold(theta)[f](f(h_0, x_1), x_2) dot gradient_bold(theta) [f](h_0, x_1), 
        dots.v, 
        gradient_bold(theta)[f]( dots f(h_0, x_1) dots, x_T) dot dots dot gradient_bold(theta)[f](h_0, x_1)
    ) $ #pause

]
#sslide[
    #side-by-side[Elman network $f$:][$ f(bold(h), bold(x), bold(theta)) = sigma(bold(theta)^top_1 overline(bold(h)) + bold(theta)^top_2 overline(bold(x))) $] #pause

    $ gradient_bold(theta) scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(
        gradient_bold(theta)[f](h_0, x_1), 
        gradient_bold(theta)[f](f(h_0, x_1), x_2) dot gradient_bold(theta) [f](h_0, x_1), 
        dots.v, 
        gradient_bold(theta)[f]( dots f(h_0, x_1) dots, x_T) dot dots dot gradient_bold(theta)[f](h_0, x_1)
    ) $ #pause

    $ gradient_bold(theta) scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(
        gradient_bold(theta)[sigma](bold(theta)_1^top overline(bold(x))_1) overline(bold(x))_1^top, 
        gradient_bold(theta)[sigma](bold(theta)_2 overline(bold(h))_1, x_2) gradient_bold(theta) [sigma] (bold(theta)_1^top overline(bold(x))_1) overline(bold(x))_1^top, 
        dots.v, 
    ) $ #pause
]

#sslide[
    $ gradient_bold(theta) scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(
        gradient_bold(theta)[sigma](bold(theta)_1^top overline(bold(x))_1) overline(bold(x))_1^top, 
        gradient_bold(theta)[sigma](bold(theta)_2 overline(bold(h))_1, bold(x)_2) gradient_bold(theta) [sigma] (bold(theta)_1^top overline(bold(x))_1) overline(bold(x))_1^top, 
        dots.v, 
    ) $ #pause

    *Question:* What's the problem? #pause

    #side-by-side[*Answer:* Vanishing gradient][$ gradient[sigma] dot gradient[sigma] dot dots = 0 $] #pause

    *Question:* What can we do? #pause

    All RNNs suffer from either exploding gradient (ReLU) or vanishing gradient (sigmoid). Active area of research!
]


#aslide(ag, 7)
#aslide(ag, 8)

#sslide[
    Jax RNN https://colab.research.google.com/drive/147z7FNGyERV8oQ_4gZmxDVdeoNt0hKta#scrollTo=TUMonlJ1u8Va

    Homework https://colab.research.google.com/drive/1CNaDxx1yJ4-phyMvgbxECL8ydZYBGQGt?usp=sharing

    Makeup lecture Saturday October 26, 13:00-16:00 
]