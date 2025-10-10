#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.0"
#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "common.typ": *
#import "@preview/algorithmic:1.0.5"
#import algorithmic: style-algorithm, algorithm-figure, algorithm
#import "@preview/mannot:0.3.0": *

#let handout = true


// FUTURE TODO: Repeat self too much on locality/equivariance, restructure and replace with something else

#set math.vec(delim: "[")
#set math.mat(delim: "[")

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(handout: handout),
  config-info(
    title: [Convolution],
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


#set math.vec(delim: "[")
#set math.mat(delim: "[")


#let stonks = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 2,
    y-tick-step: 20,
    y-min: 0,
    y-max: 100,
    x-label: $ t $,
    y-label: $ x(t) $,
    {
      plot.add(
        domain: (0, 5), 
        label: [Stock Price (MOP)],
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 10 + 100 * (0.3 * calc.sin(0.2 * t) +
                0.1 * calc.sin(1.5 * t) +
                0.05 * calc.sin(3.5 * t) +
                0.02 * calc.sin(7.0 * t))
      )
    })
})}

#let waveform = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 0.5,
    y-tick-step: 500,
    y-min: -1000,
    y-max: 1000,
    x-label: $ t $,
    y-label: $ x(t) $,
    {
      plot.add(
        domain: (0, 1), 
        label: [dBm],
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 500 * (
          calc.sin(calc.pi * 1320 * t)
        )
      )
    })
})}

#let waveform_left = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 0.5,
    y-tick-step: 500,
    x-min: 0,
    x-max: 1,
    y-min: -1000,
    y-max: 1000,
    x-label: "Time (Seconds)",
    y-label: "dBm",
    {
      plot.add(
        domain: (0, 0.33), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 500 * (
          calc.sin(2 * calc.pi * 9.1 * t)
        )
      )
      plot.add(
        domain: (.33, 1.0), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => t
      )
    })
})}

#let hello = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 0.5,
    y-tick-step: 1,
    x-min: 0,
    x-max: 1,
    y-min: 0,
    y-max: 1,
    x-label: "Time (Seconds)",
    y-label: "Hello",
    {
      plot.add(
        domain: (0, 0.33), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 1
      )
      plot.add(
        domain: (.33, 1.0), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 0
      )
      plot.add-vline(
        0.33,
        style: (stroke: (thickness: 5pt, paint: red)),
      )
    })
})}

#let waveform_right = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 0.5,
    y-tick-step: 500,
    x-min: 0,
    x-max: 1,
    y-min: -1000,
    y-max: 1000,
    x-label: "Time (Seconds)",
    y-label: "Frequency (Hz)",
    {
      plot.add(
        domain: (0.66, 1.0), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 500 * (
          calc.sin(2 * calc.pi * 9.1 * t)
        )
      )
      plot.add(
        domain: (0, 0.66), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => t
      )
    })
})}

#let implot = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (7, 7),
    x-tick-step: 64,
    y-tick-step: 64,
    y-min: 0,
    y-max: 256,
    x-min: 0,
    x-max: 256,
    x-label: [$u$ (Pixels)],
    y-label: [$v$ (Pixels)],
    {
      plot.add(
        label: $ x(u, v) $,
        domain: (0, 1), 
        style: (stroke: (thickness: 0pt, paint: red)),
        t => 1000 * (
          calc.sin(calc.pi * 1320 * t)
        )
      )
      plot.annotate({
        import cetz.draw: *
        content((128, 128), image("figures/lecture_7/ghost_dog_bw.svg", width: 7cm))
      })
    })
})}

#let implot_color = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (7, 7),
    x-tick-step: 64,
    y-tick-step: 64,
    y-min: 0,
    y-max: 256,
    x-min: 0,
    x-max: 256,
    x-label: [$u$ (Pixels)],
    y-label: [$v$ (Pixels)],
    {
      plot.add(
        label: $ x(u, v) $,
        domain: (0, 1), 
        style: (stroke: (thickness: 0pt, paint: red)),
        t => 1000 * (
          calc.sin(calc.pi * 1320 * t)
        )
      )
      plot.annotate({
        import cetz.draw: *
        content((128, 128), image("figures/lecture_1/dog.png", width: 7cm))
      })
    })
})}

#let implot_left = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (7, 7),
    x-tick-step: 64,
    y-tick-step: 64,
    y-min: 0,
    y-max: 256,
    x-min: 0,
    x-max: 256,
    x-label: [$u$ (Pixels)],
    y-label: [$v$ (Pixels)],
    {
      plot.add(
        domain: (0, 1), 
        style: (stroke: (thickness: 0pt, paint: red)),
        t => 1000 * (
          calc.sin(calc.pi * 1320 * t)
        )
      )
      plot.annotate({
        import cetz.draw: *
        rect((0, 0), (256, 256), fill: gray)
      })
      plot.annotate({
        import cetz.draw: *
        content((64, 64), image("figures/lecture_7/ghost_dog_bw.svg", width: 3cm))
      })
    })
})}

#let implot_right = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (7, 7),
    x-tick-step: 64,
    y-tick-step: 64,
    y-min: 0,
    y-max: 256,
    x-min: 0,
    x-max: 256,
    x-label: [$u$ (Pixels)],
    y-label: [$v$ (Pixels)],
    {
      plot.add(
        domain: (0, 1), 
        style: (stroke: (thickness: 0pt, paint: red)),
        t => 1000 * (
          calc.sin(calc.pi * 1320 * t)
        )
      )
      plot.annotate({
        import cetz.draw: *
        rect((0, 0), (256, 256), fill: gray)
      })
      plot.annotate({
        import cetz.draw: *
        content((192, 192), image("figures/lecture_7/ghost_dog_bw.svg", width: 3cm))
      })
    })
})}

#let conv_signal_plot = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (20, 2.7),
    y-tick-step: none,
    x-tick-step: none,
    y-min: 0,
    y-max: 1,
    x-min: 0,
    x-max: 1,
    x-label: "t",
    y-label: "",
    {
      plot.add(
        domain: (0, 0.4), 
        label: $ x(t) $,
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 0.05 + 0.05 * (
          calc.sin(calc.pi * 50 * t)
        )
      )
      plot.add(
        domain: (0.4, 0.6), 
        style: (stroke: (thickness: 5pt, paint: red)),
        //t => (t - 0.4) / 0.3 + 0.05 * (
        //  calc.sin(calc.pi * 50 * t)
        //)
        t => 0.05 * calc.sin(calc.pi * 50 * t) + 0.8 * 1 / (1 + calc.exp(-30 * (t - 0.5)))
      )
      plot.add(
        domain: (0.6, 1.0), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 0.75 + 0.05 * (
          calc.sin(calc.pi * 50 * t)
        )
      )
    })
})}  

#let conv_filter_plot = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (20, 2.75),
    y-tick-step: none,
    x-tick-step: none,
    y-ticks: (0, 0.5),
    y-min: 0,
    y-max: 0.5,
    x-min: 0,
    x-max: 1,
    x-label: "t",
    y-label: "",
    {
      plot.add(
        domain: (0, 0.2), 
        label: $ g(t) $,
        style: (stroke: (thickness: 5pt, paint: blue)),
        t => 1 / (calc.pow(2  * 3.14, 0.5)) * calc.exp(-calc.pow((t - 0.1), 2) / 0.002)
      )
    })
})}  

#let conv_result_plot = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (20, 2.75),
    y-tick-step: none,
    x-tick-step: none,
    y-min: 0,
    y-max: 1,
    x-min: 0,
    x-max: 1,
    x-label: "t",
    y-label: "",
    {
      plot.add(
        domain: (0, 0.4), 
        label: $ x(t) * g(t) $,
        style: (stroke: (thickness: 5pt, paint: purple)),
        t => 0.05 
      )
      plot.add(
        domain: (0.4, 0.6), 
        style: (stroke: (thickness: 5pt, paint: purple)),
        t => 0.01 + 0.8 * 1 / (1 + calc.exp(-30 * (t - 0.5)))
      )
      plot.add(
        domain: (0.6, 1.0), 
        style: (stroke: (thickness: 5pt, paint: purple)),
        t => 0.77 
      )
    })
})}  


#let draw_filter(x, y, cells, colors: none) = {
  import cetz.draw: *

  grid((x, y), (x + cells.len(), y + cells.at(0).len()))
  
  for i in range(cells.len()) {
    for j in range(cells.at(i).len()) {
      
      if (colors != none)  {
        let cell_color = colors.at(cells.at(i).len() - j - 1).at(i)
        if (cell_color != none){
          rect((i, j), (i + 1, j + 1), fill: cell_color)
        }
        content((x + i + 0.4, y + j + 0.6), (x + i + 0.3, y + j + 0.8), cells.at(cells.at(i).len() - j - 1).at(i))

      } else {
        content((x + i + 0.4, y + j + 0.6), (x + i + 0.3, y + j + 0.8), str(cells.at(cells.at(i).len() - j - 1).at(i)))
      }

      }
  }
}


#let draw_conv = cetz.canvas({
  import cetz.draw: *


let filter_values = (
  (1, 0, 1),
  (0, 4, 0),
  (0, 0, 2)
)

let image_values = (
  (4, 4, 4, 4),
  (0, 0, 0, 0),
  (4, 0, 0, 4),
  (4, 0, 0, 4),
)
draw_filter(0, 0, image_values)
})

= Admin
==
Exam 1 graded, scores on moodle #pause
- Mean score: $91"/"120 approx 76%$ #pause
- If you finished exam but don't have a Moodle score come see me #pause

Course grades will be curved *up* to 85% #pause
- If everyone scores an A, this is fine #pause
- In practice, do better than exam mean for A #pause

*DO NOT CHEAT* #pause
- Failed cheaters on exam 1 #pause
  - *Course grade: F* #pause
- Lowest exam dropped #pause
  - Can score 0 on exam and get A in class
==
The following students got perfect marks! (120 / 120) #pause

- Chen Zhong Hong 
- Cui Shu Hang
- Gao Yu Chen
- Lai Yong Chao 
- Sheng Jun Wei 
- Wu Cheng Qin #pause

Great job! Please consider a career in deep learning research #pause

With perfect marks on exams 1 and 2, you can skip exam 3 #pause
- Lowest exam dropped

==
Exam 2 next week, 6 preliminary questions: #pause
- (1) Wide neural networks
- (1) Relationship between KL divergence and cross entropy loss
- (1) Vanishing gradient from activation
- (1) Modern optimizers (RMSProp, GD+momentum)
- (1) 1D discrete convolution
- (1) 1D convolutional neural networks #pause

I tell you exactly what to study, no need to cheat

==
Homework 1 and 2 graded, scores on moodle #pause
- Homework 1 mean score 94/100 #pause
- Homework 2 mean score 92/100 #pause

Homework 3 due the night before exam #pause
- Finish it soon so you can study!

==
New website clones and compiles `typst` docs in your browser! #pause

Follow my lectures along in real time

https://github.com/smorad/um_cisc_7026

= Review

==
    Dirty secret of deep learning #pause

    As networks become larger, deep learning changes #pause

    We understand the rules that govern small networks (physics) #pause

    When the system becomes complex, we lose understanding (biology) #pause

    Early deep learning was inductive #pause

    Modern deep learning is deductive
==
  To improve neural networks, we looked at: #pause
  - Deeper and wider networks #pause
  - New activation functions #pause
  - Parameter initialization #pause
  - New optimization methods

==
    A 2-layer neural network can represent *any* continuous function to arbitrary precision #pause

    $ | f(bold(x), bold(theta)) - g(bold(x)) | < epsilon $ #pause

    $ lim_(d_h -> oo) epsilon = 0 $ #pause

    However, finding such $bold(theta)$ is a much harder problem

==
    Gradient descent only guarantees convergence to a *local* optima #pause

    #cimage("figures/lecture_6/poor_minima.png", height: 80%)

==
  Deeper and wider networks often perform better #pause

  We call these *overparameterized* networks #pause

  "The probability of finding a “bad” (high value) local minimum is non-zero for small-size networks and decreases quickly with network size" - Choromanska, Anna, et al. _The loss surfaces of multilayer networks._ (2014)

==
  Next, we looked at activation functions

==
    #side-by-side(align: left)[#sigmoid #pause][
        The sigmoid function can result in a *vanishing gradient* #pause
        
        $ f(bold(x), bold(theta)) = sigma(bold(theta)_3^top sigma(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x))))) $ #pause
    ] 


    #only(4)[
    $ (gradient_bold(theta_1) f)(bold(x), bold(theta)) = 
    (gradient sigma)(bold(theta)_3^top sigma(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x)))))

    dot (gradient sigma)(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x))))

    dot (gradient sigma)(bold(theta)_1^top overline(bold(x))) dot overline(bold(x))
    $ 

    ]

    #only((5,6))[
    $ (gradient_bold(theta_1) f)(bold(x), bold(theta)) = 
    underbrace((gradient sigma)(bold(theta)_3^top sigma(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x))))), < 0.25)

    dot underbrace((gradient sigma)(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x)))), < 0.25)

    dot underbrace((gradient sigma)(bold(theta)_1^top overline(bold(x))), < 0.25) dot overline(bold(x))
    $ 
    ]

    #only(6)[
      #side-by-side[
      $ max_(bold(theta), bold(x)) (gradient_bold(theta_1) f)(bold(x), bold(theta)) approx 0.01 overline(bold(x)) $
      ][
      $ lim_(ell -> oo) (gradient_bold(theta_1) f)(bold(x), bold(theta)) = 0 $
      ]
    ]

==
    To fix the vanishing gradient, use the *rectified linear unit (ReLU)*
    #side-by-side[$ sigma(z) = max(0, z) \ gradient sigma(z) = cases(0 "if" z < 0, 1 "if" z >= 0) $ #pause][#relu #pause]

    "Dead" neurons always output 0, and cannot recover

==
    To fix dying neurons, use *leaky ReLU* #pause

    #side-by-side[$ sigma(z) = max(0.1 z, z) \ gradient sigma(z) = cases(0.1 "if" z < 0, 1 "if" z >= 0) $ #pause][#lrelu #pause]

    Small negative slope prevents dead neurons (nonzero slope everywhere) #pause

    As long as one neuron is positive in each layer, non-vanishing gradient

==
  #cimage("figures/lecture_6/activations.png")

==
  Then, we discussed parameter initialization #pause

  Initialization should: #pause
  + Be random to break symmetry within a layer #pause
  + Scale the parameters to prevent vanishing or exploding gradients #pause

  $ bold(theta) tilde cal(U)[ - sqrt(6) / sqrt(2 d_h), sqrt(6) / sqrt(2 d_h)] $ #pause

  $ bold(theta) tilde cal(U)[ - sqrt(6) / sqrt(d_x + d_y), sqrt(6) / sqrt(d_x + d_y)] $

==
  Finally, we reviewed improvements to optimization: #pause
  + Stochastic gradient descent #pause
  + Adaptive optimization #pause
  + Weight decay

==
  Stochastic gradient descent (SGD) reduces the memory usage #pause

  We cannot fit the full gradient in memory #pause

  In expectation, the stochastic gradient approximates the true gradient #pause

  $ bb(E)_(bold(X)_i tilde bold(X), bold(Y)_i tilde bold(Y)) [(gradient_bold(theta) cal(L))(bold(X)_i, bold(Y)_i, bold(theta))] = (gradient_bold(theta) cal(L))(bold(X), bold(Y), bold(theta)) $ #pause

  *Note:* Only equal under infinitely many samples $bold(X)_i, bold(Y)_i$ #pause
  - Approximate expectation using finite samples #pause

  $ hat(bb(E))_(bold(X)_i tilde bold(X), bold(Y)_i tilde bold(Y)) [(gradient_bold(theta) cal(L))(bold(X)_i, bold(Y)_i, bold(theta))] approx (gradient_bold(theta) cal(L))(bold(X), bold(Y), bold(theta)) $ 

==
  Stochastic gradient approximation adds noise during optimization #pause

  This noise can prevent premature convergence to bad optima #pause

  #cimage("figures/lecture_5/saddle.png")

==
    Use `torch.utils.data.DataLoader` for SGD #pause

    ```python
    import torch
    dataloader = torch.utils.data.DataLoader(
        training_data,
        batch_size=32, # How many datapoints to sample
        shuffle=True, # Randomly shuffle each epoch
    )
    for epoch in number_of_epochs:
        for batch in dataloader:
            X_j, Y_j = batch
            loss = L(X_j, Y_j, theta)
            ...
    ```

==
  Then, we looked at adaptive variants of gradient descent #pause

  #gd_algo()


==
    Introduce *momentum* #pause

    #gd_momentum_algo

==
    Introduce *adaptive learning rate* #pause

    #gd_adaptive_algo


==
  Combine *momentum* and *adaptive learning rate* to create *Adam* #pause

  #adam_algo

==
  Many modern optimizers also include *weight decay* #pause

  Weight decay penalizes large parameters #pause

  $ cal(L)_("decay")(bold(X), bold(Y), bold(theta)) = cal(L)_("decay")(bold(X), bold(Y), bold(theta)) + lambda sum_(i) theta_i^2 $ #pause

  Creates smoother, parabolic loss landscape that is easier to optimize 

==
Let's finish coding!

https://colab.research.google.com/drive/1qTNSvB_JEMnMJfcAwsLJTuIlfxa_kyTD

= Signal Processing

==
  Our neural networks do not consider the structure of data #pause

  $ bold(x) = vec(x_1, x_2, dots.v, x_(d_x)) $ #pause

  #side-by-side[$ bold(theta) = vec(theta_0, theta_1, dots.v, theta_(d_x)) $ #pause][$ bold(theta) eq.quest vec(theta_7, theta_4, dots.v, theta_1) $]

==
  We assume no relationship or ordering for input elements or parameter #pause

  $ sigma(theta_1 x_1 + theta_2 x_2) => sigma(theta_2 x_2 + theta_1 x_1) $

  #cimage("figures/lecture_7/permute.jpg") #pause

  These images are equivalent to a perceptron! 


==
  The real world is structured #pause

  Our models should use this structure #pause
  - Better data efficiency #pause
  - Better generalization #pause

  To do so, we will think of the world as a collection of signals


==
  A *signal* represents information as a function of time, space or some other variable #pause

  $ x(t) = 2 t + 1 $ #pause

  $ x(u, v) = u^2 / v - 3 $ #pause

  $x(t), x(u, v)$ represent physical processes that we may or may not know #pause

  In *signal processing*, we analyze the meaning of signals #pause

  Knowing the meaning of signals is very useful

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
  In signal processing, we often consider two properties: #pause
  - Locality #pause
  - Translation equivariance 

==
  *Locality:* Information structured and concentrated over small regions //$x(t)$ related to nearby neighbors, information content is nonuniform //Information concentrated over small regions of space/time #pause

  #side-by-side[
    $underbrace(x(t + epsilon) approx f(x(t)) + epsilon, "Nearby regions related")$
  ][
    $underbrace(y = f(x(s))\, quad s in (- epsilon, epsilon), "Information concentrated in small region")$
    //$ x(u + epsilon, v + epsilon) approx f(x(u, v)) + g(epsilon) $
  ] #pause

  #side-by-side[#waveform][#implot]
  
==
  *Translation Equivariance:* Shift in signal results in shift in output #pause

  $ f(x(t + tau)) = y(t + tau) $ #pause

  #side-by-side[#waveform_left][#waveform_right] #pause

  #align(center)[Both say "hello"]

==
  *Translation Equivariance:* Shift in signal results in shift in output #pause

  #side-by-side[#implot_left][#implot_right]

  #align(center)[Both contain a dog]

==
  Perceptrons are not local or translation equivariant, each pixel is an independent neuron #pause

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
  A more realistic scenario of locality and translation equivariance #pause

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
  content((4, 4), image("figures/lecture_7/flowers.jpg", width: 8cm))
  draw_filter(0, 0, image_values)
  }))


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
    (white, white, white, white, white, white, none, none),
    (white, white, white, white, white, white, none, none),
    (white, white, white, white, white, white, none, none),
    (white, white, none, none, none, white, white, white),
    (white, white, none, none, none, white, white, white),
    (white, white, none, none, white, white, white, white),
    (white, white, white, white, white, white, white, white),
  )
  content((5, 5), image("figures/lecture_7/flowers.jpg", width: 6cm))
  draw_filter(0, 0, image_values, colors: image_colors)
  })) #pause

  Our brains use the world's structure to infer missing information #pause

  Our neural networks should do the same


= Convolution
==
  In signal processing, we often turn signals into other signals #pause

  #side-by-side[#waveform_left][#hello] #pause

  A standard way to transform signals is *convolution* #pause

  Convolution is translation equivariant and local

==
  Convolution is the sum of products of a signal $x(t)$ and a *filter* $g(t)$ #pause

  If the t is continuous in $x(t)$

  $ x(t) * g(t) = integral_(-oo)^(oo) x(t - tau) g(tau) d tau $ #pause

  If the t is discrete in $x(t)$

  $ x(t) * g(t) = sum_(tau=-oo)^(oo) x(t - tau) g(tau) $ #pause

  We slide the filter $g(t)$ across the signal $x(t)$

==
*Note:* Convolution slides the filter from *right to left* #pause

  $ x(t) * g(t) &= integral_(-oo)^(oo) x(t - tau) g(tau) d tau &&= dots + x(t + 10) g(t) + x(t + 9) g(t) + dots \

  x(t) * g(t) &= sum_(tau=-oo)^(oo) x(t - tau) g(tau) &&= dots + x(t + 10) g(t) + x(t + 9) g(t) + dots $ #pause

  This is equivalent to flipping the filter and sliding left to right #pause

  $ g_"flip" (t) = g(-t) $

==

  *We will assume all filters $g(t)$ are "pre-flipped" in this course* #pause
  - Simply scan the filter $g(t)$ left to right #pause
  - Easier to understand and how we implement it in neural networks #pause

  $ x(t) * g(t) &= integral_(-oo)^(oo) x(t + tau) g(tau) d tau &&= dots + x(t + 1) g(1) + x(t + 2) g(2) + dots \

  x(t) * g(t) &= sum_(tau=-oo)^(oo) x(t + tau) g(tau) &&= dots + x(t + 1) g(1) + x(t + 2) g(2) + dots $ #pause

  Confused? Let us do an example



==
  *Example:* Let us examine a low-pass filter #pause

  The filter will take a signal and remove noise, producing a cleaner signal

==
  #conv_signal_plot #pause

  #conv_filter_plot #pause

  #conv_result_plot

==
  #conv_signal_plot

  #conv_filter_plot

  Convolution is *local* to the filter $g(t)$ #pause

  Convolution is also *equivariant* to time/space shifts

==
  We use continuous convolution for analog signals #pause
  - Physics #pause
  - Control theory #pause
  - Electrical engineering #pause

  Deep learning uses discrete convolution (digital hardware) #pause
  - Images #pause
  - Recorded audio #pause
  - Anything stored as bits #pause

  It is good to know both! Continuous for theory, discrete for software #pause

  Let us do a discrete example, I think it will make convolution very clear

==
  *Example:* Discrete convolution with filter size 2 #pause 

  $ y(t) = x(t) * g(t) = sum_(tau=0)^1 x(t + tau) g(tau) $  #pause

  $ y(t) = x(t) g(0) + x(t + 1) g(1) $ #pause

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

==

  $ y(t) = x(t) * g(t) = sum_(tau=0)^1 x(t + tau) g(tau) $  

  $ y(t) = x(t) g(0) + x(t + 1) g(1) $

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

==
  $ y(t) = x(t) * g(t) = sum_(tau=0)^1 x(t + tau) g(tau) $  

  $ y(t) = x(t) g(0) + x(t + 1) g(1) $

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

==
  $ y(t) = x(t) * g(t) = sum_(tau=0)^1 x(t + tau) g(tau) $  

  $ y(t) = x(t) g(0) + x(t + 1) g(1) $

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

==
  $ y(t) = x(t) * g(t) = sum_(tau=0)^1 x(t + tau) g(tau) $  

  $ y(t) = x(t) g(0) + x(t + 1) g(1) $
  
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

==
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

  *Question:* Does anybody see a connection to neural networks? #pause

  *Hint:* What if I rewrite the filter? #pause

  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    1, 2, 3, 4, 5 ;
    theta_1, theta_2 ;
    #hide[1]; 
  )
  $ 

==
  $ y(t) = x(t) * g(t) = sum_(tau=0)^1 x(t + tau) g(tau) $  

  $ y(t) = x(t) g(0) + x(t + 1) g(1) $

  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    #redm[$1$], #redm[$2$], 3, 4, 5 ;
    #redm[$theta_1$], #redm[$theta_2$] ;
    #redm[$theta_1 + 2 theta_2$] 
  )
  $ 

==
  $ y(t) = x(t) * g(t) = sum_(tau=0)^1 x(t + tau) g(tau) $  

  $ y(t) = x(t) g(0) + x(t + 1) g(1) $
  
  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    1, #redm[$2$], #redm[$3$], 4, 5 ;
    , #redm[$theta_1$], #redm[$theta_2$] ;
    theta_1 + 2 theta_2 , #redm[$2 theta_1 + 3 theta_2$]
  )
  $ #pause


==
  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    1, #redm[$2$], #redm[$3$], 4, 5 ;
    , #redm[$theta_1$], #redm[$theta_2$] ;
    theta_1 + 2 theta_2 , #redm[$2 theta_1 + 3 theta_2$]
  )
  $ #pause

  Just like neural networks, convolution is a linear operation #pause

  It is a weighted sum of the inputs, just like a neuron #pause

  $ x(t) * g(t) &= sum_(tau=0)^d_x x(t + tau) dot g(tau) \ 
  f(bold(x), bold(theta)) &= sum_(i=0)^d_x x_i dot theta_i
  $  

==
  $ x(t) * g(t) &= sum_(tau=0)^d_x x(t + tau) dot g(tau) \ 
  f(bold(x), bold(theta)) &= sum_(i=0)^d_x x_i dot theta_i
  $  

  *Question:* How does convolution differ from a neuron? #pause

  *Answer:* In a neuron, each input $x_i$ has a different parameter $theta_i$. In convolution, we reuse (slide) $theta_i$ over $x_1, x_2, dots$

= Convolutional Neural Networks

==
  Convolutional neural networks (CNNs) use convolutional layers #pause

  Their translation equivariance and locality make them very efficient #pause

  They also scale to variable-length sequences #pause

  Efficiently expands neural networks to images, videos, sounds, etc 

==
  CNNs have been around since the 1980's #pause

  #cimage("figures/lecture_1/timeline.svg") #pause

  2012: GPU and CNN efficiency resulted in breakthroughs

==
  So how does a convolutional neural network work?

==
  Recall the neuron #pause
  #side-by-side[Neuron][$ f(bold(x), bold(theta)) = sigma(bold(theta)^top overline(bold(x))) $] #pause

  #side-by-side[#waveform #pause][
  $ f(vec(x(0.1), x(0.2), dots.v), bold(theta)) = \ sigma(theta_0 + theta_1 x(0.1) + theta_2 x(0.2) + dots) $
  ]

==
  #side-by-side[#waveform][
  $ f(vec(x(0.1), x(0.2), dots.v), bold(theta)) = \ sigma(theta_0 + theta_1 x(0.1) + theta_2 x(0.2) + dots) $
  ] #pause

  *Question:* How many parameters do we need? #pause

  *Answer 1:* 10, parameters scale with sequence length #pause

  *Question:* What if sequence is 1.1 seconds long? #pause *A:* Train new network

==
  One parameter for each timestep does not work well #pause

  $ f(vec(x(0.1), x(0.2), dots.v), vec(theta_0, theta_1, dots.v)) \
  = sigma(theta_0 + theta_1 x(0.1) + theta_2 x(0.2) + theta_3 x(0.3) + theta_4 x(0.4) + theta_5 x(0.5) + dots) $
  #pause

  *Question:* How many parameters if speech is 2 hours long? #uncover("4-")[*A:* 72,000] #pause

  *Question:* What if speech is 2 hours and 0.1 seconds long? #pause

  *Answer:* Train a new neural network

==
  $ f(vec(x(0.1), x(0.2), dots.v), vec(theta_0, theta_1, dots.v)) \
  = sigma(theta_0 + theta_1 x(0.1) + theta_2 x(0.2) + theta_3 x(0.3) + theta_4 x(0.4) + theta_5 x(0.5) + dots) $
  #pause

  What if we reuse parameters?

  $ f(vec(x(0.1), x(0.2), dots.v), vec(theta_0, theta_1, theta_2)) = 
  vec(
    sigma(theta_0 + theta_1 x(0.1) + theta_2 x(0.2)),
    sigma(theta_0 + theta_1 x(0.2) + theta_2 x(0.3)),
    sigma(theta_0 + theta_1 x(0.3) + theta_2 x(0.4)),
    dots.v
  ) $ 

==
  $ f(vec(x(0.1), x(0.2), dots.v), vec(theta_0, theta_1, theta_2)) = 
  vec(
    sigma(theta_0 + theta_1 x(0.1) + theta_2 x(0.2)),
    sigma(theta_0 + theta_1 x(0.2) + theta_2 x(0.3)),
    sigma(theta_0 + theta_1 x(0.3) + theta_2 x(0.4)),
    dots.v
  ) $ #pause

  This is convolution, we slide our filter $g(t) = vec(theta_0, theta_1, theta_2)$ over the input $x(t)$

==
  We can write both a perceptron and convolution layer in similar forms

  #side-by-side[$ f(x(t), bold(theta)) = underbrace(sigma(bold(theta)^top vec(1, x(0.1), x(0.2), dots.v)), "Perceptron") $ #pause][
  $ f(x(t), bold(theta)) = underbrace(vec(
    sigma(bold(theta)^top vec(1, x(0.1), x(0.2))),
    sigma(bold(theta)^top vec(1, x(0.2), x(0.3))),
    dots.v
  ), "Convolution layer") $
  ]

  A convolution layer applies a "mini" perceptron at each timestep

==
  #side-by-side(align: left)[$ f(x(t), bold(theta)) = vec(
    sigma(bold(theta)^top vec(1, x(0.1), x(0.2))),
    sigma(bold(theta)^top vec(1, x(0.2), x(0.3))),
    dots.v
  ) $ #pause][
    Local, only considers a few nearby inputs at a time #pause

    Translation equivariant, each output corresponds to input 
  ]

==
  #side-by-side(align: left)[ $ f(x(t), bold(theta)) = vec(
    sigma(bold(theta)^top vec(1, x(0.1), x(0.2))),
    sigma(bold(theta)^top vec(1, x(0.2), x(0.3))),
    dots.v
  ) $ #pause ][
    *Question:* What is the shape of the results? #pause
  ]

  *Answer 1:* Depends on sequence length and filter size! #pause

  *Answer 2:* $T - k + 1$, where $T$ is sequence length and $k$ is filter length 

==
  Maybe we want to predict a single output, e.g. stock price tomorrow #pause

  #stonks #pause

  But convolutional layers output a variable-length signal ($T - k + 1$)!

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
  *Question:* $x(t)$ is a function, what is the function signature? #pause

  *Answer:* 

  $ x: bb(R)_+ |-> bb(R) $ #pause

  So far, we have considered: #pause
  - 1 dimensional variable $t$ #pause
  - 1 dimensional output/channel $x(t)$ #pause
  - 1 filter #pause

  We must consider a more general case #pause

  Things will get more complicated, but the core idea is exactly the same

= More Dimensions
==
  #side-by-side[#implot][
    *Question:* How many input dimensions for $x$? #pause

    *Answer:* 2: $u, v$ #pause

    *Question:* How many output dimensions/channels for $x$? #pause

    *Answer:* 1, black/white value
    ]

    $ x: underbrace({0 dots 255}, "width") times underbrace({0 dots 255}, "height") |-> underbrace([0, 1], "Color values") $

==
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

==
  #side-by-side[#implot_color][
    *Question:* How many input dimensions for $x$? #pause

    *Answer:* 2: $u, v$ #pause

    *Question:* How many output dimensions for $x$? #pause

    *Answer:* 3 -- red, green, and blue channels
    ]

    $ x: underbrace({0 dots 255}, "width") times underbrace({0 dots 255}, "height") |-> underbrace([0, 1]^3, "Color values") $

==
  #cimage("figures/lecture_7/ghost_dog_rgb.png") #pause

  Computers represent 3 color channel each with 256 integer values #pause

  But we usually convert the colors $x(u, v) in [0, 1]$ for scale reasons #pause

  $ mat(R / 255, G / 255, B / 255) $

==
  The pixels extend in 2 directions (height, width) $x(u, v)$ #pause

  Each pixel contains 3 colors (channels) $x(u, v) = mat(r, g, b)^top$ #pause

  We can store this signal as a tensor #pause

  $ bold(x) = 
    mat(
      underbrace(mat(
        .13, .14, .12, .10;
        .80, .40, .20, .05;
        .15, .08, .75, .16;
        .21, .38, .90, .78;
      ), "red"), 
      underbrace(mat(
        .15, .08, .75, .16;
        .80, .40, .20, .05;
        .21, .38, .90, .78;
        .13, .14, .12, .10;
      ), "green"), 
      underbrace(mat(
        .80, .40, .20, .05;
        .21, .38, .90, .78;
        .15, .08, .75, .16;
        .13, .14, .12, .10;
      ), "blue"), 
    )^top
  $ #pause

  This form is called $C H W$ (channel, height, width) format 

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
  The multi-dimension, multi-channel, multi-filter equations are very long, let's skip them #pause
  
  *Question:* What is the shape of $bold(theta)$ for a single layer? #pause

  *Answer:* 

  //$ bold(theta) in bb(R)^(overbrace((k + 1), "Filter" u) times overbrace(k, "Filter" v) times overbrace(d_x, "Input channels") times overbrace(d_y, "Output channels")) $

  $ bold(theta) in bb(R)^(c_x times c_y times k times k + c_y) $ #pause
  
  - Input channels: $c_x$
  - Output channels: $c_y + 1$ 
  - Filter $u$ (height): $k$
  - Filter $v$ (width): $k$ #pause

  Convolve $c_y$ filters of size $k times k$ across $c_x$ channels, bias for each output

==
  #side-by-side(align: left)[
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


= Deeper Networks
==
  #cimage("figures/lecture_1/timeline.svg") #pause

  AlexNet: Train a neural network on a GPU ($n=1.2m$, 6 days) #pause

  Paper by Krizhevsky, Sutskever (OpenAI CSO), and Hinton (Nobel)

==
  #cimage("figures/lecture_7/alexnet.png")

==
  #cimage("figures/lecture_7/alexnet-layers.png")

==
  Since AlexNet, there have been larger and better models #pause

  - VGG #pause
  - ResNet #pause
  - DenseNet #pause
  - MobileNet #pause
  - ResNeXt #pause

  ResNet, MobileNet, and ResNeXt are still used today! #pause

  Each one introduces a few tricks to obtain better results 

==
  After ResNet, marginal improvements #pause

  #cimage("figures/lecture_7/cnn_models.png")

= Coding
==
  ```python
  import torch
  c_x = 3 # Number of colors
  c_y = 32
  k = 2 # Filter size
  h, w = 128, 128 # Image size

  conv1 = torch.nn.Conv2d(
    in_channels=c_x, 
    out_channels=c_y,
    kernel_size=2
  )
  image = torch.rand((1, c_x, h, w)) # Torch requires BCHW
  out = conv1(image) # Shape(1, c_y, h - k, w - k)
  ```

==
  ```python
  import jax, equinox
  c_x = 3 # Number of colors
  c_y = 32
  k = 2 # Filter size
  h, w = 128, 128 # Image size
  conv1 = equinox.nn.Conv2d(
    in_channels=c_x, 
    out_channels=c_y,
    kernel_size=2,
    key=jax.random.key(0)
  )
  image = jax.random.uniform(jax.random.key(1), (c_x, h, w)) 
  out = conv1(image) # Shape(c_y, h - k, w - k)
  ```

==
  ```python
  import torch
  conv1 = torch.nn.Conv2d(3, c_h, 2)
  pool1 = torch.nn.AdaptiveAvgPool2d((a, a))
  conv2 = torch.nn.Conv2d(c_h, c_y, 2)
  pool2 = torch.nn.AdaptiveAvgPool2d((b, b))
  linear = torch.nn.Linear(c_y * b * b)
  z_1 = conv1(image)
  z_1 = torch.nn.functional.leaky_relu(z_1)
  z_1 = pool1(z_1) # Shape(1, c_h, a, a)
  z_2 = conv1(z1) 
  z_2 = torch.nn.functional.leaky_relu(z_2)
  z_2 = pool2(z_2) # Shape(1, c_y, b, b)
  z_3 = linear(z_2.flatten())
  ```

==
  ```python
  import jax, equinox
  conv1 = equinox.nn.Conv2d(3, c_h, 2)
  pool1 = equinox.nn.AdaptiveAvgPool2d((a, a))
  conv2 = equinox.nn.Conv2d(c_h, c_y, 2)
  pool2 = equinox.nn.AdaptiveAvgPool2d((b, b))
  linear = equinox.nn.Linear(c_y * b * b)
  z_1 = conv1(image,(3, h, w))
  z_1 = jax.nn.leaky_relu(z_1)
  z_1 = pool1(z_1) # Shape(c_h, a, a)
  z_2 = conv1(z1) 
  z_2 = jax.nn.leaky_relu(z_2)
  z_2 = pool2(z_2) # Shape(c_y, b, b)
  z_3 = linear(z_2.flatten())
  ```
==
  https://colab.research.google.com/drive/1IRRSsvdeC4a5AEWF_1WX9iveBqGSppn1#scrollTo=YVkCyz78x4Rp