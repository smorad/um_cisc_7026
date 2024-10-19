#import "@preview/cetz:0.2.2"
#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

// FUTURE TODO: Repeat self too much on locality/equivariance, restructure and replace with something else

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
        t => 100 * (0.3 * calc.sin(0.2 * t) +
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
    y-min: 0,
    y-max: 1,
    x-min: 0,
    x-max: 1,
    x-label: "t",
    y-label: "",
    {
      plot.add(
        domain: (0, 0.2), 
        label: $ g(t) $,
        style: (stroke: (thickness: 5pt, paint: blue)),
        t => calc.exp(-calc.pow((t - 0.1), 2) / 0.002)
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
        content((x + i + 0.4, y + j + 0.6), (i, j), cells.at(cells.at(i).len() - j - 1).at(i))

      } else {
        content((x + i + 0.4, y + j + 0.6), (i, j), str(cells.at(cells.at(i).len() - j - 1).at(i)))
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
/*
let colors = (
  red, red, red, red,
  red, red, red, red,
  red, red, red, red,
  red, red, red, red,
)
*/
draw_filter(0, 0, image_values)
//content((2, 2), image("figures/lecture_7/ghost_dog_bw.svg", width: 4cm))
})


// Signals, large continuous time inputs
// Want translation invariance and scalability to long sequences
// Introduce convolution (continuous)


#let ag = (
  [Review],
  [Signal Processing], 
  [Convolution],
  [Convolutional Neural Networks],
  [Additional Dimensions],
  [Deeper Networks],
  [Coding]
)


#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 7: Convolution"
)

#title-slide(
  title: [Convolution],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
)

#sslide[
  #side-by-side[#cimage("figures/lecture_7/hinton_nobel.jpeg")][John Hopfield and Geoffrey Hinton used tools from physics to construct methods that helped lay the foundation for today’s powerful machine learning. Machine learning based on artificial neural networks is currently revolutionising science, engineering and daily life.
]
]

#sslide[
  John Hopfield - Hopfield networks, an older type of neural network #pause

  Geoffrey Hinton - Most well-known for backpropagation #pause

  Geoffrey Hinton's student also created AlexNet, which we will discuss today
]

#sslide[
  https://www.youtube.com/watch?v=qrvK_KuIeJk
]

#sslide[
  #side-by-side[
    #cimage("figures/lecture_7/alphafold.png")
  ][Demis Hassabis and John Jumper have successfully utilised artificial intelligence to predict the structure of almost all known proteins. David Baker has learned how to master life’s building blocks and create entirely new proteins.]
]

#sslide[
  https://www.youtube.com/watch?v=gg7WjuFs8F4 #pause

  Alphafold solves a regression problem #pause

  $ X: {A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y}^k \ "Sequence of amino acids" $ #pause

  $ Y: bb(R)^(j times 3) \
  "3D coordinates (x, y, z) of each atom in the protein" $ #pause

  $ f: X times Theta |-> Y \ 
  f "is a neural network known as a transformer" $
]

#sslide[
  Hinton, Hassabis, and Jumper attended Cambridge!
]

#sslide[
  Plan for makeup lecture Saturday October 26 #pause

  Afternoon? Evening? #pause

  I will teach Autoencoders and Generative Models #pause
  
  Prof. Qingbiao Li will teach you Graph Neural Networks on Monday October 28
]

#aslide(ag, none)
#aslide(ag, 0)

#sslide[
    Dirty secret of deep learning #pause

    As networks become larger, deep learning changes #pause

    We understand the rules that govern small networks (physics) #pause

    One the system becomes sufficiently complex, we lose understanding (biology) #pause

    Modern deep learning is a science, where we advance through trial and error
]

#sslide[
  To improve neural networks, we looked at: #pause
  - Deeper and wider networks #pause
  - New activation functions #pause
  - Parameter initialization #pause
  - New optimization methods
]
#sslide[
    A 2-layer neural network can represent *any* continuous function to arbitrary precision #pause

    $ | f(bold(x), bold(theta)) - g(bold(x)) | < epsilon $ #pause

    $ lim_(d_h -> oo) epsilon = 0 $ #pause

    However, finding such $bold(theta)$ is a much harder problem
]
#sslide[
    Gradient descent only guarantees convergence to a *local* optima #pause

    #cimage("figures/lecture_6/poor_minima.png", height: 80%)
]

#sslide[
  Deeper and wider networks often perform better #pause

  We call these *overparameterized* networks #pause

  "The probability of finding a “bad” (high value) local minimum is non-zero for small-size networks and decreases quickly with network size" - Choromanska, Anna, et al. _The loss surfaces of multilayer networks._ (2014)

]
#sslide[
  #cimage("figures/lecture_6/filters.png", width: 120%)  
]

#sslide[
    ```python
    import torch
    d_x, d_y, d_h = 1, 1, 256
    net = torch.nn.Sequential(
        torch.nn.Linear(d_x, d_h),
        torch.nn.Sigmoid(),
        torch.nn.Linear(d_h, d_h),
        torch.nn.Sigmoid(),
        ...
        torch.nn.Linear(d_h, d_y),
    )

    x = torch.ones((d_x,))
    y = net(x)
    ```
]

#sslide[
    ```python
    import jax, equinox
    d_x, d_y, d_h = 1, 1, 256
    net = equinox.nn.Sequential([
        equinox.nn.Linear(d_x, d_h),
        equinox.nn.Lambda(jax.nn.sigmoid), 
        equinox.nn.Linear(d_h, d_h),
        equinox.nn.Lambda(jax.nn.sigmoid), 
        ...
        equinox.nn.Linear(d_h, d_y),
    ])

    x = jax.numpy.ones((d_x,))
    y = net(x)
    ```
]

#sslide[
  Next, we looked at activation functions
]
#sslide[
    #side-by-side[#sigmoid #pause][
        The sigmoid function can result in a *vanishing gradient* #pause
        
        $ f(bold(x), bold(theta)) = sigma(bold(theta)_3^top sigma(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x))))) $ #pause
    ] 


    #only(4)[
    $ gradient_bold(theta_1) f(bold(x), bold(theta)) = 
    gradient [sigma](bold(theta)_3^top sigma(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x)))))

    dot gradient[sigma](bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x))))

    dot gradient[sigma](bold(theta)_1^top overline(bold(x)))
    $ 

    ]

    #only((5,6))[
    $ gradient_bold(theta_1) f(bold(x), bold(theta)) = 
    underbrace(gradient [sigma](bold(theta)_3^top sigma(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x))))), < 0.5)

    dot underbrace(gradient[sigma](bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x)))), < 0.5)

    dot underbrace(gradient[sigma](bold(theta)_1^top overline(bold(x))), < 0.5)
    $ 
    ]

    #only(6)[
    $ gradient_bold(theta_1) f(bold(x), bold(theta)) approx 0 $
    ]
]

#sslide[
    To fix the vanishing gradient, use the *rectified linear unit (ReLU)*
    #side-by-side[$ sigma(x) = max(0, x) \ gradient sigma(x) = cases(0 "if" x < 0, 1 "if" x >= 0) $ #pause][#relu #pause]

    "Dead" neurons always output 0, and cannot recover
]

#sslide[
    To fix dead neurons, use *leaky ReLU* #pause

    #side-by-side[$ sigma(x) = max(0.1 x, x) \ gradient sigma(x) = cases(0.1 "if" x < 0, 1 "if" x >= 0) $ #pause][#lrelu #pause]

    Small negative slope allows dead neurons to recover
]

#sslide[
  #cimage("figures/lecture_6/activations.png")
]

#sslide[
  Then, we discussed parameter initialization #pause

  Initialization should: #pause
  + Be random to break symmetry within a layer #pause
  + Scale the parameters to prevent vanishing or exploding gradients #pause

  $ bold(theta) tilde cal(U)[ - sqrt(6) / sqrt(2 d_h), sqrt(6) / sqrt(2 d_h)] $ #pause

  $ bold(theta) tilde cal(U)[ - sqrt(6) / sqrt(d_x + d_y), sqrt(6) / sqrt(d_x + d_y)] $
]

#sslide[
  Finally, we reviewed improvements to optimization: #pause
  + Stochastic gradient descent #pause
  + Adaptive optimization #pause
  + Weight decay
]

#sslide[
  Stochastic gradient descent (SGD) reduces the memory usage #pause

  Rather than compute the gradient over the entire dataset, we approximate the gradient over a subset of the data 
]

#sslide[
  Stochastic gradient descent also inserts noise into the optimization process #pause

  This noise can prevent premature convergence to bad optima #pause

  #cimage("figures/lecture_5/saddle.png")
]

#sslide[
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
]

#sslide[
  Then, we looked at adaptive variants of gradient descent #pause

    #algorithm({
    import algorithmic: *

    Function("Gradient Descent", args: ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$), {

      Cmt[Randomly initialize parameters]
      Assign[$bold(theta)$][$"Glorot"()$] 

      For(cond: $i in 1 dots t$, {
        Cmt[Compute the gradient of the loss]        
        Assign[$bold(J)$][$gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta))$]
        Cmt[Update the parameters using the negative gradient]
        Assign[$bold(theta)$][$bold(theta) - alpha dot bold(J)$]
      })

    Return[$bold(theta)$]
    })
  })
]

#sslide[
    Introduce *momentum* first #pause

    #algorithm({
    import algorithmic: *

    Function(redm[$"Momentum"$] + " Gradient Descent", args: ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$, redm[$beta$]), {

      Assign[$bold(theta)$][$"Glorot"()$] 
      Assign[#redm[$bold(M)$]][#redm[$bold(0)$] #text(fill:red)[\# Init momentum]]

      For(cond: $i in 1 dots t$, {
        Assign[$bold(J)$][$gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta))$ #text(fill: red)[\# Represents acceleration]]
        Assign[#redm[$bold(M)$]][#redm[$beta dot bold(M) + (1 - beta) dot bold(J)$] #text(fill: red)[\# Momentum and acceleration]]
        Assign[$bold(theta)$][$bold(theta) - alpha dot #redm[$bold(M)$]$]
      })

    Return[$bold(theta)$]
    })
  })
]

#sslide[
    Now *adaptive learning rate* #pause

    #algorithm({
    import algorithmic: *

    Function(redm[$"RMSProp"$], args: ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$, $beta$, redm[$epsilon$]), {

      Assign[$bold(theta)$][$"Glorot"()$] 
      Assign[#redm[$bold(V)$]][#redm[$bold(0)$] #text(fill: red)[\# Init variance]] 

      For(cond: $i in 1 dots t$, {
        Assign[$bold(J)$][$gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta))$ \# Represents acceleration]
        Assign[#redm[$bold(V)$]][#redm[$beta dot bold(V) + (1 - beta) dot bold(J) dot.circle bold(J) $] #text(fill: red)[\# Magnitude]]
        Assign[$bold(theta)$][$bold(theta) - alpha dot #redm[$bold(J) ⊘ root(dot.circle, bold(V) + epsilon)$]$ #text(fill: red)[\# Rescale grad by prev updates]]
      })

    Return[$bold(theta)$]
    })
  })
]


#sslide[
    Combine *momentum* and *adaptive learning rate* to create *Adam* #pause

  #algorithm({
    import algorithmic: *

    Function("Adaptive Moment Estimation", args: ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$, greenm[$beta_1$], bluem[$beta_2$], bluem[$epsilon$]), {
      Assign[$bold(theta)$][$"Glorot"()$] 
      Assign[$#greenm[$bold(M)$], #bluem[$bold(V)$]$][$bold(0)$] 

      For(cond: $i in 1 dots t$, {
        Assign[$bold(J)$][$gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta))$]
        Assign[#greenm[$bold(M)$]][#greenm[$beta_1 dot bold(M) + (1 - beta_1) bold(J)$] \# Compute momentum]
        Assign[#bluem[$bold(V)$]][#bluem[$beta_2 dot bold(V) + (1 - beta_2) dot bold(J) dot.circle bold(J)$] \# Magnitude]

        Assign[$bold(theta)$][$bold(theta) - alpha dot #greenm[$bold(M)$] #bluem[$⊘ root(dot.circle, bold(V) + epsilon)$]$ \# Adaptive param update]
      })

    Return[$bold(theta)$ \# Note, we use biased $bold(M), bold(V)$ for clarity]
    })
  }) 
]

#sslide[
  Many modern optimizers also include *weight decay* #pause

  Weight decay penalizes large parameters #pause

  $ cal(L)_("decay")(bold(X), bold(Y), bold(theta)) = cal(L)_("decay")(bold(X), bold(Y), bold(theta)) + lambda sum_(i) theta_i^2 $ #pause

  This results in a smoother, parabolic loss landscape that is easier to optimize 
]

#aslide(ag, 0)
#aslide(ag, 1)

#sslide[
  In our networks, we do not consider the structure of data #pause

  $ bold(x) = vec(x_1, x_2, dots.v, x_(d_x)) $ #pause

  #side-by-side[$ bold(theta) = vec(theta_0, theta_1, dots.v, theta_(d_x)) $ #pause][$ bold(theta) eq.quest vec(theta_7, theta_4, dots.v, theta_1) $]
]

#sslide[
  We treat images as a vector, with no relationship between nearby pixels #pause

  These images are equivalent to a neural network

  #cimage("figures/lecture_7/permute.jpg") #pause

  It is a miracle that our neural networks could classify clothing!

  /*
  #side-by-side[
    $ mat(
      x_(1,1), x_(1, 2), x_(1, 3);
      x_(2,1), x_(2, 2), x_(2, 3);
      x_(3,1), x_(3, 2), x_(3, 3);
    ) $ #pause
    ][
    $ vec(
      x_(1,1), x_(1, 2), x_(1, 3), x_(2,1), dots.v
    ) $ #pause
    ][
    $ vec(theta_0, theta_1, theta_2, theta_3, dots.v) $
  ]*/
]

#sslide[
  There is structure inherent in the real world #pause

  By representing this structure, we can make more efficient neural networks that learn faster and generalize better #pause

  To do so, we will think of the world as a collection of signals
]



#slide(title: [Signal Processing])[
  A *signal* represents information as a function of time, space or some other variable #pause

  $ x(t) = 2 t + 1 $ #pause

  $ x(u, v) = u^2 / v - 3 $ #pause

  $x(t), x(u, v)$ represent physical processes that we may or may not know #pause

  In *signal processing*, we analyze the meaning of signals #pause

  Knowing the meaning of signals is very useful
]

#slide(title: [Signal Processing])[
  $ x(t) = "stock price" $ #pause

  #align(center, stonks) #pause

  There is an underlying structure to $x(t)$ #pause

  *Structure:* Tomorrow's stock price will be close to today's stock price
]

#slide(title: [Signal Processing])[
  $ x(t) = "audio" $ #pause

  #align(center, waveform) #pause

  *Structure:* Nearby waves form syllables #pause

  *Structure:* Nearby syllables combine to create meaning 

]

#slide(title: [Signal Processing])[
  $ x(u, v) = "image" $ #pause
  
  #align(center, implot) #pause

  *Structure:* Repeated components (circles, symmetry, eyes, nostrils, etc)
]

#slide(title: [Signal Processing])[
  In signal processing, we often consider two properties: #pause
  - Locality #pause
  - Translation equivariance 
]

#slide(title: [Signal Processing])[
  *Locality:* Information concentrated over small regions of space/time #pause

  #side-by-side[#waveform][#implot]
  
  // TODO Add output signal
]

#slide(title: [Signal Processing])[
  *Translation Equivariance:* Shift in signal results in shift in output #pause

  #side-by-side[#waveform_left][#waveform_right] #pause

  #align(center)[Both say "hello"]
]

#slide(title: [Signal Processing])[
  *Translation Equivariance:* Shift in signal results in shift in output #pause

  #side-by-side[#implot_left][#implot_right]

  #align(center)[Both contain a dog]
]

#slide(title: [Signal Processing])[
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
]

#slide(title: [Signal Processing])[
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
  })) #pause

  //How can we represent these properties in neural networks?
]

#slide(title: [Signal Processing])[
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

  //How can we represent these properties in neural networks?
]



#aslide(ag, 1)
#aslide(ag, 2)


#slide(title: [Convolution])[
  In signal processing, we often turn signals into other signals #pause

  #side-by-side[#waveform_left][#hello] #pause

  A standard way to transform signals is *convolution* #pause

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
  *Example:* Let us examine a low-pass filter #pause

  The filter will take a signal and remove noise, producing a cleaner signal
]

#slide(title: [Convolution])[

  #conv_signal_plot #pause

  #conv_filter_plot #pause

  #conv_result_plot
]


#slide(title: [Convolution])[

  #conv_signal_plot

  #conv_filter_plot

  Convolution is *local* to the filter $g(t)$ #pause

  Convolution is also *equivariant* to time/space shifts
]

#slide(title: [Convolution])[
  Often, we use continuous time/space convolution for analog signals #pause
  - Physics #pause
  - Control theory #pause
  - Electrical engineering #pause

  Almost all deep learning occurs on digital hardware (discrete time/space) #pause
  - Images #pause
  - Recorded audio #pause
  - Anything stored as bits #pause

  But it is good to know both! Continuous variables for theory. Discrete variables for software 
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

  *Question:* Does anybody see a connection to neural networks? #pause

  *Hint:* What if I rewrite the filter? #pause

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


#slide(title: [Convolution])[

  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    #redm[$1$], #redm[$2$], 3, 4, 5 ;
    #redm[$theta_2$], #redm[$theta_1$] ;
    #redm[$theta_2 + 2 theta_1$] 
  )
  $ 

]

#sslide[

  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    1, #redm[$2$], #redm[$3$], 4, 5 ;
    , #redm[$theta_2$], #redm[$theta_1$] ;
    theta_2 + 2 theta_1 , #redm[$2 theta_2 + 3 theta_1$]
  )
  $ #pause


  #side-by-side[#waveform_left][#hello]
]

#sslide[
  $
  vec(
    x(t),
    g(t),
    y(t)
  ) = mat(
    1, #redm[$2$], #redm[$3$], 4, 5 ;
    , #redm[$theta_2$], #redm[$theta_1$] ;
    theta_2 + 2 theta_1 , #redm[$2 theta_2 + 3 theta_1$]
  )
  $ #pause

  Just like neural networks, convolution is a linear operation #pause

  It is a weighted sum of the inputs, just like a neuron #pause

  *Question:* How does convolution differ from a neuron? #pause

  *Answer:* In a neuron, each input $x_i$ has a different parameter $theta_i$. In convolution, we reuse (slide) $theta_i$ over $x_1, x_2, dots$
]

#aslide(ag, 2)
#aslide(ag, 3)

#sslide[
  Convolutional neural networks (CNNs) use convolutional layers #pause

  Their translation equivariance and locality make them very efficient #pause

  They also scale to variable-length sequences #pause

  Efficiently expands neural networks to images, videos, sounds, etc 
]

#sslide[
  CNNs have been around since the 1980's #pause

  #cimage("figures/lecture_1/timeline.svg") #pause

  2012: GPU and CNN efficiency resulted in breakthroughs
]

#sslide[
  So how does a convolutional neural network work?
]

#sslide[
  Recall the neuron #pause
  #side-by-side[Neuron for single $x$:][$ f(x, bold(theta)) = sigma(theta_1 x + theta_0) $] #pause

  #side-by-side[#waveform #pause][
  $ f(vec(x(0.1), x(0.2), dots.v), bold(theta)) = \ sigma(theta_0 + theta_1 x(0.1) + theta_2 x(0.2) + dots) $
  ]
]

#sslide[
  #side-by-side[#waveform][
  $ f(vec(x(0.1), x(0.2), dots.v), bold(theta)) = \ theta_0 + theta_1 x(0.1) + theta_2 x(0.2) + dots $
  ] #pause

  *Question:* How many parameters do we need? #pause

  *Answer 1:* 10, parameters scale with sequence length #pause

  *Question:* What if our sequence is 1.1 seconds long?
]

#sslide[
  One parameter for each timestep does not work well #pause

  $ f(vec(x(0.1), x(0.2), dots.v), vec(theta_0, theta_1, dots.v)) \
  = sigma(theta_0 + theta_1 x(0.1) + theta_2 x(0.2) + theta_3 x(0.3) + theta_4 x(0.4) + theta_5 x(0.5) + dots) $
  #pause

  *Question:* How many parameters if speech is 2 hours long? #uncover("4-")[72,000] #pause

  *Question:* What if speech is 2 hours and 0.1 seconds long? #pause

  *Answer:* Train a new neural network

]

#sslide[
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
  ) $ #pause
]

#sslide[
  $ f(vec(x(0.1), x(0.2), dots.v), vec(theta_0, theta_1, theta_2)) = 
  vec(
    sigma(theta_0 + theta_1 x(0.1) + theta_2 x(0.2)),
    sigma(theta_0 + theta_1 x(0.2) + theta_2 x(0.3)),
    sigma(theta_0 + theta_1 x(0.3) + theta_2 x(0.4)),
    dots.v
  ) $ #pause

  This is convolution, we slide our filter $g(t) = vec(theta_0, theta_1, theta_2)$ over the input $x(t)$
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

  A convolution layer applies a "mini" perceptron to every few timesteps
]


#sslide[
  #side-by-side[$ f(x(t), bold(theta)) = vec(
    sigma(bold(theta)^top vec(1, x(0.1), x(0.2))),
    sigma(bold(theta)^top vec(1, x(0.2), x(0.3))),
    dots.v
  ) $ #pause][
    Local, only considers a few nearby inputs at a time #pause

    Translation equivariant, each output corresponds to input 
  ]
]

#sslide[
  #side-by-side[ $ f(x(t), bold(theta)) = vec(
    sigma(bold(theta)^top vec(1, x(0.1), x(0.2))),
    sigma(bold(theta)^top vec(1, x(0.2), x(0.3))),
    dots.v
  ) $ #pause ][
    *Question:* What is the shape of the results? #pause
  ]

  *Answer 1:* Depends on sequence length and filter size! #pause

  *Answer 2:* $T - k + 1$, where $T$ is sequence length and $k$ is filter length 
]

#sslide[
  Maybe we want to predict a single output for a sequence #pause

  #stonks
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
  *Question:* $x(t)$ is a function, what is the function signature? #pause

  *Answer:* 

  $ x: bb(R)_+ |-> bb(R) $ #pause

  So far, we have considered: #pause
  - 1 dimensional variable $t$ #pause
  - 1 dimensional output/channel $x(t)$ #pause
  - 1 filter #pause

  We must consider a more general case #pause

  Things will get more complicated, but the core idea is exactly the same
]

#aslide(ag, 3)
#aslide(ag, 4)

#sslide[
  #side-by-side[#implot][
    *Question:* How many input dimensions for $x$? #pause

    *Answer:* 2: $u, v$

    *Question:* How many output dimensions/channels for $x$? #pause

    *Answer:* 1, black/white value
    ]

    $ x: underbrace(bb(Z)_(0, 255), "width") times underbrace(bb(Z)_(0, 255), "height") |-> underbrace([0, 1], "Color values") $
]

#sslide[
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

#sslide[
  #side-by-side[#implot_color][
    *Question:* How many input dimensions for $x$? #pause

    *Answer:* 2: $u, v$

    *Question:* How many output dimensions for $x$? #pause

    *Answer:* 3 -- red, green, and blue channels
    ]

    $ x: underbrace(bb(Z)_(0, 255), "width") times underbrace(bb(Z)_(0, 255), "height") |-> underbrace([0, 1]^3, "Color values") $
]

#sslide[
  #cimage("figures/lecture_7/ghost_dog_rgb.png") #pause

  Computers represent 3 color channel each with 256 integer values #pause

  But we usually convert the colors $x(u, v) in [0, 1]$ for scale reasons #pause

  $ mat(R / 255, G / 255, B / 255) $
]

#sslide[

  The pixels extend in 2 directions (input) $x(u, v)$ #pause

  Each pixel contains 3 colors (channels) $x(u, v) = mat(r, g, b)^top$ #pause

  And the pixels extend in 2 directions (variables) #pause

  $ bold(x)(u, v) = 
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

  This form is called $C H W$ (channel, height, width) format #pause

  //Convolutional filter must process this data!
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
  I will not bore you with the full equations #pause

  *Question:* What is the shape of $bold(theta)$ for a single layer? #pause

  *Answer:* 

  //$ bold(theta) in bb(R)^(overbrace((k + 1), "Filter" u) times overbrace(k, "Filter" v) times overbrace(d_x, "Input channels") times overbrace(d_y, "Output channels")) $

  $ bold(theta) in bb(R)^(c_x times c_y times k times k + c_y) $
  
  - Input channels: $c_x$
  - Output channels: $c_y$
  - Filter $u$ (height): $k + 1$
  - Filter $v$ (width): $k$ #pause

  Convolve $c_y$ filters of size $k times k$ across $c_x$ channels, bias for each output
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

#aslide(ag, 4)
#aslide(ag, 5)

#sslide[
  #cimage("figures/lecture_1/timeline.svg") #pause

  AlexNet: Train a neural network on a GPU ($n=1.2m$, 6 days) #pause

  Paper by Krizhevsky, Sutskever (OpenAI CSO), and Hinton (Nobel)
]

/*
#sslide[
  AlexNet created by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton #pause

  - Hinton is a godfather of deep learning #pause
  - Sutskever was cofounder/chief scientist at OpenAI #pause
  - Krizhevsky worked at Google research

  AlexNet was one of the first neural networks to run on a GPU #pause

  Trained on 1.2M images over 6 days #pause

]

#sslide[
  The ideas behind AlexNet were not new, it was primarily the *scale* that made it interesting #pause

  It was a very big network at the time #pause

  It was trained longer than other networks #pause

  Only possible using special hardware (GPUs)
]
*/

#sslide[
  #cimage("figures/lecture_7/alexnet.png")
]

#sslide[
  #cimage("figures/lecture_7/alexnet-layers.png")
]

#sslide[
  Since AlexNet, there have been larger and better models #pause

  - VGG #pause
  - ResNet #pause
  - DenseNet #pause
  - MobileNet #pause
  - ResNeXt #pause

  ResNet, MobileNet, and ResNeXt are still used today! #pause

  Each one introduces a few tricks to obtain better results 
]

#sslide[
  After ResNet, marginal improvements #pause

  #cimage("figures/lecture_7/cnn_models.png")

]

#aslide(ag, 5)
#aslide(ag, 6)

#sslide[
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
]


#sslide[
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
]

#sslide[
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
]

#sslide[
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
]

#sslide[
  https://colab.research.google.com/drive/1IRRSsvdeC4a5AEWF_1WX9iveBqGSppn1#scrollTo=YVkCyz78x4Rp
]