#import "@preview/cetz:0.2.2"
#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm


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
    x-label: "Time (Days)",
    y-label: "Stock Price (MOP)",
    {
      plot.add(
        domain: (0, 5), 
        label: $ x(t) $,
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
    x-label: "Time (Seconds)",
    y-label: "Frequency (Hz)",
    {
      plot.add(
        domain: (0, 1), 
        label: $ x(t) $,
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
    y-label: "Frequency (Hz)",
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

#aslide(ag, none)
#aslide(ag, 0)

#slide(title: [Review])[

]

#aslide(ag, 0)
#aslide(ag, 1)



#slide(title: [Signal Processing])[
  So far, we have not considered the structure of inputs $X$ #pause

  We treat images as a vector, with no relationship between neighboring pixels #pause

  Neurons $i$ in layer $ell$ has no relationship to neuron $j$ in layer $ell$ #pause

  However, there is structure inherent in the real world #pause

  By representing this structure, we can make more efficient neural networks that generalize better #pause

  To do so, we must think of the world as a collection of signals
]



#slide(title: [Signal Processing])[
  A *signal* represents information as a function of time, space or some other variable #pause

  $ x(t) = dots $ 

  $ x(u, v) = dots $ #pause

  $x(t), x(u, v)$ are some physical processes that we may or may not know #pause

  *Signal processing* is a field of research that focuses on analyzing the meaning of signals #pause

  Knowing the meaning of signals is very useful
]

#slide(title: [Signal Processing])[
  #cimage("figures/lecture_7/stonks.jpg", height: 80%) #pause

  $ x(t) = "stock price" $ 

]

#slide(title: [Signal Processing])[
  $ x(t) = "stock price" $

  #align(center, stonks) #pause

  There is an underlying structure that we do not fully understand #pause

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
  In signal processing, we often consider: #pause
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
  })) #pause
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
  })) #pause

  //How can we represent these properties in neural networks?
]



#aslide(ag, 1)
#aslide(ag, 2)


#slide(title: [Convolution])[
  In signal processing, we often turn signals into other signals #pause

  #side-by-side[#waveform_left][#hello] #pause

  A standard way to transform signals is *convolution* #pause

  Convolution is translation equivariant and can be local
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

  Almost all deep learning occurs on digital hardware (discrete time/space)
  - Images #pause
  - Quantized audio #pause
  - Anything stored as bits instead of a function #pause

  But it is good to know both! Continuous variables for theory. Discrete variables for software 
]

#slide(title: [Convolution])[
  $
  vec(
    g(t),
    x(t),
    y(t)
  ) = mat(
    2, 1;
    1, 2, 3, 4, 5;
    space ; 
  )
  $
]

#slide(title: [Convolution])[
  $
  vec(
    g(t),
    x(t),
    y(t)
  ) = mat(
    #redm[$2$], #redm[$1$] ;
    #redm[$1$], #redm[$2$], 3, 4, 5;
    #redm[$4$]; 
  )
  $
]

#slide(title: [Convolution])[
  $
  vec(
    g(t),
    x(t),
    y(t)
  ) = mat(
    , #redm[$2$], #redm[$1$] ;
    1, #redm[$2$], #redm[$3$], 4, 5;
    4, #redm[$5$]; 
  )
  $
]

#slide(title: [Convolution])[
  $
  vec(
    g(t),
    x(t),
    y(t)
  ) = mat(
    , , #redm[$2$], #redm[$1$] ;
    1, 2, #redm[$3$], #redm[$4$], 5;
    4, 5, #redm[$10$]; 
  )
  $
]

#slide(title: [Convolution])[
  $
  vec(
    g(t),
    x(t),
    y(t)
  ) = mat(
    , , , #redm[$2$], #redm[$1$] ;
    1, 2, 3, #redm[$4$], #redm[$5$] ;
    4, 5, 10, #redm[$13$]; 
  )
  $
]


#slide(title: [Convolution])[
  $
  vec(
    g(t),
    x(t),
    y(t)
  ) = mat(
    2, 1 ;
    1, 2, 3, 4, 5 ;
    4, 5, 10, 13; 
  )
  $ #pause

  *Question:* Does anybody see a connection to neural networks? #pause

  *Hint:* What if I rewrite the filter? #pause

  $
  vec(
    g(t),
    x(t),
    y(t)
  ) = mat(
    theta_2, theta_1 ;
    1, 2, 3, 4, 5 ;
    4, 5, 10, 13; 
  )
  $ 
]


#slide(title: [Convolution])[

  $
  vec(
    g(t),
    x(t),
    y(t)
  ) = mat(
    #redm[$theta_2$], #redm[$theta_1$] ;
    #redm[$1$], #redm[$2$], 3, 4, 5 ;
    #redm[$theta_2 + 2 theta_1$] , 5, 10, 13; 
  )
  $ 

]

#slide(title: [Convolution])[

  $
  vec(
    g(t),
    x(t),
    y(t)
  ) = mat(
    , #redm[$theta_2$], #redm[$theta_1$] ;
    1, #redm[$2$], #redm[$3$], 4, 5 ;
    theta_2 + 2 theta_1 , #redm[$2 theta_2 + 3 theta_1$], 10, 13; 
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
  CNNs have been around since the 1970's #pause

  #cimage("figures/lecture_1/timeline.svg") #pause

  2012: GPU and CNN efficiency resulted in breakthroughs
]

#sslide[
  So how does a convolutional neural network work? #pause

  Like before, we will start with linear functions and derive a convolutional layer
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

  *Question:* Any problems besides locality/equivariance? #pause

  *Answer 1:* Parameters scale with sequence length #pause

  *Answer 2:* Parameters only for exactly 1 second waveforms
]

#sslide[
  To fix problems, each timestep cannot use different parameters #pause

  $ f(vec(x(0.1), x(0.2), dots.v), vec(theta_0, theta_1, dots.v)) \
  = sigma(theta_0 + theta_1 x(0.1) + theta_2 x(0.2) + theta_3 x(0.3) + theta_4 x(0.4) + theta_5 x(0.5) + dots) $
  #pause

  $ f(vec(x(0.1), x(0.2), dots.v), vec(theta_0, theta_1, theta_2)) = 
  vec(
    sigma(theta_0 + theta_1 x(0.1) + theta_2 x(0.2)),
    sigma(theta_0 + theta_1 x(0.2) + theta_2 x(0.3)),
    sigma(theta_0 + theta_1 x(0.3) + theta_2 x(0.4)),
    dots.v
  ) $
]


#sslide[
  $ f(vec(x(0.1), x(0.2), dots.v), vec(theta_0, theta_1, theta_2)) = 
  vec(
    sigma(theta_0 + theta_1 x(0.1) + theta_2 x(0.2)),
    sigma(theta_0 + theta_1 x(0.2) + theta_2 x(0.3)),
    sigma(theta_0 + theta_1 x(0.3) + theta_2 x(0.4)),
    dots.v
  ) $

  This is a convolutional layer! #pause
    - Local, only considers two inputs at a time #pause
    - Translation equivariant, each output corresponds to input #pause

]

#sslide[
  We can write both the neuron and convolution in vector form

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
  #side-by-side[ $ f(x(t), bold(theta)) = vec(
    sigma(bold(theta)^top vec(1, x(0.1), x(0.2))),
    sigma(bold(theta)^top vec(1, x(0.2), x(0.3))),
    dots.v
  ) $ #pause ][
    *Question:* What is the shape of the results? #pause
  ]

  *Answer 1:* Depends on sampling rate and filter size! #pause

  *Answer 2:* $T - k$, where $T$ is sequence length and $k$ filter length 
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

  $ "MeanPool"(z(t)) = 1 / T "SumPool"(z(t))
  $
]

#sslide[
  *Question:* $x(t)$ is a function, what is the function signature? #pause

  *Answer:* 

  $ x: bb(R)_+ |-> bb(R) $ #pause

  So far, we have considered: #pause
  - 1 dimensional variable $t$ #pause
  - 1 dimensional input/channel $x(t)$ #pause
  - 1 filter #pause

  We must consider a more general case #pause

  Things will get more complicated, but the core idea is exactly the same
]

#aslide(ag, 3)
#aslide(ag, 4)

#sslide[
  #side-by-side[#implot][
    *Question:* How many input dimensions for $x$? #pause

    *Answer:* 2, $u, v$

    *Question:* How many output dimensions for $x$? #pause

    *Answer:* 1, black/white value
    ]

    $ x: underbrace(bb(Z)_(0, 255), "width") times underbrace(bb(Z)_(0, 255), "height") |-> underbrace(bb(Z)_(0, 255), "Color values") $
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

    *Answer:* 2, $u, v$

    *Question:* How many output dimensions for $x$? #pause

    *Answer:* 3 -- red, green, and blue channels
    ]

    $ x: underbrace(bb(Z)_(0, 255), "width") times underbrace(bb(Z)_(0, 255), "height") |-> underbrace([0, 1]^3, "Color values") $
]

#sslide[
  #cimage("figures/lecture_7/ghost_dog_rgb.png") #pause

  Computers represent 3 color channels each with 256 integer values #pause

  But we usually convert the colors to be in $[0, 1]$ for scale reasons #pause

  $ mat(R / 255, G / 255, B / 255) $
]

#sslide[

  Each pixel contains 3 colors (channels) #pause

  And the pixels extend in 2 directions (variable) #pause

  $ bold(x)(u, v) = 
    mat(
      underbrace(mat(
        130, 140, 120, 103;
        80, 140, 120, 105;
        130, 140, 75, 165;
        210, 140, 90, 150;
      ), "red"), 
      underbrace(mat(
        130, 140, 75, 165;
        210, 140, 90, 150;
        130, 140, 120, 103;
        80, 140, 120, 105;
      ), "green"), 
      underbrace(mat(
        210, 140, 90, 150;
        130, 140, 75, 165;
        110, 140, 120, 103;
        80, 140, 120, 105;
      ), "blue"), 
    )^top
  $ #pause

  This form is called $C H W$ (channel, height, width) format

  Convolutional filter must process this data!
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
      (1, 0),
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

  $ bold(theta) in bb(R)^(c_x times c_y times (k + 1) times k) $
  
  - Input channels: $c_x$
  - Output channels: $c_y$
  - Filter $u$ (height): $k + 1$
  - Filter $v$ (width): $k$
]

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
  #side-by-side[
    One last thing, stride allows you to "skip" cells during convolution #pause

    This can decrease the size of image without pooling #pause
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
  pool1 = torch.nn.AdaptivePool2d((a, a))
  conv2 = torch.nn.Conv2d(c_h, c_y, 2)
  pool2 = torch.nn.AdaptivePool2d((b, b))
  linear = torch.nn.Linear(c_y * b * b)
  z_1 = conv1(image)
  z_1 = torch.nn.functional.leaky_relu(z_1)
  z_1 = pool(z_1) # Shape(1, c_h, a, a)
  z_2 = conv1(z1) 
  z_2 = torch.nn.functional.leaky_relu(z_2)
  z_2 = pool(z_2) # Shape(1, c_y, b, b)
  z_3 = linear(z_2.flatten())
  ```
]

#sslide[
  ```python
  import jax, equinox
  conv1 = equinox.nn.Conv2d(3, c_h, 2)
  pool1 = equinox.nn.AdaptivePool2d((a, a))
  conv2 = equinox.nn.Conv2d(c_h, c_y, 2)
  pool2 = equinox.nn.AdaptivePool2d((b, b))
  linear = equinox.nn.Linear(c_y * b * b)
  z_1 = conv1(image,(3, h, w))
  z_1 = jax.nn.leaky_relu(z_1)
  z_1 = pool(z_1) # Shape(c_h, a, a)
  z_2 = conv1(z1) 
  z_2 = jax.nn.leaky_relu(z_2)
  z_2 = pool(z_2) # Shape(c_y, b, b)
  z_3 = linear(z_2.flatten())
  ```
]

#sslide[
  Single channel, single filter, single variable, $bold(theta) in bb(R)^(k + 1), k = 2 $ #pause

  $ f(x(t), bold(theta)) = mat(
    sigma(bold(theta)^top vec(1, x(1), x(2))),
    sigma(bold(theta)^top vec(1, x(2), x(3))),
    dots
  )^top $

  Single channel, single filter, *two variables*, $bold(theta) in bb(R)^(2 k + 1), k = 2 $ #pause

  $ f(x(t), bold(theta)) = mat(
    sigma(bold(theta)^top mat(
      1, 0, 0; 
      x(1, 1), x(1, 2), x(1, 3);
      x(2, 1), x(2, 2), x(2, 3)
    )),
    sigma(bold(theta)^top mat(
      1, 0, 0; 
      x(2, 1), x(2, 2), x(2, 3);
      x(3, 1), x(3, 2), x(3, 3)
    )),
  )^top $

  /*
  Three input channels, single filter, $bold(theta) in bb(R)^(d_x  times (k + 1)), d_x = 3, k = 2$ #pause

  $ f(x(u, v), bold(theta)) = mat(
    sigma(bold(theta)^top mat(
      1, 1, 1; 
      x_1(1), x_2(1), x_3(1);
      x_1(2), x_2(2), x_3(2)
    )),
    sigma(bold(theta)^top mat(
      1, 1, 1; 
      x_1(2), x_2(2), x_3(2);
      x_1(3), x_2(3), x_3(3)
    )),
    dots
  )^top $
  */
]

#sslide[
  *Three channels*, single filter, two variables, $bold(theta) in bb(R)^(2 k + 1), k = 2 $ 

  $ f_r (x(t), bold(theta)) = mat(
    sigma(bold(theta)_r^top mat(
      1, 0, 0; 
      x(1, 1), x(1, 2), x(1, 3);
      x(2, 1), x(2, 2), x(2, 3)
    )),
    sigma(bold(theta)_r^top mat(
      1, 0, 0; 
      x(2, 1), x(2, 2), x(2, 3);
      x(3, 1), x(3, 2), x(3, 3)
    )),
  )^top $ #pause

]

  
#sslide[
  We only considered one filter #pause

  In a perceptron, many parallel neurons makes a wide neural network #pause

  In a CNN, many parallel filters makes a wide CNN #pause

  $ bold(z)(t) = f(x(t), bold(theta)) = mat(
    sigma(bold(theta)^top vec(1, x(0.1), x(0.2))),
    sigma(bold(theta)^top vec(1, x(0.2), x(0.3))),
    dots
  )^top $ #pause 
]

#slide(title: [Convolution])[
  Neuron:
    $ sigma(bold(theta)_1^top overline(bold(x))(1) 
    + bold(theta)_2^top overline(bold(x))(2) + dots) 
    
    = sigma(sum_(i=0)^d_x theta_(1,i) overline(x)_i +
    sum_(i=0)^d_x theta_(2,i) overline(x)_i + 
    dots) $ #pause

  Convolution: 
  $ bold(theta)_1^top overline(bold(x))(t) + bold(theta)_2^top overline(bold(x))(t + 1) = (sum_(i=0)^d_x theta_(1, i) overline(x)_i (t)) + (sum_(i=0)^d_x theta_(2, i) overline(x)_i (t + 1)) $ #pause


  $ vec(bold(theta_1), bold(theta_2)) * overline(bold(x))(t) = mat(
    bold(theta)_1^top overline(bold(x)) (0) 
      + bold(theta)_2^top overline(bold(x)) (1) quad , 
    theta_1^top overline(bold(x)) (1) 
      + bold(theta)_2^top overline(bold(x)) (2) quad ,
    dots,
  ) $
]

#slide(title: [Convolution])[
  $ vec(bold(theta)_1, bold(theta)_2) * overline(bold(x))(t) = mat(
    bold(theta)_1^top overline(bold(x)) (0) 
      + bold(theta)_2^top overline(bold(x)) (1) quad , 
    theta_1^top overline(bold(x)) (1) 
      + bold(theta)_2^top overline(bold(x)) (2) quad ,
    dots,
  ) $ #pause

  We call this a *convolutional layer* #pause

  *Question:* Anything missing? #pause

  *Answer:* Activation function! #pause

  $ mat(
    sigma(bold(theta)_1^top overline(bold(x)) (0) 
      + bold(theta)_2^top overline(bold(x)) (1)) quad , 
    sigma(theta_1^top overline(bold(x)) (1) 
      + bold(theta)_2^top overline(bold(x)) (2)) quad ,
    dots,
  ) $ #pause

  Much better

  // TODO: Much more param efficient
  // Scale to arbitrary sequences

]


#slide(title: [Convolution])[
  Convolution is *local*, in this example, we only consider two consecutive timesteps #pause

  Convolution is *shift equivariant*, if $bold(theta)_1, bold(theta)_2$ detect "hello", it does not matter whether "hello" occurs at $x(0), x(1)$ or $x(100), x(101)$
]

#slide(title: [Convolution])[
  // TODO: Applications of 1D convolution
]

#slide(title: [Convolution])[
  ```python
  import jax, equinox
  # Assume a sequence of length m
  # Each timestep has dimension d_x
  x = stock_data # Shape (d_x, time)
  conv_layer = equinox.nn.Conv1d(
    in_channels=d_x,
    out_channels=d_y,
    kernel_size=k # Size of filter in timesteps/parameters,
    key=jax.random.key(0)
  )

  z = jax.nn.leaky_relu(conv_layer(x)) 
  ```
]

#slide(title: [Convolution])[
  ```python
  import torch
  # Assume a sequence of length m
  # Each timestep has dimension d_x
  # Torch requires 3 dims! Be careful!
  x = stock_data # Shape (batch, d_x, time)
  conv_layer = torch.nn.Conv1d(
    in_channels=d_x,
    out_channels=d_y,
    kernel_size=k # Size of filter in timesteps/parameters,
  )

  z = jax.nn.leaky_relu(conv_layer(x)) 
  ```
]

#aslide(ag, 2)
#aslide(ag, 3)

#slide(title: [2D Convolution])[
  We defined convolution over one variable $t$ #pause

  For images, we often have two variables denoting width and height $u, v$

  $ x(u, v) $ #pause

  We can also do convolutions over two dimensions #pause

  Most image-based neural networks use convolutions
]

// TODO: Alexnet etc changing ml