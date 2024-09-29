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
        rect((i, j), (i + 1, j + 1), fill: red)
        content((x + i + 0.4, y + j + 0.6), (i, j), str(cells.at(cells.at(i).len() - j - 1).at(i)))

      } else{
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
let colors = (
  red, red, red, red,
  red, red, red, red,
  red, red, red, red,
  red, red, red, red,
)
draw_filter(0, 0, image_values)
//content((2, 2), image("figures/lecture_7/ghost_dog_bw.svg", width: 4cm))
})


// Signals, large continuous time inputs
// Want translation invariance and scalability to long sequences
// Introduce convolution (continuous)


#let agenda(index: none) = {
  let ag = (
    [Review],
    [Signal Processing], 
    [Convolution],
    [2D Convolution],
    [Coding]
  )
  for i in range(ag.len()){
    if index == i {
      enum.item(i + 1)[#text(weight: "bold", ag.at(i))]
    } else {
      enum.item(i + 1)[#ag.at(i)]
    }
  }
}



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

#slide(title: [Agenda])[#agenda(index: none)]
#slide(title: [Agenda])[#agenda(index: 0)]

#slide(title: [Review])[

]

#slide(title: [Agenda])[#agenda(index: 0)]
#slide(title: [Agenda])[#agenda(index: 1)]



#slide(title: [Signal Processing])[
  So far, we have not considered the structure of inputs $X$ #pause

  We treat images as a vector, with no relationship between neighboring pixels #pause

  Neurons $i$ in layer $ell$ has no relationship to neuron $j$ #pause

  However, there is structure inherent in the real world #pause

  By representing this structure within neural networks, we can make neural networks that are more efficient and generalize better #pause

  To do so, we must think of the world as a collection of signals
]



#slide(title: [Signal Processing])[
  A *signal* represents information as a function of time, space or some other variable #pause

  $ x(t) = dots $ 

  $ x(u, v) = dots $ #pause

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

  *Structure:* Tomorrow's stock price will be close to today's stock price

]

#slide(title: [Signal Processing])[
  $ x(t) = "audio" $ #pause

  #align(center, waveform) #pause

  *Structure:* Nearby waves form syllables

]

#slide(title: [Signal Processing])[
  $ x(u, v) = "image" $ #pause
  
  #align(center, implot) #pause

  *Structure:* Repeated components (eyes, nostrils, etc)
]

#slide(title: [Signal Processing])[
  In signal processing, we often consider: #pause
  - Locality #pause
  - Translation invariance
]

#slide(title: [Signal Processing])[
  *Locality:* Information concentrated over small regions of space/time #pause

  #side-by-side[#waveform][#implot]
  
  // TODO Add output signal
]

#slide(title: [Signal Processing])[
  *Translation Invariance:* Signal does not change when shifted in space/time #pause

  #side-by-side[#waveform_left][#waveform_right] #pause

  #align(center)[Both say "hello"]
]

#slide(title: [Signal Processing])[
  *Translation Invariance:* Signal does not change when shifted #pause

  #side-by-side[#implot_left][#implot_right]

  #align(center)[Both contain a dog]
]

#slide(title: [Signal Processing])[
  Perceptrons are not local or translation invariant, each pixel is an independent neuron #pause

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

  How can we get these properties in neural networks?
]

#slide(title: [Agenda])[#agenda(index: 1)]
#slide(title: [Agenda])[#agenda(index: 2)]


#slide(title: [Convolution])[
  In signal processing, we often turn signals into other signals #pause

  #side-by-side[#waveform_left][#hello] #pause

  A standard way to transform signals is *convolution*
]


#slide(title: [Convolution])[
  Convolution is the sum of products of a signal $x(t)$ and a *filter* $g(t)$ #pause

  If time and space is continuous, we write convolution as

  $ x(t) * g(t) = integral_(-oo)^(oo) x(t - tau) g(tau) d tau $ #pause

  If $x, g$ are discrete time or space, we use a sum instead of integral

  $ x(t) * g(t) = sum_(tau=-oo)^(oo) x(t - tau) g(tau) $ #pause

  We slide the filter across the signal, taking the product as we go #pause
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

  Convolution is also *invariant* to time/space shifts
]

#slide(title: [Convolution])[
  Often, we use continuous time/space convolution for analog signals #pause

  For digital signals, we use discrete time/space
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

  *Answer:* In a neuron, each input $x_i$ has a different parameter $theta_i$. In convolution, we reuse $theta_i$ on $x_j, x_k, dots$
]

#slide(title: [Convolution])[
  #side-by-side[Neuron:][$ bold(theta)^top overline(bold(x)) = sum_(i=0)^d_x theta_i overline(x)_i $] #pause

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

  Convolution is *shift invariant*, if $bold(theta)_1, bold(theta)_2$ detect "hello", it does not matter whether "hello" occurs at $x(0), x(1)$ or $x(100), x(101)$
]

#slide(title: [Agenda])[#agenda(index: 2)]
#slide(title: [Agenda])[#agenda(index: 3)]

#slide(title: [2D Convolution])[
]