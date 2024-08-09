#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.1.2": canvas, draw

#set text(size: 25pt)
#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 1: Introduction"
)
#set math.vec(delim: "[")
#set math.mat(delim: "[")

#let side-by-side(columns: none, gutter: 1em, align: center + horizon, ..bodies) = {
  let bodies = bodies.pos()
  let columns = if columns ==  none { (1fr,) * bodies.len() } else { columns }
  if columns.len() != bodies.len() {
    panic("number of columns must match number of content arguments")
  }

  grid(columns: columns, gutter: gutter, align: align, ..bodies)
}


#let cimage(..args) = { 
  align(center + horizon, image(..args))
}



#title-slide(
  title: [Neural Networks],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
  //logo: image("logo.jpg", width: 25%)
)

#slide()[
  + We looked at linear and polynomial $f$
    + Looked at both classification and regression
    + They have problems
      + Input features scale poorly
      + Bad performance around edges
    + Neural networks fix many of these problems
    + What is a neural network?
      + Draw linear model as neural network
    + Based on theory of the brain
        + Invented ages ago
        + Only recently have we learned to harness them
    + Neuron theory
      + Connectivity
      + Activation function
    + Parallels between real/artificial neuron
    + Matrix/graph duality
    + Single layer perceptron
    + Issues with one layer
      + Not universal function approximator
    + Backprops
      + Provides a way to train nn
        + Assigns "fault" for each neuron
      + Recall closed form for linear model
        + We use the gradient of the linear model
      + We use a similar approach
]

#slide[
  + Limitations of linear models
  + History and overview of neural networks
  + Neurons
  + Perceptron
  + Multilayer Perceptron
  + Backpropagation
  + Gradient descent
]

#slide[
  We previously looked at linear and polynomial models for regression #pause

  $ f(bold(x), bold(theta)) = theta_0 + bold(theta) bold(x) = theta_0 + theta_1 x_1 + theta_2 x_2, dots $ #pause

  $ bold(theta) = (bold(X)^top bold(X) )^(-1) bold(X)^top bold(y) $
]

#slide[
  Linear models are useful for simple problems #pause

  Issues with very complex problems #pause
  + Poor scalability #pause
  + Prone to overfitting #pause
  + Polynomials do not generalize well
]

#slide[
  Issues with very complex problems
  + *Poor scalability* 
  + Polynomials do not generalize well
]

#slide[
  Polynomials fit tabular data well #pause

  However, they scale poorly to higher-dimensional data like image pixels #pause

  #side-by-side[#cimage("figures/lecture_1/dog.png", height: 30%)][$ 256 times 256 "pixels" = 65536 "pixels" $]
]

#slide[
  #side-by-side[#cimage("figures/lecture_1/dog.png", height: 30%)][$ 256 times 256 "pixels" = 65536 "pixels" $]
  
  What does the design matrix look like for an n-degree polynomial? #pause

]

#slide[
  $ bold(X) = mat(
    x_1^n, x_1^(n-1), dots, x_1^1, 1;
    x_2^n, x_2^(n-1), dots, x_2^1, 1;
    dots.v, dots.v, dots.down, dots.v, dots.v;
    x_p^n, x_p^(n-1), dots, x_p^1, 1;
    x_1^(n-1) x_2, x^(n-2) x_2^2, dots, 0, 1;
    dots.v, dots.v, , dots.v, dots.v
  ) $ #pause

  *Question:* How big is the matrix for 65,536 pixels and $n=3$?

  *Answer:* $65,536^3 approx 10^14$ parameters #pause
]

#slide[
  *Question:* How big is the matrix for 65,536 pixels and $n=3$?

  *Answer:* $65,536^3 approx 10^14$ parameters #pause

  We must invert $bold(X)^top bold(X)$, requiring $O(n^3)$ time #pause

  Largest matrix ever inverted is $approx 10^12$ #pause

  For comparison, GPT-4 has $10^12$ parameters #pause

  #v(2em)

  #align(center)[Polynomial regression scales poorly to high dimensional data]
]

#slide[
  Issues with very complex problems
  + *Poor scalability* 
  + Polynomials do not generalize well
]

#slide[
  Issues with very complex problems
  + Poor scalability
  + *Polynomials do not generalize well*
]

#slide[
  Polynomials tend towards $-oo, oo$ outside of the support #pause

  $ f(x) = x^3-2x^2-x+2 $ #pause

  #cimage("figures/lecture_3/polynomial_generalize.png", height: 50%) #pause

  If breed of dog missing from training set, we still want to classify it as dog!
]

#slide[
  Linear and polynomial regression have issues #pause
  + Poor scalability #pause
  + Polynomials do not generalize well
]

#focus-slide[Relax]

#slide[
  Can we improve upon the linear/polynomial model? #pause

  Yes, with neural networks #pause

  TODO: diagram/flowchart history of ML
]

#slide[
  *Brain:* Biological neurons $->$ Biological neural network #pause

  *Computer:* Artificial neurons $->$ Artificial neural network #pause  
]

#slide[
  Neurons send and receive electrical impulses along axons and dendrites #pause
  #cimage("figures/lecture_3/neuron_anatomy.jpg")
]

#slide[
  How does a neuron send an impulse ("fire")? #pause

  Incoming impulses (via dendrites) change the electric potential of the neuron #pause
  
  #cimage("figures/lecture_3/bio_neuron_activation.png", height: 50%) #pause

  Pain triggers initial nerve impulse, sets of impulse chain into the brain
]

#slide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") #pause
  
  *Question:* How would you model a neuron mathematically?
]


#slide[
  Let us define a neuron as a function #pause

  #side-by-side[#cimage("figures/lecture_3/neuron_anatomy.jpg")][
    #only((2,3))[
      Neuron has a structure of dendrites

    ]
    #only(3)[
      $ f(vec(theta_1, theta_2, dots.v, theta_n)) = f(vec(1, 0, dots.v, 1)) $

    ]
    #only((4,5))[
      Each incoming dendrite has some voltage potential

    ]

    #only(5)[
      $ f(vec(x_1, dots.v, x_n), vec(theta_(1),  dots.v, theta_(n)) ) $

      $ vec(x_1, dots.v, x_n) = vec(0.5, dots.v, -0.3) $
    ]

    #only((6, 7))[
      Voltage potentials sum together to give us the voltage in the cell body

    ]

    #only(7)[
      $ f(vec(x_1, dots.v, x_n), vec(theta_(1),  dots.v, theta_(n)) ) = sum_(i=1)^n x_i theta_i $
    ]

    #only((8, 9, 10))[
      The axon fires only if the voltage is over a threshold
    ]
    #only((9, 10))[
      
      $ sigma(x) = #image("figures/lecture_3/heaviside.png", height: 30%) $
    ]
    #only(10)[
      $ f(vec(x_1, dots.v, x_n), vec(theta_(1),  dots.v, theta_(n)) ) = sigma(sum_(i=1)^n x_i theta_i) $
    ]
  ]
]

#slide[
  This is almost the artificial neuron!
  $ f(vec(x_1, dots.v, x_n), vec(theta_(1),  dots.v, theta_(n)) ) = sigma(sum_(i=1)^n x_i theta_i) $ #pause

  $ f(bold(x), bold(theta)) = sigma(sum_(i=1)^n x_i theta_i) $

  *Question:* Does it look familiar to any other functions we have seen? #pause

  *Answer:* The linear model!
]
#slide[
  #side-by-side[$ f(bold(x), bold(theta)) = sigma(sum_(i=1)^n x_i theta_i) $][Artificial neuron] #pause

  #side-by-side[$ f(bold(x), bold(theta)) = theta_0 + theta_1 x_1 + theta_2 x_2  + dots + theta_n x_n $][Linear model] #pause

  It is the linear model with an activation function! #pause

  We add a bias term to the neuron, for the same reason we add a bias term to the linear model #pause

  $ f(bold(x), bold(theta)) = sigma(theta_0 + sum_(i=1)^n x_i theta_i) $ #pause
]


#slide[
  #side-by-side(gutter: 4em)[#cimage("figures/lecture_3/neuron_anatomy.jpg")
  $ f(vec(x_1, dots.v, x_n), vec(theta_(1),  dots.v, theta_(n)) ) = sigma(theta_0 + sum_(i=1)^n x_i theta_i) $  
  ][  
  #cimage("figures/lecture_3/neuron.png")]
]

#focus-slide[Relax]

#slide[
  #side-by-side[#cimage("figures/lecture_3/neuron.png") #pause][
    #align(left)[
    
      Recall that in machine learning we deal with functions #pause
     
      What kinds of functions can our neuron represent? #pause

      Let us start with a logical AND function
    ]
   ]
]

#slide[
  #side-by-side[#cimage("figures/lecture_3/neuron.png")][
    #align(left)[
    
      Recall the activation function (Heaviside step) #pause

      #cimage("figures/lecture_3/heaviside.png", height: 50%)

      $
        H(x) = cases(
          1 "if" x > 0,
          0 "if" x <= 0
        )
      $

    ]
   ]
]

#slide[
    Implement AND using an artificial neuron #pause
    
    $ f(x_1, x_2, bold(theta)) = H(theta_0 + x_1 theta_1 + x_2 theta_2) $ #pause
      
    $ bold(theta) = mat(theta_0, theta_1, theta_2)^top = mat(-1, 1, 1)^top $ #pause
    
    #align(center, table(
      columns: 5,
      inset: 0.4em,
      $x_1$, $x_2$, $y$, $f(x_1, x_2, bold(theta))$, $hat(y)$,
      $0$, $0$, $0$, $H(-1 + 1 dot 0 + 1 dot 0) = H(-1)$, $0$,
      $0$, $1$, $0$, $H(-1 + 1 dot 0 + 1 dot 1) = H(0)$, $0$,
      $1$, $0$, $0$, $H(-1 + 1 dot 1 + 1 dot 0) = H(0)$, $0$,
      $1$, $1$, $1$, $H(-1 + 1 dot 1 + 1 dot 1) = H(1)$, $1$
  ))
]

#slide[
    Implement OR using an artificial neuron #pause
    
    $ f(x_1, x_2, bold(theta)) = H(theta_0 + x_1 theta_1 + x_2 theta_2) $ #pause
      
    $ bold(theta) = mat(theta_0, theta_1, theta_2)^top = mat(0, 1, 1)^top $ #pause
    
    #align(center, table(
      columns: 5,
      inset: 0.4em,
      $x_1$, $x_2$, $y$, $f(x_1, x_2, bold(theta))$, $hat(y)$,
      $0$, $0$, $0$, $H(0 + 1 dot 0 + 1 dot 0) = H(0)$, $0$,
      $0$, $1$, $0$, $H(0 + 1 dot 1 + 1 dot 0) = H(1)$, $1$,
      $1$, $0$, $1$, $H(0 + 1 dot 0 + 1 dot 1) = H(1)$, $1$,
      $1$, $1$, $1$, $H(1 + 1 dot 1 + 1 dot 1) = H(2)$, $1$
  ))
]

#slide[
    Implement XOR using an artificial neuron #pause
    
    $ f(x_1, x_2, bold(theta)) = H(theta_0 + x_1 theta_1 + x_2 theta_2) $ #pause
      
    $ bold(theta) = mat(theta_0, theta_1, theta_2)^top = mat(?, ?, ?)^top $ #pause
    
    #align(center, table(
      columns: 5,
      inset: 0.4em,
      $x_1$, $x_2$, $y$, $f(x_1, x_2, bold(theta))$, $hat(y)$,
      $0$, $0$, $0$, [This is IMPOSSIBLE!], $$,
      $0$, $1$, $1$, $$, $$,
      $1$, $0$, $1$, $$, $$,
      $1$, $1$, $0$, $$, $$
  ))
]

#slide[
  Why can't we represent XOR using a neuron? #pause
  
  $ f(x_1, x_2, bold(theta)) = H(theta_0 + x_1 theta_1 + x_2 theta_2) $ #pause

  We can only represent $H("linear function")$ #pause

  XOR is not a linear combination of $x_1, x_2$! #pause

  We want to represent any function, not just linear functions #pause

  Let us think back to biology, maybe it has an answer
]

#slide[
  *Brain:* Biological neurons $->$ Biological neural network #pause

  *Computer:* Artificial neurons $->$ Artificial neural network  
]

#slide[
  Connect artificial neurons into a network

  #grid(
    columns: 2,
    align: center,
    column-gutter: 2em,
    cimage("figures/lecture_3/neuron.png", width: 80%), cimage("figures/lecture_3/deep_network.png", height: 75%),
    [Neuron], [Neural Network] 
  )
]

#slide[
  #side-by-side[
     #cimage("figures/lecture_3/deep_network.png", width: 100%)
  ][
    Adding neurons in *parallel* creates a *wide* neural network #pause
  
    Adding neurons in *series* creates a *deep* neural network #pause

    Today's powerful neural networks are both *wide* and *deep* #pause

    Let us try to implement XOR using a wide and deep neural network
  ]
]

#slide[
    Implement XOR using a deep and wide neural network #pause
    
    $ f(x_1, x_2, bold(theta)) = H( & theta_(3, 0) \
       + & theta_(3, 1) quad dot quad H(theta_(1,0) + x_1 theta_(1,1) + x_2 theta_(1,2)) \ 
      + & theta_(3, 2) quad dot quad H(theta_(2,0) + x_1 theta_(2,1) + x_2 theta_(2,2))) $ #pause
      
    $ bold(theta) = mat(
      theta_(1,0), theta_(1,1), theta_(1,2);
      theta_(2,0), theta_(2,1), theta_(2,2);
      theta_(3,0), theta_(3,1), theta_(3,2)
    ) = mat(
      -0.5, 1, 1; 
      -1.5, 1, 1;
      -0.5, 1, -2
    ) $ #pause
]

#slide[
  What other functions can we represent using a deep and wide neural network? #pause

  Consider a one-dimensional arbitrary function $g(x) = y$ #pause

  We can approximate $g$ using our neural network $f$ #pause

  $ f(x_1, x_2, bold(theta)) = H( & theta_(3, 0) \
       + & theta_(3, 1) quad dot quad H(theta_(1,0) + x_1 theta_(1,1) + x_2 theta_(1,2)) \ 
      + & theta_(3, 2) quad dot quad H(theta_(2,0) + x_1 theta_(2,1) + x_2 theta_(2,2))) $
]

#slide[
  *Proof Sketch:* Approximate a function $g(x)$ using a linear combination of Heaviside functions

  #only(2)[#cimage("figures/lecture_3/function_noapproximation.svg", height: 50%)]

  #only((3, 4))[#cimage("figures/lecture_3/function_approximation.svg", height: 50%)]

  #only(4)[$ "Roughly, " [lim_(n |-> oo) theta_(2, 0) + theta_(2, 1) sum_(j = 1)^n sigma(theta_(1, 0) + theta_(1, j) x) ] = g(x); quad forall g $]
]

#slide[
  More formally, a wide and deep neural network is a *universal function approximator* #pause

  It can approximate *any* continuous function to precision $epsilon$ #pause

  $ | g(bold(x)) - f(bold(x), bold(theta)) | < epsilon $ #pause
  
  As we increase the width and depth of the network, $epsilon$ shrinks #pause


  #side-by-side[$ g(#image("figures/lecture_1/dog.png", height: 20%)) = "Dog"$][$ g(#image("figures/lecture_1/muffin.png", height: 20%)) = "Muffin" $] 
  
  #align(center)[Very powerful finding! The basis of deep learning.]
]

#focus-slide[Relax]

#slide[
  We call this form of a neural network a *feedforward network* or *perceptron* (invented in 1943) #pause

  #cimage("figures/lecture_3/mark_1_perceptron.jpeg", height: 60%)

  $20 times 20$ grid of pixels to process images
]

#slide[

  #cimage("figures/lecture_3/timeline.svg", width: 85%) #pause

  If the deep neural network was invented in 1958, why did it take 70 years for us to care about deep learning? #pause


  Many small improvements over time eventually made NNs feasible
  
]

#slide[
  The neural network we created today is called a feedforward network or perceptron #pause

  When the network is deep, we call it a Multi-Layer Perceptron (MLP) #pause

  We often use the term "layers", when referring to a specific depth of the neural network
    - Four-layer MLP means a neural network with a depth of four
]


/*
#slide[
  #side-by-side[#cimage("figures/neuron.png", width: 80%)][#cimage("figures/heaviside.png", height: 50%)] #pause
  *Question:* What kind of functions can we represent with our neuron? #pause

  *Hint:* The neuron is linear regression with an activation function
]

#slide[
  #side-by-side[#cimage("figures/neuron.png", width: 80%)][#cimage("figures/heaviside.png", height: 50%)] #pause
  *Answer:* Linear functions with cutoff
]


#slide[
  #side-by-side[#cimage("figures/neuron.png") #pause][
    The output of the neuron depends on the activation function $sigma$
  ]
]



#slide[
  #side-by-side[#cimage("figures/neuron.png") #pause][
    *Question:* What functions can a single neuron represent?
    *Hint:* Think back to linear regression #pause

    *Answer:*
  ]
]

#slide[
  #side-by-side[#cimage("figures/neuron.png") #pause][
    Many biological neurons (brain) $->$ many artificial neurons (deep neural network)
  ]
  
]
*/
