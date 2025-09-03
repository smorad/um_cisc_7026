#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.0"
#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "common.typ": *
#import "@preview/pinit:0.2.2": *

// TODO Swap order of ones in design


// TODO: Missing x^2 term when we show polynomial+multivariate example (not 2^3, should be 3^2 + 1)

// For students: you may want to change this to true
// otherwise you will get one slide for each new line
#let handout = false

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(handout: handout),
  config-info(
    title: [Neural Networks],
    subtitle: [CISC 7026 - Introduction to Deep Learning],
    author: [Steven Morad],
    institution: [University of Macau],
    logo: image("figures/common/bolt-logo.png", width: 4cm)
  ),
  header-right: none,
  header: self => utils.display-current-heading(level: 1)
)
#set math.vec(delim: "[")
#set math.mat(delim: "[")

// TODO: Deeper neural networks are more efficient
// FUTURE TODO: Should not waste m/n in linear regression, use c for count and d_x, d_y
// TODO: Fix nn image indices
// TODO: Implement XOR is transposed
// TODO: is xor network actually wide?
// TODO: Handle subscripts for input dim rather than sample
// TODO: Emphasize importance of very deep/wide nn

#let argmin_plot = canvas(length: 1cm, {
  plot.plot(size: (8, 4),
    x-tick-step: 1,
    y-tick-step: 2,
    {
      plot.add(
        domain: (-2, 2), 
        x => calc.pow(1 + x, 2),
        label: $ (x + 1)^2 $
      )
    })
})


#title-slide()

== Outline <touying:hidden>

#components.adaptive-columns(
    outline(title: none, indent: 1em, depth: 1)
)

/*
    [Review],
    [Multivariate linear regression],
    [Limitations of linear regression],
    [History of neural networks],
    [Biological neurons],
    [Artificial neurons],
    [Wide neural networks],
    [Deep neural networks],
    [Practical considerations]
*/

= Agenda
==

= Review
==

  Since you are very educated, we focused on how education affects life expectancy #pause

  Studies show a causal effect of education on health #pause
    - _The causal effects of education on health outcomes in the UK Biobank._ Davies et al. _Nature Human Behaviour_. #pause
    - By staying in school, you are likely to live longer #pause
    - Being rich also helps, but education alone has a *causal* relationship with life expectancy

==
  *Goal:* Given your education, predict your life expectancy #pause

  $X in bb(R)_+:$ Years in school #pause
  
  $Y in bb(R)_+:$ Age of death #pause

  $Theta in bb(R)^2:$ Parameters #pause 

  $ f: X times Theta |-> Y $ #pause


  *Approach:* Learn the parameters $theta$ such that 

  $ f(x, theta) = y; quad x in X, y in Y $

==
  Started with a linear function $f$ #pause

  #align(center, grid(
    columns: 2,
    align: center,
    column-gutter: 2em,
    $ f(x, bold(theta)) = f(x, vec(theta_0, theta_1)) = theta_0 + theta_1 x $,
    cimage("figures/lecture_2/example_regression_graph.png", height: 50%)
  )) #pause
  
  Then, we derived the square error function #pause

  $ "error"(f(x, bold(theta)), y) = (f(x, bold(theta)) - y)^2 $

==
  We wrote the loss function for a single datapoint $x_[i], y_[i]$ using the square error

  $ cal(L)(x_[i], y_[i], bold(theta)) = "error"(f(x_[i], bold(theta)),  y_[i]) = (f(x_[i], bold(theta)) - y_[i])^2 $ #pause

  But we wanted to learn a model over *all* the data, not a single datapoint #pause

  We wanted to make *new* predictions, to *generalize* #pause

  $ bold(x) = mat(x_[1], x_[2], dots, x_[n])^top, bold(y) = mat(y_[1], y_[2], dots, y_[n])^top $ #pause

  $ 
  cal(L)(bold(x), bold(y), bold(theta)) = sum_(i=1)^n "error"(f(x_[i], bold(theta)),  y_[i]) = sum_(i=1)^n (f(x_[i], bold(theta)) - y_[i])^2 
  $

==
  Our objective was to find the parameters that minimized the loss function over the dataset #pause

  We introduced the $argmin$ operator #pause

  #side-by-side[ $f(x) = (x + 1)^2$][#argmin_plot] #pause

  $ argmin_x f(x) = -1 $

==
  With the $argmin$ operator, we formally wrote our optimization objective #pause

  $ 
   #text(fill: color.red)[$argmin_bold(theta)$] cal(L)(bold(x), bold(y), bold(theta)) &= #text(fill: color.red)[$argmin_bold(theta)$] sum_(i=1)^n "error"(f(x_[i], bold(theta)),  y_[i]) \ &= #text(fill: color.red)[$argmin_bold(theta)$] sum_(i=1)^n (f(x_[i], bold(theta)) - y_[i])^2 
  $ 

==
  We defined the design matrix $overline(bold(X))$ #pause

  $ overline(bold(X)) = mat(bold(1), bold(x)) = mat(1, x_[1]; 1, x_[2]; dots.v, dots.v; 1, x_[n]) $ #pause

  We use the design matrix to find an *analytical* solution to the optimization objective #pause
  
$ bold(theta) = (overline(bold(X))^top overline(bold(X)))^(-1) overline(bold(X))^top bold(y) $ 

==
  With this analytical solution, we were able to learn a linear model #pause

  #cimage("figures/lecture_2/linear_regression.png", height: 60%)

==
  Then, we used a trick to extend linear regression to nonlinear models #pause

  $ overline(bold(X)) = mat(1, x_[1]; 1, x_[2]; dots.v, dots.v; 1, x_[n]) => overline(bold(X)) = mat(1, log(1 + x_[1]); 1, log(1 + x_[2]); dots.v, dots.v; 1, log(1 + x_[n])) $

==
  We extended to polynomials, which are *universal function approximators* #pause

  $ overline(bold(X)) = mat(1, x_[1]; 1, x_[2]; dots.v, dots.v; 1, x_[n]) => overline(bold(X)) = mat(
    1, x_[1], dots, x^m_[1]; 
    1, x_[2], dots, x^m_[2]; 
    dots.v, dots.v, dots.down; 
    1, x_[3], dots, x^m_[3]; 
    ) $ #pause

  $ f: X times Theta |-> bb(R) $ #pause

  $ Theta in bb(R)^2 => Theta in bb(R)^(m+1) $ 

==
  Finally, we discussed overfitting #pause

  $ f(x, bold(theta)) = theta_0 + theta_1 x + dots + theta_m x^m $ #pause

  #grid(
    columns: 3,
    row-gutter: 1em,
    image("figures/lecture_2/polynomial_regression_n2.png"),
    image("figures/lecture_2/polynomial_regression_n3.png"),
    image("figures/lecture_2/polynomial_regression_n5.png"),
    $ m = 2 $,
    $ m = 3 $,
    $ m = 5 $
  )

==
  We care about *generalization* in machine learning #pause

  So we should always split our dataset into a training dataset and a testing dataset #pause

  #cimage("figures/lecture_2/train_test_regression.png", height: 60%)

// 16:00 fast
= Multivariate Linear Regression

==

  Last time, we assumed a single-input system #pause

  Years of education: $X in bb(R)$ #pause

  But sometimes we want to consider multiple input dimensions #pause

  Years of education, BMI, GDP: $X in bb(R)^3$ #pause

  We can solve these problems using linear regression too

==
  For multivariate problems, we will define the input dimension as $d_x$ #pause

  $ bold(x) in X; quad X in bb(R)^(d_x) $ #pause

  We will write the vectors as

  $ bold(x)_[i] = vec(
    x_([i], 1),
    x_([i], 2),
    dots.v,
    x_([i], d_x)
  ) $ #pause

  $x_([i], 1)$ refers to the first dimension of training data $i$

==
  The design matrix for a *multivariate* linear system is

  $ overline(bold(X)) = mat(
    1, x_([1], 1), x_([1], 2),  dots, x_([1], d_x); 
    1, x_([2], 1), x_([2], 2),  dots, x_([2], d_x); 
    dots.v, dots.v, dots.v, dots.down, dots.v ; 
    1, x_([n], 1), x_([n], 2),  dots, x_([n], d_x); 
  ) $ #pause

  Remember $x_([n], d_x)$ refers to dimension $d_x$ of training data $n$ #pause

  The solution is the same as before

  $ bold(theta) = (overline(bold(X))^top overline(bold(X)) )^(-1) overline(bold(X))^top bold(y) $ 

// 22:00 fast


= Limitations of Linear Regression
==
  Linear models are useful for certain problems #pause
  + Analytical solution #pause
  + Low data requirement #pause

  Issues arise with other problems #pause
  + Poor scalability #pause
  + Polynomials do not generalize well

==
  Issues arise with other problems
  + *Poor scalability* 
  + Polynomials do not generalize well

==
  So far, we have seen: #pause

  #side-by-side[
    One-dimensional polynomial functions
    $ overline(bold(X)) = mat(
    1, x_[1], dots, x^m_[1]; 
    1, x_[2], dots, x^m_[2]; 
    dots.v, dots.v, dots.down; 
    1, x_[3], dots, x^m_[3]; 
    ) $ #pause][
    Multi-dimensional linear functions
    $ overline(bold(X)) = mat(
    1, x_([1], 1), x_([1], 2),  dots, x_([1], d_x); 
    1, x_([2], 1), x_([2], 2),  dots, x_([2], d_x); 
    dots.v, dots.v, dots.v, dots.down, dots.v, ; 
    1, x_([n], 1), x_([n], 2),  dots, x_([n], d_x); 
  ) $ #pause
  ]

  Combine them to create multi-dimensional polynomial functions #pause

==
  Let us do an example #pause

  #side-by-side[*Task:* predict how many #text(fill: color.red)[#sym.suit.heart] a photo gets on social media][#cimage("figures/lecture_1/dog.png", height: 30%)] #pause

  $ f: X times Theta |-> Y; quad X: "Image", quad Y: "Number of " #redm[$#sym.suit.heart$]  $ #pause

  $ X in bb(Z)_+^(256 times 256) = bb(Z)_+^(65536); quad Y in bb(Z)_+ $  #pause

  Highly nonlinear task, use a polynomial with order $m=20$ 

==
// TODO FIX THIS 2025
  $ overline(bold(X)) = mat(bold(x)_([1], d_x), dots, bold(x)_([n], d_x))^top $ #pause

  $ &bold(x)_([i], D) = \ &mat(
    underbrace(x_([i], d_x)^m x_([i], d_x - 1)^m dots x_([i], 1)^m, (d_x => 1, x^m)),
    underbrace(x_([i], d_x)^m x_([i], d_x - 1)^m dots x_([i], 2)^m, (d_x => 2, x^m)),
    dots,
    underbrace(x_([i], d_x)^(m-1) x_([i], d_x - 1)^(m-1) dots x_([i], 1)^m, (d_x => 1, x^(m-1))),
    dots,
  )
  $


  *Question:* How many columns in this matrix? #pause

  *Hint:* $d_x = 2, m = 3$: $x^3 + y^3 + x^2 y + y^2 x + x y + x + y + 1$ #pause

  *Answer:* $(d_x)^m = 65536^20 + 1 approx 10^96$

==
  How big is $10^96$? #pause

  *Question:* How many atoms are there in the universe? #pause

  *Answer:* $10^82$ #pause

  There is not enough matter in the universe to represent one row #pause

  #side-by-side[We cannot predict how many #text(fill: color.red)[#sym.suit.heart] the picture will get][#cimage("figures/lecture_1/dog.png", height: 30%)] #pause

==
  Polynomial regression does not scale to large inputs
  + *Poor scalability* 
  + Polynomials do not generalize well

==
  Issues arise with other problems
  + Poor scalability
  + *Polynomials do not generalize well*

==
  What happens to polynomials outside of the support (dataset)? #pause

  Take the limit of polynomials to see their behavior #pause

  #side-by-side[$ lim_(x -> oo) theta_0 + theta_1 x + dots + theta_m x^m $][Equation of a polynomial] #pause

  #side-by-side[$ lim_(x -> oo) x^m (theta_0 / x^m + theta_1 / x^(m-1) + dots + theta_m) $][Factor out $x^m$] #pause

  #side-by-side[$ lim_(x -> oo) x^m dot lim_(x-> oo) (theta_0 / x^m + theta_1 / x^(m-1) + dots + theta_m) $][Split the limit (limit of products)] 

==
  #side-by-side[$ lim_(x -> oo) x^m dot lim_(x-> oo) (theta_0 / x^m + theta_1 / x^(m-1) + dots + theta_m) $][Split the limit (limit of products)] #pause
  
  #side-by-side[$ (lim_(x -> oo) x^m) dot (0 + 0 + dots + theta_m ) $][Evaluate right limit] #pause

  #side-by-side[$ theta_m lim_(x -> oo) x^m  $][Rewrite] #pause

  #side-by-side[$ theta_m lim_(x -> oo) x^m = oo $][If $theta_m > 0$] #pause

  #side-by-side[$ theta_m lim_(x -> oo) x^m = -oo $][If $theta_m < 0$]

==
  Polynomials quickly tend towards $-oo, oo$ outside of the support #pause

  $ f(x) = 2 -x - 2x^2 + x^3 $ #pause

  #cimage("figures/lecture_3/polynomial_generalize.png", height: 50%) #pause

  Remember, to predict new data we want our functions to generalize

==
  Linear regression has issues #pause
  + Poor scalability #pause
  + Polynomials do not generalize well

// 38:00 fast

==
  We can use neural networks as an alternative to linear regression #pause

  Neural network benefits: #pause
    + Scale to large inputs #pause
    + Slightly better generalization #pause

  Drawbacks: #pause
    + No analytical solution #pause
    + High data requirement 



  //#cimage("figures/lecture_1/timeline.svg", height: 50%) 
// 40:00 fast
= History of Neural Networks
==
  In 1939-1945, there was a World War #pause

  Militaries invested funding for research, and invented the computer #pause

  #cimage("figures/lecture_3/turing.jpg", height: 70%)

==
  #side-by-side[Meanwhile, a neuroscientist and mathematician (McCullough and Pitts) were trying to understand the human brain][#cimage("figures/lecture_3/mccullough-pitts.png", height: 70%)] #pause

  They designed the theory for the first neural network

==
  Rosenblatt implemented this neural network theory on a computer a few years later #pause
  #side-by-side[
    At the time, computers were very slow and expensive
][#cimage("figures/lecture_3/original_nn.jpg", height: 70%)] 

==
  Through advances in theory and hardware, neural networks became slightly better #pause

  #cimage("figures/lecture_1/timeline.svg", height: 40%) #pause

  Around 2012, these improvements culminated in neural networks that perform like humans

==
  So what is a neural network? #pause

  It is a function, inspired by how the brain works #pause

  $ f: X times Theta |-> Y $ 

==
  Brains and neural networks rely on *neurons* #pause

  *Brain:* Biological neurons $->$ Biological neural network #pause

  *Computer:* Artificial neurons $->$ Artificial neural network #pause

  First, let us review biological neurons #pause

  *Note:* I am not a neuroscientist! I may make simplifications or errors with biology

= Biological Neurons
==
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  A simplified neuron consists of many parts 

// 47:00 fast
==
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  Neurons send messages based on messages received from other neurons

==
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  Incoming electrical signals travel along dendrites

==
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  Electrical charges collect in the Soma (cell body)

==
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  The axon outputs an electrical signal to other neurons

==
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  The axon terminals will connect to dendrites of other neurons through a synapse

==
  #cimage("figures/lecture_3/synapse.png", height: 60%)
  The synapse converts electrical signal, to chemical signal, back to electrical signal #pause

  Synaptic weight determines how well a signal crosses the gap

==
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  For our purposes, we can model the axon terminals, dendrites, and synapses to be one thing

==
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  The neuron takes many inputs, and produces a single output

==
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  The neuron will only output a signal down the axon ("fire") at certain times

==
  How does a neuron decide to send an impulse ("fire")? #pause

  #side-by-side[Incoming impulses (via dendrites) change the electric potential of the neuron][  #cimage("figures/lecture_3/bio_neuron_activation.png", height: 50%)] #pause

  In a parallel circuit, we can sum voltages together #pause

  Many active dendrites will add together and trigger an impulse
  
==
  #side-by-side[Pain triggers initial nerve impulse, starts a chain reaction into the brain][#cimage("figures/lecture_3/nervous-system.jpg")]

==
  #side-by-side[When the signal reaches the brain, we will think][#cimage("figures/lecture_3/nervous-system.jpg")]

==
  #side-by-side[After thinking, we will take action][#cimage("figures/lecture_3/nervous-system.jpg")]

// 57:00

= Artificial Neurons
==
  #cimage("figures/lecture_3/neuron_anatomy.jpg", height: 50%) #pause

  *Question:* How could we write a neuron as a function? $quad f: "___" |-> "___"$ #pause

  *Answer*:

  $ f: underbrace(bb(R)^(d_x), "Dendrite voltages") times underbrace(bb(R)^(d_x), "Synaptic weight") |-> underbrace(bb(R), "Axon voltage") $

==
  Let us implement an artifical neuron as a function #pause

  #side-by-side[#cimage("figures/lecture_3/neuron_anatomy.jpg")][
    #only((2,3))[
      Neuron has a structure of dendrites with synaptic weights

    ]
    #only(3)[
      $ f(
        #redm[$vec(theta_1, theta_2, dots.v, theta_(d_x))$])
      $

      $ f(#redm[$bold(theta)$]) $

    ]
    #only((4,5))[
      Each incoming dendrite has some voltage potential
    ]

    #only(5)[
      $ f(#redm[$vec(x_(1), dots.v, x_(d_x))$], vec(theta_(1),  dots.v, theta_(d_x)) ) $ 
      
      $ f(#redm[$bold(x)$], bold(theta)) $
    ]

    #only((6, 7))[
      Voltage potentials sum together to give us the voltage in the cell body

    ]

    #only(7)[
      $ f(vec(x_(1), dots.v, x_(d_x)), vec(theta_(1),  dots.v, theta_(d_x)) ) = #redm[$sum_(j=1)^(d_x) theta_j x_(j)$] $

      $ f(bold(x), bold(theta)) = #redm[$bold(theta)^top bold(x)$] $
    ]

    #only((8, 9, 10))[
      The axon fires only if the voltage is over a threshold
    ]
    #only((9, 10))[
      
      $ sigma(x)= H(x) = #image("figures/lecture_3/heaviside.png", height: 30%) $
    ]
    #only(10)[
      $ f(vec(x_(1), dots.v, x_(n)), vec(theta_(1),  dots.v, theta_(n)) ) = #redm[$sigma$] (sum_(j=1)^(d_x) theta_j x_(j) ) $
    ]
  ]
// 1:05

==
  #side-by-side[Maybe we want to vary the activation threshold][#cimage("figures/lecture_3/bio_neuron_activation.png", height: 30%)][#image("figures/lecture_3/heaviside.png", height: 30%)] #pause

  $ f(vec(#redm[$1$], x_(1), dots.v, x_(d_x)), vec(#redm[$theta_0$], theta_(1),  dots.v, theta_(d_x)) ) = sigma(#redm[$theta_0$] + sum_(j=1)^(d_x) theta_j x_j) = sigma(sum_(#redm[$j=0$])^(d_x) theta_j x_j) $ #pause

  $ overline(bold(x)) = vec(1, bold(x)), quad f(bold(x), bold(theta)) = sigma(bold(theta)^top overline(bold(x))) $

==
  $ f(bold(x), bold(theta)) = sigma(bold(theta)^top overline(bold(x))) $ #pause

  This is the artificial neuron! #pause

  Let us write out the full equation for a neuron #pause
  
  $ f(bold(x), bold(theta)) = sigma( theta_0 1 + theta_1 x_1 + dots + theta_(d_x) x_(d_x) ) $ #pause

  *Question:* Does this look familiar to anyone? #pause

  *Answer:* Inside $sigma$ is the multivariate linear model!

  $ f(bold(x), bold(theta)) = theta_0 + theta_1 x_1 + dots + theta_(d_x) x_(d_x)  $

==
  We model a neuron using a linear model and activation function #pause
  #side-by-side(gutter: 4em)[#cimage("figures/lecture_3/neuron_anatomy.jpg", height: 40%)
  ][  
  #cimage("figures/lecture_3/neuron.svg", height: 40%)]

  $ f(bold(x), bold(theta)) = sigma(bold(theta)^top overline(bold(x))) $ 

==
  $ f(bold(x), bold(theta)) = sigma(bold(theta)^top overline(bold(x))) $ #pause

  Sometimes, we will write $bold(theta)$ as a bias and weight $b, bold(w)$ #pause

  $ bold(theta) = vec(b, bold(w)); quad vec(theta_0, theta_1, dots.v, theta_(d_x)) = vec(b_" ", w_1, dots.v, w_(d_x)) $ #pause

  $ f(bold(x), vec(b, bold(w))) = b + bold(w)^top bold(x) $ 

// 1:15

==
  #side-by-side[#cimage("figures/lecture_3/neuron.svg") #pause][
    #align(left)[
    
      In machine learning, we represent functions #pause
     
      What kinds of functions can our neuron represent? #pause

      Let us consider some *boolean* functions #pause

      Let us start with a logical AND function
    ]
   ]

==
  #side-by-side[#cimage("figures/lecture_3/neuron.png")][
    #align(left)[
    
      *Review:* Activation function (Heaviside step function) #pause

      #cimage("figures/lecture_3/heaviside.png", height: 50%)

      $
        sigma(x) = H(x) = cases(
          1 "if" x > 0,
          0 "if" x <= 0
        )
      $

    ]
   ]

==
    Implement AND using an artificial neuron #pause
    
    $ f(mat(x_1, x_2)^top, mat(theta_0, theta_1, theta_2)^top) = sigma(theta_0 1 + theta_1 x_1 + theta_2 x_2) $ #pause
      
    $ bold(theta) = mat(theta_0, theta_1, theta_2)^top = mat(-1, 1, 1)^top $ #pause
    
    #align(center, table(
      columns: 5,
      inset: 0.4em,
      $x_1$, $x_2$, $y$, $f(x_1, x_2, bold(theta))$, $hat(y)$,
      $0$, $0$, $0$, $sigma(-1 dot 1 + 1 dot 0 + 1 dot 0) = sigma(-1)$, $0$,
      $0$, $1$, $0$, $sigma(-1 dot 1 + 1 dot 0 + 1 dot 1) = sigma(0)$, $0$,
      $1$, $0$, $0$, $sigma(-1 dot 1 + 1 dot 1 + 1 dot 0) = sigma(0)$, $0$,
      $1$, $1$, $1$, $sigma(-1 dot 1 + 1 dot 1 + 1 dot 1) = sigma(1)$, $1$
  ))

==
    Implement OR using an artificial neuron #pause
    
    $ f(mat(x_1, x_2)^top, mat(theta_0, theta_1, theta_2)^top) = sigma(theta_0 1 + theta_1 x_1 + theta_2 x_2) $ #pause
      
    $ bold(theta) = mat(theta_0, theta_1, theta_2)^top = mat(0, 1, 1)^top $ #pause
    
    #align(center, table(
      columns: 5,
      inset: 0.4em,
      $x_1$, $x_2$, $y$, $f(x_1, x_2, bold(theta))$, $hat(y)$,
      $0$, $0$, $0$, $sigma(1 dot 0 + 1 dot 0 + 1 dot 0) = sigma(0)$, $0$,
      $0$, $1$, $0$, $sigma(1 dot 0 + 1 dot 1 + 1 dot 0) = sigma(1)$, $1$,
      $1$, $0$, $1$, $sigma(1 dot 0 + 1 dot 0 + 1 dot 1) = sigma(1)$, $1$,
      $1$, $1$, $1$, $sigma(1 dot 0 + 1 dot 1 + 1 dot 1) = sigma(2)$, $1$
  ))


// Approx 1:30

==
    Implement XOR using an artificial neuron #pause
    
    $ f(mat(x_1, x_2)^top, mat(theta_0, theta_1, theta_2)^top) = sigma(theta_0 1 + theta_1 x_2 + theta_2 x_2) $ #pause
      
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

==
  Why can't we represent XOR using a neuron? #pause
  
  $ f(mat(x_1, x_2)^top, mat(theta_0, theta_1, theta_2)^top) = sigma(1 theta_0 + x_1 theta_1 + x_2 theta_2) $ #pause

  We can only represent $sigma("linear function")$ #pause

  XOR is not a linear combination of $x_1, x_2$! #pause

  We want to represent any function, not just linear functions #pause

  Let us think back to biology, maybe it has an answer

==
  *Brain:* Biological neurons $->$ Biological neural network #pause

  *Computer:* Artificial neurons $->$ Artificial neural network

==
  Connect artificial neurons into a network

  #grid(
    columns: 2,
    align: center,
    column-gutter: 2em,
    cimage("figures/lecture_3/neuron.svg", width: 80%), cimage("figures/lecture_3/deep_network.png", height: 75%),
    [Neuron], [Neural Network] 
  )

==
  #side-by-side[
     #cimage("figures/lecture_3/deep_network.png", width: 100%)
  ][
    Adding neurons in *parallel* creates a *wide* neural network #pause
  
    Adding neurons in *series* creates a *deep* neural network #pause

    Today's powerful neural networks are both *wide* and *deep*
  ]

= Wide Neural Networks
==
  How do we express a *wide* neural network mathematically? #pause

  A single neuron:

  $ f: bb(R)^(d_x) times Theta |-> bb(R) $ 
  
  $ Theta in bb(R)^(d_x + 1) $ #pause

  $d_y$ neurons (wide):

  $ f: bb(R)^(d_x) times Theta |-> bb(R)^(d_y) $ 
  
  $ Theta in bb(R)^((d_x + 1) times d_y) $

==
  For a single neuron:
  
  $ f(vec(x_1, dots.v, x_(d_x)), vec(theta_0,  theta_1, dots.v, theta_(d_x)) ) = sigma(sum_(i=0)^(d_x) theta_i overline(x)_i) $ #pause

  $ f(bold(x), bold(theta)) = sigma(b + bold(w)^top bold(x)) $

==
  // Must be m by n (m rows, n cols)
  #text(size: 24pt)[
  For a wide network:
  $ f(vec(x_1, x_2, dots.v, x_(d_x)), mat(theta_(0,1), theta_(0,2), dots, theta_(0,d_y); theta_(1,1), theta_(1,2), dots, theta_(1, d_y); dots.v, dots.v, dots.down, dots.v; theta_(d_x, 1), theta_(d_x, 2), dots, theta_(d_x, d_y)) ) = vec(
    sigma(sum_(i=0)^(d_x) theta_(i,1) overline(x)_i ),
    sigma(sum_(i=0)^(d_x) theta_(i,2) overline(x)_i ),
    dots.v,
    sigma(sum_(i=0)^(d_x) theta_(i,d_y) overline(x)_i ),
  )
  $  


  $ f(bold(x), bold(theta)) = 
    sigma(bold(theta)^top overline(bold(x))); quad bold(theta)^top in bb(R)^( d_y times (d_x + 1) ) 
  $ #pause
  $
    f(bold(x), vec(bold(b), bold(W))) = sigma( bold(b) + bold(W)^top bold(x) ); quad bold(b) in bb(R)^(d_y), bold(W) in bb(R)^( d_x times d_y ) 
  $
  ]

= Deep Neural Networks
==
  How do we express a *deep* neural network mathematically? #pause

  A wide network and deep network have a similar function signature:

  $ f: bb(R)^(d_x) times Theta |-> bb(R)^(d_y) $ #pause

  But the parameters change!

  Wide: $Theta in bb(R)^((d_x + 1) times d_y)$ #pause 

  Deep: $Theta in bb(R)^((d_x + 1) times d_h) times bb(R)^((d_h + 1) times d_h) times dots times bb(R)^((d_h + 1) times d_y)$ #pause

  $ bold(theta) = mat(bold(theta)_1, bold(theta)_2, dots, bold(theta)_ell)^top = mat(bold(phi), bold(psi), dots, bold(xi))^top $ 

==
  A wide network:

  $ f(bold(x), bold(theta)) = sigma(bold(theta)^top overline(bold(x))) $ #pause

  A deep network has many internal functions

    $ f_1(bold(x), bold(phi)) = sigma(bold(phi)^top overline(bold(x))) quad

    f_2(bold(x), bold(psi)) = sigma(bold(psi)^top overline(bold(x))) quad

    dots quad

    f_(ell)(bold(x), bold(xi)) = sigma(bold(xi)^top overline(bold(x))) $ #pause


  $ f(bold(x), bold(theta)) = f_(ell) (dots f_2(f_1(bold(x), bold(phi)), bold(psi)) dots bold(xi) ) $

==
  Written another way

  $ bold(z)_1 = f_1(bold(x), bold(phi)) = sigma(bold(phi)^top overline(bold(x))) $ #pause
  $ bold(z)_2 = f_2(bold(z_1), bold(psi)) = sigma(bold(psi)^top overline(bold(z))_1) $ #pause
  $ dots.v $ #pause
  $ bold(hat(y)) = f_(ell)(bold(x), bold(xi)) = sigma(bold(xi)^top overline(bold(z))_(ell - 1)) $

  We call each function a *layer* #pause

  A deep neural network is made of many layers

==
  What functions can we represent using a deep neural network? #pause

  *Proof Sketch:* Approximate a continuous function $g: bb(R) |-> bb(R)$ using a linear combination of Heaviside functions #pause

  #only(2)[#cimage("figures/lecture_3/function_noapproximation.svg", height: 50%)]

  #only((3,4))[#cimage("figures/lecture_3/function_approximation.svg", height: 50%)]

  #only(4)[$exists (bold(theta) in bb(R)^(1 times d_h), bold(phi) in bb(R)^((d_h + 1) times d_1)) "such that" lim_(d_h  |-> oo) [ bold(phi)^top sigma(overline(bold(theta)^top overline(x)))]$
  ] 

==
  A deep neural network is a *universal function approximator* #pause

  It can approximate *any* continuous function $g(x)$ to precision $epsilon$ #pause

  $ | g(bold(x)) - f(bold(x), bold(theta)) | < epsilon $ #pause

  Making the network deeper or wider decreases $epsilon$ #pause

  #align(center)[#underline[Very powerful finding! The basis of deep learning.]] #pause

  #side-by-side[*Task:* predict how many #text(fill: color.red)[#sym.suit.heart] a photo gets on social media][#cimage("figures/lecture_1/dog.png", height: 30%)] 


= Practical Considerations
==
  We call wide neural networks *perceptrons* #pause

  We call deep neural networks *multi-layer perceptrons* (MLP) #pause

  #cimage("figures/lecture_3/timeline.svg", width: 85%)

==
  *All* the models we examine in this course will use MLPs #pause
    - Recurrent neural networks #pause
    - Graph neural networks #pause
    - Transformers #pause
    - Chatbots #pause

  It is very important to understand MLPs! #pause

  I will explain them again very simply 

==
  A *layer* is a linear operation and an activation function

  #side-by-side[
    $ f(bold(x), bold(theta)) = sigma(bold(theta)^top overline(bold(x))) $ #pause
  ][
    $ f(bold(x), vec(bold(b), bold(W))) = sigma(bold(b) + bold(W)^top bold(x)) $ #pause
  ]

  #side-by-side[Many layers makes a deep neural network][
  #text(size: 22pt)[
    $ bold(z)_1 &= f(bold(x), bold(theta)) \
    bold(z)_2 &= f(bold(z)_1, bold(theta)) \ quad bold(hat(y)) &= f(bold(z)_2, bold(theta)) $
  ]
  ]

==
  Let us create a wide neural network in colab! https://colab.research.google.com/drive/1bLtf3QY-yROIif_EoQSU1WS7svd0q8j7?usp=sharing

==
  #side-by-side(align: left + top)[
    Linear regression: #pause

    $+$ Analytical solution #pause

    $+$ Low data requirement #pause

    $-$ Poor scalability #pause

    $-$ Poor polynomials generalization #pause
  ][
  Neural networks: #pause

    $-$ No analytical solution #pause

    $-$ High data requirement #pause

    $+$ Scale to large inputs #pause

    $+$ Slightly better generalization #pause
  ]

  Next time, we will find out how to train our neural network #pause

  Unlike linear regression, finding $theta$ is much more difficult for neural networks



/*
#slide[

  #text(size: 21pt)[
    ```python
    import torch
    from torch import nn

    class MyNetwork(nn.Module):
      def __init__(self):
        super().__init__() # Required by pytorch
        self.input_layer = nn.Linear(5, 3) # 3 neurons, 5 inputs each 
        self.output_layer = nn.Linear(3, 1) # 1 neuron with 3 inputs
    
      def forward(self, x):
        z = torch.heaviside(self.input_layer(x))
        y = self.output_layer(z)
        return y
    ```
  ]
]

#slide[
  #text(size: 21pt)[
    ```python
    import jax, equinox
    from jax import numpy as jnp
    from equinox import nn

    class MyNetwork(equinox.Module):
      input_layer: nn.Linear # Required by equinox
      output_layer: nn.Linear

      def __init__(self):
        self.input_layer = nn.Linear(5, 3, key=jax.random.PRNGKey(0))
        self.output_layer = nn.Linear(3, 1, key=jax.random.PRNGKey(1))

      def __call__(self, x):
        z = jnp.heaviside(self.input_layer(x))
        y = self.output_layer(z)
        return y 
    ```
  ]
]


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
