#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *

#set math.vec(delim: "[")
#set math.mat(delim: "[")

#let la = $angle.l$
#let ra = $angle.r$


// FUTURE TODO: Should not waste m/n in linear regression, use c for count and ell for input
// TODO: Work in bias earlier, as a means to shift the activation function
// TODO: Design matrix is not square, discuss XtX
// TODO: Reuse of n in neurons
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

#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 1: Introduction"
)

#title-slide(
  title: [Neural Networks],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
  //logo: image("logo.jpg", width: 25%)
)

/*
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
*/

#let agenda(index: none) = {
  let ag = (
    [Review],
    [Multivariate linear regression],
    [Limitations of linear regression],
    [History of neural networks],
    [Biological neurons],
    [Artificial neurons],
    [Perceptron],
    [Multilayer Perceptron]
  )
  for i in range(ag.len()){
    if index == i {
      enum.item(i + 1)[#text(weight: "bold", ag.at(i))]
    } else {
      enum.item(i + 1)[#ag.at(i)]
    }
  }
}

#slide[#agenda(index: none)]

#slide[#agenda(index: 0)]

#slide(title: [Review])[
  Since you are very educated, we focused on how education affects life expectancy #pause

  Studies show a causal effect of education on health #pause
    - _The causal effects of education on health outcomes in the UK Biobank._ Davies et al. _Nature Human Behaviour_. #pause
    - By staying in school, you are likely to live longer #pause
    - Being rich also helps, but education alone has a *causal* relationship with life expectancy
]

#slide(title: [Review])[
  *Task:* Given your education, predict your life expectancy #pause

  $X in bb(R)_+:$ Years in school #pause
  
  $Y in bb(R)_+:$ Age of death #pause

  $Theta in bb(R)^2:$ Parameters #pause 

  $ f: X times Theta |-> Y $


  *Approach:* Learn the parameters $theta$ such that 

  $ f(x, theta) = y; quad x in X, y in Y $ #pause

  *Goal:* Given someone's education, predict how long they will live
]

#slide(title: [Review])[
  Started with a linear function $f$ #pause

  #align(center, grid(
    columns: 2,
    align: center,
    column-gutter: 2em,
    $ f(x, bold(theta)) = f(x, vec(theta_1, theta_0)) = theta_1 x + theta_0 $,
    cimage("figures/lecture_2/example_regression_graph.png", height: 50%)
  )) #pause
  
  Then, we derived the square error function #pause

  $ "error"(f(x, bold(theta)), y) = (f(x, bold(theta)) - y)^2 $
]

#slide(title: [Review])[
  We wrote the loss function for a single datapoint $x_i, y_i$ using the square error

  $ cal(L)(x_i, y_i, bold(theta)) = "error"(f(x_i, bold(theta)),  y_i) = (f(x_i, bold(theta)) - y_i)^2 $ #pause

  But we wanted to learn a model over *all* the data, not a single datapoint #pause

  We wanted to make *new* predictions, to *generalize* #pause

  $ bold(x) = mat(x_1, x_2, dots, x_n)^top, bold(y) = mat(y_1, y_2, dots, y_n)^top $ #pause

  $ 
  cal(L)(bold(x), bold(y), bold(theta)) = sum_(i=1)^n "error"(f(x_i, bold(theta)),  y_i) = sum_(i=1)^n (f(x_i, bold(theta)) - y_i)^2 
  $
]

#slide(title: [Review])[
  Our objective was to find the parameters that minimized the loss function over the dataset #pause

  We introduced the $argmin$ operator #pause

  #side-by-side[ $f(x) = (x + 1)^2$][#argmin_plot] #pause

  $ argmin_x f(x) = -1 $
]

#slide(title: [Review])[
  With the $argmin$ operator, we formally wrote our optimization objective #pause

  $ 
   #text(fill: color.red)[$argmin_bold(theta)$] cal(L)(bold(x), bold(y), bold(theta)) &= #text(fill: color.red)[$argmin_bold(theta)$] sum_(i=1)^n "error"(f(x_i, bold(theta)),  y_i) \ &= #text(fill: color.red)[$argmin_bold(theta)$] sum_(i=1)^n (f(x_i, bold(theta)) - y_i)^2 
  $ #pause
]

#slide(title: [Review])[
  We defined the design matrix $bold(X)_D$ #pause

  $ bold(X)_D = mat(bold(x), bold(1)) = mat(x_1, 1; x_2, 1; dots.v, dots.v; x_n, 1) $ #pause

  With the design matrix, provided an *analytical* solution to the optimization objective #pause
  
$ bold(theta) = (bold(X)_D^top bold(X)_D )^(-1) bold(X)_D^top bold(y) $ #pause
]

#slide(title: [Review])[
  With this analytical solution, we were able to learn a linear model #pause

  #cimage("figures/lecture_2/linear_regression.png", height: 60%)
]

#slide(title: [Review])[
  Then, we used a trick to extend linear regression to nonlinear models #pause

  $ bold(X)_D = mat(x_1, 1; x_2, 1; dots.v, dots.v; x_n, 1) => bold(X)_D = mat(log(1 + x_1), 1; log(1 + x_2), 1; dots.v, dots.v; log(1 + x_n), 1) $ #pause
]

#slide(title: [Review])[
  We extended to polynomial, which are *universal function approximators* #pause

  $ bold(X)_D = mat(x_1, 1; x_2, 1; dots.v, dots.v; x_n, 1) => bold(X)_D = mat(
    x_1^m, x_1^(m-1), dots, x_1, 1; 
    x_2^m, x_2^(m-1), dots, x_2, 1; 
    dots.v, dots.v, dots.down; 
    x_n^m, x_n^(m-1), dots, x_n, 1
    ) $ #pause

  $ f: X times Theta |-> bb(R) $ #pause

  $ Theta in bb(R)^2 => Theta in bb(R)^m $ 
]

#slide(title: [Review])[
  Finally, we discussed overfitting #pause

  $ f(x, bold(theta)) = theta_n x^m + theta_(m - 1) x^(m - 1), dots, theta_1 + x^1 + theta_0 $ #pause

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
]

#slide(title: [Review])[
  We care about *generalization* in machine learning #pause

  So we should always split our dataset into a training dataset and a testing dataset #pause

  #cimage("figures/lecture_2/train_test_regression.png", height: 60%)
]

#slide[#agenda(index: 0)]
#slide[#agenda(index: 1)]

#slide[
  Last time, we assumed a single-input system #pause

  Years of education: $X in bb(R)$ #pause

  But sometimes we want to consider multiple input dimensions #pause

  Years of education, BMI, GDP: $X in bb(R)^3$ #pause

  We can solve these problems using linear regression too
]

#slide[
  For multivariate problems, we use vector inputs #pause

  $ bold(x) in X; quad X in bb(R)^3 $ #pause

  I will write

  $ bold(x)_i = vec(
    x_i angle.l 1 angle.r,
    x_i angle.l 2 angle.r,
    x_i angle.l 3 angle.r
  ) $ #pause

  $x_i angle.l 1 angle.r$ refers to the first dimension of training data $i$
]

#slide[
  Assume an input space $X in bb(R)^ell$ #pause

  The design matrix for this *multivariate* linear system is

  $ bold(X)_D = mat(bold(X), bold(1)) = mat(
    x_1 angle.l ell angle.r, x_1 angle.l ell - 1 angle.r, dots, 1; 
    x_2 angle.l ell angle.r, x_2 angle.l ell - 1 angle.r, dots, 1; 
    dots.v, dots.v, dots.down, dots.v; 
    x_n angle.l ell angle.r, x_n angle.l ell - 1 angle.r, dots, 1
  ) $ #pause

  Remember $x_n angle.l ell$ refers to dimension $ell$ of training data $n$ #pause
]

#slide[
  We previously looked at linear and polynomial models for regression #pause

  $ f(bold(x), bold(theta)) = bold(X)_D bold(theta) = theta_(m) x^m + theta_(m - 1) x^(m - 1) + dots + theta_0 $ #pause

  $ bold(theta) = (bold(X)^top bold(X) )^(-1) bold(X)^top bold(y) $
]

#slide[
  Linear models are useful for certain problems #pause
  + Interpretability #pause
  + Low data requirement #pause

  But issues arise with other problems #pause
  + Poor scalability #pause
  + Polynomials do not generalize well
]

#slide[
  Issues with very complex problems
  + *Poor scalability* 
  + Polynomials do not generalize well
]

#slide[
  Last time, we learned a polynomial function of a *one-dimensional* $x$ using linear regression #pause

  However, we can also learn such functions for *multi-dimensional* $x$

  //Polynomials fit tabular data well #pause

  //However, they scale poorly to higher-dimensional data like image pixels #pause

  #side-by-side[#cimage("figures/lecture_1/dog.png", height: 30%)][$ X in bb(Z)_+^(256 times 256) $] #pause

  This image contains 65536 pixels, so $x$ has 65536 dimensions
]

#slide[
  #side-by-side[#cimage("figures/lecture_1/dog.png", height: 30%)][$ 256 times 256 "pixels" = 65536 "pixels" $] #pause
  
  What does the design matrix look like for an m-degree polynomial of this image? 

]

#slide[
  $ bold(X)_D = mat(
    x_1^m, x_1^(m-1), dots, x_1^1, 1;
    x_2^m, x_2^(m-1), dots, x_2^1, 1;
    dots.v, dots.v, dots.down, dots.v, dots.v;
    x_n^m, x_n^(m-1), dots, x_n^1, 1;
    x_1^(n-1) x_2, x^(n-2) x_2^2, dots, 0, 1;
    dots.v, dots.v, dots.down, dots.v, dots.v;
    x_1 x_2 dots x_n, 0, dots, 0, 1;
  ) $ #pause

  *Question:* How big is the matrix for 65,536 pixels and $m=3$?

  *Answer:* $65,536^3 approx 10^14$ parameters #pause
]

#slide[
  *Question:* How big is the matrix for 65,536 pixels and $n=3$?

  *Answer:* $65,536^3 approx 10^14$ parameters #pause

  For comparison, GPT-4 has $10^12$ parameters #pause

  We must invert $bold(X)_D^top bold(X)_D$, requiring $O(n^3)$ time #pause

  Largest matrix ever inverted has $approx 10^12$ elements #pause

  One day this will be possible, but today it is not #pause

  Polynomial regression scales poorly to high dimensional data
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
  What happens to polynomials outside of the support (dataset)? #pause

  #side-by-side[$ theta_m x^m + theta_(m-1) x^(m-1) + dots $][Equation of a polynomial] #pause

  #side-by-side[$ x^m (theta_m + theta_(m-1) / x + dots) $][Factor out $x^m$] #pause

  #side-by-side[$ lim_(x -> oo) x^m (theta_m + theta_(m-1) / x + dots) $][Take the limit] #pause

  #side-by-side[$ lim_(x -> oo) x^m dot lim_(x-> oo) (theta_m + theta_(m-1) / x + dots) $][Split the limit (limit of products)] 
]

#slide[
  #side-by-side[$ lim_(x -> oo) x^m dot lim_(x-> oo) (theta_m + theta_(m-1) / x + dots) $][Split the limit (limit of products)]
  
  #side-by-side[$ (lim_(x -> oo) x^m) dot (theta_m + 0 + dots) $][Evaluate right limit] #pause

  #side-by-side[$ theta_m lim_(x -> oo) x^m  $][Rewrite] #pause

  #side-by-side[$ theta_m lim_(x -> oo) x^m = oo $][If $theta_m > 0$] #pause

  #side-by-side[$ theta_m lim_(x -> oo) x^m = -oo $][If $theta_m < 0$]
]
#slide[
  Polynomials quickly tend towards $-oo, oo$ outside of the support #pause

  $ f(x) = x^3-2x^2-x+2 $ #pause

  #cimage("figures/lecture_3/polynomial_generalize.png", height: 50%) #pause

  Remember, to predict new data we want our functions to generalize
]

#slide[
  Linear regression has issues #pause
  + Poor scalability #pause
  + Polynomials do not generalize well
]

#slide[#agenda(index: 1)]
#slide[#agenda(index: 2)]

#slide[
  Can we improve upon linear regression? #pause

  Yes, with neural networks

  //#cimage("figures/lecture_1/timeline.svg", height: 50%) 
]

#slide[
  In 1939-1945, there was a World War #pause

  Militaries invested funding for research, and invented the computer #pause

  #cimage("figures/lecture_3/turing.jpg", height: 70%)
]

#slide[
  #side-by-side[Meanwhile, a neuroscientist and mathematician (McCullough and Pitts) were trying to understand the human brain][#cimage("figures/lecture_3/mccullough-pitts.png", height: 70%)] #pause

  They designed the theory for the first neural network
]

#slide[
  #side-by-side[
  A few years later, Rosenblatt implemented this neural network using a new invention -- the computer
][#cimage("figures/lecture_3/original_nn.jpg", height: 70%)] 
]

#slide[
  Through advances in theory and hardware, neural networks became slightly better #pause

  #cimage("figures/lecture_1/timeline.svg", height: 40%) #pause

  Around 2012, these improvements culminated in neural networks that perform like humans
]


#slide[
  So what is a neural network? #pause

  It is a function, inspired by how the brain works #pause

  $ f: X times Theta |-> Y $ 

]

#slide[
  Brains and neural networks rely on *neurons* #pause

  *Brain:* Biological neurons $->$ Biological neural network #pause

  *Computer:* Artificial neurons $->$ Artificial neural network #pause

  First, let us review biological neurons #pause

  *Note:* I am not a neuroscientist! I may make simplifications or errors with biology
]

#slide[#agenda(index: 2)]
#slide[#agenda(index: 3)]

#slide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  A simplified neuron consists of many parts 

]

#slide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  Neurons send and process messages from other neurons
]

#slide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  Incoming electrical signals travel along dendrites
]

#slide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  Dendrites are not all equal! Different dendrites have different diameters and structures
]

#slide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  The axon outputs an electrical signal to other neurons
]

#slide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  The axon terminals will connect to dendrites of other neurons
]

#slide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  For our purposes, we can consider the axon terminals  and dendrites to be the same thing
]

#slide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  The neuron takes many inputs, and produces a single output
]

#slide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  The neuron will only output a signal down the axon ("fire") at certain times
]

#slide[
  How does a neuron decide to send an impulse ("fire")? #pause

  #side-by-side[Incoming impulses (via dendrites) change the electric potential of the neuron][  #cimage("figures/lecture_3/bio_neuron_activation.png", height: 50%)] #pause

  Recall that in a parallel circuit, we can sum voltages together #pause

  Many active dendrites will add together and trigger an impulse
  
]

#slide[
  #side-by-side[Pain triggers initial nerve impulse, starts a chain reaction into the brain][#cimage("figures/lecture_3/nervous-system.jpg")]
]

#slide[
  #side-by-side[When the signal reaches the brain, we will think][#cimage("figures/lecture_3/nervous-system.jpg")]
]

#slide[
  #side-by-side[After thinking, we will take action][#cimage("figures/lecture_3/nervous-system.jpg")]
]

#slide[#agenda(index: 3)]
#slide[#agenda(index: 4)]

#slide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg", height: 50%) #pause

  *Question:* How could we write a neuron as a function? $quad f: "___" |-> "___"$ #pause

  *Answer*:

  $ f: underbrace(bb(R)^m, "Dendrite voltages") times underbrace(bb(R)^m, "Dendrite size") |-> underbrace(bb(R), "Axon voltage") $
]


#slide[
  Let us implement an artifical neuron as a function #pause

  #side-by-side[#cimage("figures/lecture_3/neuron_anatomy.jpg")][
    #only((2,3))[
      Neuron has a structure of dendrites

    ]
    #only(3)[
      $ f(vec(theta_1, theta_2, dots.v, theta_n)) = f(vec(0.5, 3.1, dots.v, 2.0)) $

    ]
    #only((4,5))[
      Each incoming dendrite has some voltage potential
    ]

    #only(5)[
      $ f(vec(x_i angle.l 1 angle.r, dots.v, x_i angle.l n angle.r), vec(theta_(1),  dots.v, theta_(n)) ) $

      $ bold(x)_i = vec(x_i angle.l 1 angle.r, dots.v, x_i angle.l n angle.r) = vec(0.5, dots.v, -0.3) $
    ]

    #only((6, 7))[
      Voltage potentials sum together to give us the voltage in the cell body

    ]

    #only(7)[
      $ f(vec(x_i angle.l 1 angle.r, dots.v, x_i angle.l n angle.r), vec(theta_(1),  dots.v, theta_(n)) ) = sum_(j=1)^n x_i angle.l j angle.r theta_j $
    ]

    #only((8, 9, 10))[
      The axon fires only if the voltage is over a threshold
    ]
    #only((9, 10))[
      
      $ sigma(x) = #image("figures/lecture_3/heaviside.png", height: 30%) $
    ]
    #only(10)[
      $ f(vec(x_1, dots.v, x_n), vec(theta_(1),  dots.v, theta_(n)) ) = sigma(sum_(j=1)^n x_i angle.l j angle.r theta_j) $
    ]
  ]
]

#slide[
  This is almost the artificial neuron!
  $ f(vec(x_1, dots.v, x_n), vec(theta_(1),  dots.v, theta_(n)) ) = sigma(sum_(j=1)^n x_i angle.l j angle.r theta_j) $ #pause

  $ f(bold(x), bold(theta)) = sigma(sum_(j=1)^n x angle.l j angle.r theta_j) $ #pause

  Let us write this out for clarity
]

#slide[
  $ f(bold(x), bold(theta)) = sigma(
    x angle.l n angle.r theta_n + 
    x angle.l n-1 angle.r theta_(n-1) + 
    dots + 
    x angle.l 1 angle.r theta_1
  ) $ #pause

  *Question:* Does this look familiar to anyone? #pause

  *Answer:* This is a multivariate linear model!
]


/*
  *Question:* Does it look familiar to any other functions we have seen? #pause

  *Answer:* The linear model!
]
*/
#slide[
  #side-by-side[$ f(bold(x), bold(theta)) = sigma(sum_(i=1)^n x_i theta_i) $][Artificial neuron] #pause

  #side-by-side[$ f(bold(x), bold(theta)) = theta_0 + theta_1 x_1 + theta_2 x_2  + dots + theta_n x_n $][Linear model] #pause

  It is the linear model with an activation function! #pause

  We add a bias term to the neuron, for the same reason we add a bias term to the linear model #pause

  $ f(bold(x), bold(theta)) = sigma(theta_0 + sum_(i=1)^n x_i theta_i) $ #pause
]


#slide[
  #side-by-side(gutter: 4em)[#cimage("figures/lecture_3/neuron_anatomy.jpg")
  $ f(vec(x_1, dots.v, x_n), vec(theta_(0),  dots.v, theta_(n)) ) = sigma(theta_0 + sum_(i=1)^n x_i theta_i) $  
  ][  
  #cimage("figures/lecture_3/neuron.png")]
]

#slide[
  #text(size: 23pt)[
  We can also write a neuron in terms of a dot product #pause

  $ f(vec(x_1, dots.v, x_n), vec(theta_(0),  dots.v, theta_(n)) ) = sigma(theta_0 + bold(theta)_(1:n) dot bold(x)) $ #pause

  We often write the parameters as a *weight* $bold(w)$ and *bias* $b$ #pause

  $ f(vec(x_1, dots.v, x_n), vec(b, w_1, dots.v, w_(n)) ) = sigma(b + bold(w) dot bold(x)) $ #pause

  $ b = theta_0, bold(w) = bold(theta)_(1:n) $
  ]
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
  How do we express a *wide* neural network mathematically? #pause

  A single neuron:

  $ f: bb(R)^n, bold(theta) |-> bb(R) $ #pause

  Multiple neurons (wide):

  $ f: bb(R)^n, bold(theta) |-> bb(R)^m $ #pause
]

#slide[
  For a single neuron:
  
  $ f(vec(x_1, dots.v, x_n), vec(theta_(0),  dots.v, theta_(n)) ) = sigma(theta_0 + sum_(i=1)^n x_i theta_i) $  

  $ f(vec(x_1, dots.v, x_n), vec(theta_(0),  dots.v, theta_(n)) ) = sigma(theta_0 + bold(theta)_(1:n) dot bold(x)) $ #pause
]

#slide[
  // Must be m by n (m rows, n cols)
  $ f(vec(x_1, dots.v, x_n), mat(theta_(1,0), theta_(2, 0), dots, theta_(n,0); theta_(1,1), theta_(2,1), dots, theta_(n, 1); dots.v, dots.v, dots.down, dots.v; theta_(1, m), theta_(2, m), dots, theta_(m, n)) ) = vec(
    sigma( theta_(1,0) + sum_(i=1)^n x_i theta_(1,i)  ),
    sigma( theta_(2,0) + sum_(i=1)^n x_i theta_(2,i)  ),
    dots.v,
    sigma( theta_(m,0) + sum_(i=1)^n x_i theta_(m,i)  ),
  ) 
  $  
  Each row in the output corresponds to the output of a single neuron #pause

  This is very confusing to write, but we can rewrite it as matrix multiplication
]

#slide[
  $ f(vec(x_1, dots.v, x_n), mat(theta_(1,0), theta_(2, 0), dots, theta_(n,0); theta_(1,1), theta_(2,1), dots, theta_(n, 1); dots.v, dots.v, dots.down, dots.v; theta_(1, m), theta_(2, m), dots, theta_(m, n)) ) = vec(
    sigma( theta_(1,0) + sum_(i=1)^n x_i theta_(1,i)  ),
    sigma( theta_(2,0) + sum_(i=1)^n x_i theta_(2,i)  ),
    dots.v,
    sigma( theta_(m,0) + sum_(i=1)^n x_i theta_(m,i)  ),
  ) 
  $ #pause

  $ f(bold(x), bold(theta)) = sigma( bold(theta)_(dot, 0) + bold(theta)_(dot, 1:n) bold(x) ) $ #pause

  $ f(bold(x), (bold(b), bold(W))) = sigma( bold(b) + bold(W) bold(x) ) $
]

#slide[
  How do we express a *deep* neural network mathematically? #pause

  A single neuron:

  $ f: bb(R)^n, bold(theta) |-> bb(R) $ #pause

  Multiple neurons (deep):

  $ f: bb(R)^n, bold(theta), bold(psi), dots, bold(rho) |-> bb(R)^m $ 
]

#slide[
  A single neuron

  $ f(bold(x), bold(theta)) = bold(theta)_(dot, 0) + bold(theta)_(dot, 1:n) bold(x) $ #pause

  A composition of neurons with parameters $bold(theta), bold(psi), bold(rho)$

  #text(size: 22pt)[
    $ f_1(bold(x), bold(theta)) = bold(theta)_(dot, 0) + bold(theta)_(dot, 1:n) bold(x) quad

    f_2(bold(x), bold(psi)) = bold(psi)_(dot, 0) + bold(psi)_(dot, 1:n) bold(x) quad

    dots quad

    f_(ell)(bold(x), bold(rho)) = bold(rho)_(dot, 0) + bold(rho)_(dot, 1:n) bold(x) $ #pause
  ] #pause


  $ f_(ell) (dots f_2(f_1(bold(x), bold(theta)_1), bold(psi)) dots ) $
]

#slide[
  Written more plainly as

  $ bold(z)_1 = f_1(bold(x), bold(theta)) = bold(theta)_(dot, 0) + bold(theta)_(dot, 1:n) bold(x) $ #pause
  $ bold(z)_2 = f_2(bold(z_1), bold(psi)) = bold(psi)_(dot, 0) + bold(psi)_(dot, 1:n) bold(z)_1 $ #pause
  $ dots.v $ #pause
  $ bold(y) = f_(ell)(bold(x), bold(rho)) = bold(rho)_(dot, 0) + bold(rho)_(dot, 1:n) bold(z)_(ell - 1) $ #pause
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

  #only(4)[$ "Roughly, " exists bold(theta) => lim_(n |-> oo) [ theta_(2, 0) + theta_(2, 1) sum_(j = 1)^n sigma(theta_(1, 0) + theta_(1, j) x) ] = g(x); quad forall g $]
]

#slide[
  More formally, a wide and deep neural network is a *universal function approximator* #pause

  It can approximate *any* continuous function to precision $epsilon$ #pause

  $ | g(bold(x)) - f(bold(x), bold(theta)) | < epsilon $ #pause
  
  As we increase the width and depth of the network, $epsilon$ shrinks #pause

  #side-by-side[$ g(#image("figures/lecture_1/dog.png", height: 20%)) = "Dog"$][$ g(#image("figures/lecture_1/muffin.png", height: 20%)) = "Muffin" $] 
  
  #align(center)[Very powerful finding! The basis of deep learning.]
]

#slide[
  All the models we examine in this course will use this neural network structure internally #pause
    - Transformers #pause
    - Graph neural networks #pause
]

#focus-slide[Relax]

#slide[
  We call this form of a neural network a *feedforward network* or *perceptron* (invented in 1943) #pause

  #cimage("figures/lecture_3/mark_1_perceptron.jpeg", height: 60%)

  $20 times 20$ grid of pixels to process images
]

#slide[

  #cimage("figures/lecture_3/timeline.svg", width: 85%) #pause

  *Question:* If the deep neural network was invented in 1958, why did it take 70 years for us to care about deep learning?
]

#slide[
  *Answer:* Deep learning requires very deep and wide networks
    + Hardware advances enabled very deep and wide networks #pause

    + Many theoretical improvements allow us to successfully train deeper and wider networks #pause
]

#slide[
  The neural network we created today is called a feedforward network or perceptron #pause

  When the network is deep, we call it a Multi-Layer Perceptron (MLP) #pause

  We often use the term "layers", when referring to a specific depth of the neural network
    - Four-layer MLP means a neural network with a depth of four
    - Corresponds to four parameter matrices in $bold(theta)$
]

#slide[
  Let us construct deep and wide neural networks in `torch` and `jax`
]

#slide[
  Here are the equations for one neural network layer 

  #side-by-side[$ f(bold(x), bold(theta)) = sigma( bold(theta)_(dot, 0) + bold(theta)_(dot, 1:n) bold(x) ) $][or][
  $ f(bold(x), (bold(b), bold(W))) = sigma( bold(b) + bold(W) bold(x) ) $ ] #pause

  We must implement the linear function $bold(b) + bold(W) bold(x)$ and the activation function $sigma$ to create a neural network layer #pause

  Let us do this in colab! https://colab.research.google.com/drive/1bLtf3QY-yROIif_EoQSU1WS7svd0q8j7?usp=sharing
]


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
