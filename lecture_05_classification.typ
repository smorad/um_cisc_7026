#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "common.typ": *
#import "@preview/pinit:0.2.2": *
#import "@preview/algorithmic:1.0.4"
#import algorithmic: style-algorithm, algorithm-figure, algorithm
#import "@preview/mannot:0.3.0": *

#let handout = true

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(handout: handout),
  config-info(
    title: [Classification],
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

#let sgd =  algorithm(
  line-numbers: false,
  {
    import algorithmic: *
    Procedure(
      "Gradient Descent",
      ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$),
      {
        For(
          $i in {1 dots t}$,
          {
            Comment[Compute the gradient of the loss]        
            Assign[$bold(J)$][$(gradient_bold(theta) cal(L))(bold(X), bold(Y), bold(theta))$]
            Comment[Update the parameters using the negative gradient]
            Assign[$bold(theta)$][$bold(theta) - alpha dot bold(J)$]
          },
        )
        Return[$bold(theta)$]
      },
    )
  }
)

#let heaviside = canvas(length: 1cm, {
  plot.plot(size: (8, 4),
    x-tick-step: 5,
    y-tick-step: 2,
    {
      plot.add(
        domain: (-10, 10), 
        x => calc.clamp(calc.floor(x + 1), 0, 1),
        style: (stroke: (paint: red, thickness: 4pt)),
      )
    })
})

#let sigmoid = canvas(length: 1cm, {
  plot.plot(size: (8, 4),
    x-tick-step: 5,
    y-tick-step: 2,
    {
      plot.add(
        domain: (-10, 10), 
        x => calc.pow(1 + calc.exp(-x), -1),
        style: (stroke: (paint: red, thickness: 4pt)),
      )
    })
})

// FUTURE TODO: Remove event space, only sample space
// FUTURE TODO: Fix bounds for probs (0, 1) or [0, 1]

#title-slide()

== Outline <touying:hidden>

#components.adaptive-columns(
    outline(title: none, indent: 1em, depth: 1)
)

= Admin

==

#cimage("figures/lecture_5/wedding.jpg")

==
I am sorry I was away for so long! #pause

Was only supposed to miss one lecture
- Lecture day changed from Tuesday to Friday 
- Big storm cancelled flights

==
Yutao tells me his lecture went well
- Hopefully you enjoyed it!

My research students told me exam went well

==
There is a possibility you lost power and internet during the storm

I was not here to answer homework questions during office hours

I will extend the Assignment 2 deadline by one week

==

Assignment 3 is now released

Discuss it at the end of lecture

Do not forget about the final project!

You should start to form groups soon
- Talk to your classmates, make friends!

= Review

==
  Last time, we reviewed derivatives #pause

  $ f'(x) = d / (d x) f = (d f) / (d x) = lim_(h -> 0) (f(x + h) - f(x)) / h $ #pause

  and gradients #pause
  
  $ (gradient_bold(x) f)(bold(x)) = vec((partial f(bold(x))) / (partial x_1), (partial f(bold(x))) / (partial x_2), dots.v, (partial f(bold(x))) / (partial x_n)) $

==
  Gradients are important in deep learning for two reasons: #pause

  $ bold("Reason 1:") f(bold(x)) "has critical points at" (gradient_bold(x) f)(bold(x)) = 0 $ #pause

  #cimage("figures/lecture_5/saddle.png") #pause

  With optimization, we attempt to find minima of loss functions

==
  Gradients are important in deep learning for two reasons: #pause

  *Reason 2:* For problems without analytical solutions, the gradient (slope) is necessary for gradient descent #pause

  #cimage("figures/lecture_4/gradient_descent_3d.png", height: 60%)

==
  First, we derived the solution to linear regression #pause

  $ cal(L)(bold(X), bold(Y), bold(theta)) = sum_(i=1)^n ( f(bold(x)_[i], bold(theta)) - bold(y)_[i] )^2 $ #pause

  $ cal(L)(bold(X), bold(Y), bold(theta)) = ( bold(Y) - overline(bold(X)) bold(theta) )^top ( bold(Y) - overline(bold(X)) bold(theta) ) $  #pause

  $ cal(L)(bold(X), bold(Y), bold(theta)) = 
  underbrace(underbrace(( bold(Y) - overline(bold(X)) bold(theta) )^top, "Linear function of " theta quad)
  underbrace(( bold(Y) - overline(bold(X)) bold(theta) ), "Linear function of " theta), "Quadratic function of " theta) $

==
  #side-by-side[A quadratic function has a single critical point, which must be a global minimum][#cimage("figures/lecture_4/quadratic_parameter_space.png", height: 100%)]

==
  We found the analytical solution for linear regression by finding where the gradient was zero and solving for $bold(theta)$ #pause
  
  $ (gradient_bold(theta) cal(L))(bold(X), bold(Y), bold(theta)) = 0 $ #pause

  $ bold(theta) = (overline(bold(X))^top overline(bold(X)))^(-1) overline(bold(X))^top bold(Y) $ #pause

  Which solves
  $ argmin_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) $ 

==
  For neural networks, the square error loss is no longer quadratic #pause

  #side-by-side[$ cal(L)(x, y, bold(theta)) = (f(x, bold(theta)) - y)^2 $][Loss function] #pause
  #side-by-side[$ f(x, bold(theta)) = sigma(theta_0 + theta_1 x) $][Neural network model] #pause

  Now, we plug the model $f$ into the loss function #pause 

  $ cal(L)(x, y, bold(theta)) = (sigma(theta_0 + theta_1 x) - y)^2 $ #pause

  $ cal(L)(x, y, bold(theta)) = 
  underbrace((sigma(theta_0 + theta_1 x) - y), "Nonlinear function of" theta) 
  underbrace((sigma(theta_0 + theta_1 x) - y), "Nonlinear function of" theta) $ #pause

  There is no analytical solution for $bold(theta)$

==
  Instead, we found neural network parameters through gradient descent #pause

  Gradient descent is an optimization method for differentiable functions #pause

  We went over both the intuition and mathematical definitions

==
  #side-by-side[
    #cimage("figures/lecture_4/lightning.jpg", height: 100%) #pause
   ][
    #cimage("figures/lecture_4/hiking_slope.jpg", height: 100%)
   ] 

==
  #cimage("figures/lecture_4/parameter_space.png", height: 100%)

==
  The gradient descent algorithm: 

  #sgd


==
  We derived the $gradient_bold(theta) cal(L)$ for deep neural networks using the chain rule #pause

  $ (gradient_bold(theta) cal(L))(bold(X), bold(Y), bold(theta)) = sum_(i = 1)^n 2 ( f(bold(x)_[i], bold(theta)) - bold(y)_[i]) #redm[$(gradient_bold(theta) f)(bold(x)_[i], bold(theta))$] 
  $ #pause
   

  $ 
  #redm[$(gradient_(bold(theta)) f)(bold(x), bold(theta))$] = (gradient_(bold(phi), bold(psi), dots, bold(xi)) f)(bold(x), vec(bold(phi), bold(psi), dots.v, bold(xi))) = vec(
    (gradient_bold(phi) f_1)(bold(x), bold(phi)), 
    (gradient_(bold(psi)) f_2)(bold(z)_1, bold(psi)), 
    dots.v, 
    #redm[$(gradient_(bold(xi)) f_(ell))(bold(z)_(ell - 1), bold(xi))$]
  ) 
  $ #pause

  $ #redm[$(gradient_(bold(xi)) f_ell) (bold(z)_(ell - 1), bold(xi))$] = (sigma(bold(xi)^top overline(bold(z))_(ell - 1)) dot.o (1 - sigma(bold(xi)^top overline(bold(z))_(ell - 1)))) overline(bold(z))_(ell - 1) $ 

==
  The heaviside step function was not differentiable #pause

  We could not use it with gradient descent #pause

  Replaced heaviside with smooth approximation (sigmoid) #pause

  #side-by-side[
    #heaviside
  ][
    #sigmoid
  ]

  //#side-by-side[#cimage("figures/lecture_4/heaviside.svg")][#cimage("figures/lecture_4/sigmoid.svg")][ $ sigma(z) = 1 / (1 + e^(-z)) $]


==
  In `jax`, we compute the gradient using the `jax.grad` function #pause

  ```python
  import jax

  def L(theta, X, Y):
    ...

  # Create a new function that is the gradient of L
  # Then compute gradient of L for given inputs
  J = jax.grad(L, argnums=2)(X, Y, theta)
  # Update parameters
  alpha = 0.0001
  theta = theta - alpha * J
  ```

==
  In `torch`, we backpropagate through a graph of operations
  
  ```python
  import torch
  optimizer = torch.optim.SGD(params, lr=0.0001)

  def L(model, X, Y):
    ...
  # Pytorch will record a graph of all operations
  # Everytime you do theta @ x, it stores inputs and outputs 
  loss = L(X, Y, model) # compute loss 
  # Traverse the graph backward and compute the gradient
  loss.backward() # Sets .grad attribute on each parameter
  optimizer.step() # Update the parameters using .grad
  optimizer.zero_grad() # Set .grad to zero, DO NOT FORGET!!
  ```
==


// 30:00
  First, a video of one application of gradient descent 

  https://youtu.be/kGDO2e_qiyI?si=ZopZKy-6WQ4B0csX #pause

  #v(1em) 
  Time for some interactive torch coding

  https://colab.research.google.com/drive/1W8WVZ8n_9yJCcOqkPVURp_wJUx3EQc5w

// ~60:00


= Classification
==
  Many problems in ML can be reduced to *regression* or *classification* #pause

  *Regression* asks how many #pause
  - How long will I live? #pause
  - How much rain will there be tomorrow? #pause
  - How far away is this object? #pause

  *Classification* asks which one #pause
  - Is this a dog or muffin? #pause
  - Will it rain tomorrow? Yes or no? #pause
  - What color is this object? #pause
  
  So far, we only looked at regression. Now, let us look at classification

==
  *Task:* Given a picture of clothes, predict the text description #pause

  $X: bb(Z)_(0,255)^(32 times 32) #image("figures/lecture_5/classify_input.svg", width: 80%)$ #pause

  $Y : & {"T-shirt, Trouser, Pullover, Dress, Coat,"\ 
    & "Sandal, Shirt, Sneaker, Bag, Ankle boot"}$ #pause

  *Approach:* Learn $bold(theta)$ that produce *conditional probabilities*  #pause

  $ f(bold(x), bold(theta)) = P(bold(y) | bold(x)) = P(vec("T-Shirt", "Trouser", dots.v) mid(|) #image("figures/lecture_5/shirt.png", height: 20%)) = vec(0.2, 0.01, dots.v) $

==

= Probability Theory
==
  Classification tasks require *probability theory*, so let us review #pause

  In probability, we have *experiments* and *outcomes* #pause

  An experiment yields one of many possible outcomes #pause

  #side-by-side[Experiment][Outcome] #pause
  #side-by-side[Flip a coin][Heads] #pause
  #side-by-side[Walk outside][Rain] #pause
  #side-by-side[Grab clothing from closest][Coat] 
  

==
  The *sample space* $S$ defines all possible outcomes for an experiment #pause

  #side-by-side[Experiment][Sample Space $S$] #pause
  #side-by-side[Flip a coin][$ S = {"heads", "tails"} $] #pause

  #side-by-side[Walk outside][$ S = {"rain", "sun", "wind", "cloud"} $] #pause

  #side-by-side[Take clothing from closet][$ S = {"T-shirt", "Trouser", "Pullover", "Dress", \ 
  "Coat", "Sandal", "Shirt", "Sneaker", "Bag", \
  "Ankle boot"} $]

==
  An *event* $E$ is a specific subset of the sample space #pause

  #side-by-side[Experiment][Sample Space][Event] #pause

  #side-by-side[Flip a coin][$ S = {"heads", "tails"} $][$ E = {"heads"} $] #pause

  #side-by-side[Walk outside][$ S = {"rain", "sun", "wind", "cloud"} $][$ E = {"rain", "wind"} $] #pause

  #side-by-side[Take from closet][$ {
    "T-shirt", "Trouser", \ 
    "Pullover",  "Dress", \
  "Coat", "Sandal", "Shirt", \ 
  "Sneaker", "Bag", "Ankle boot"} $][$ E = {"Shirt", "T-Shirt", "Coat"} $]

==
  The *probability* measures how likely an event is to occur #pause

  The probability must be between 0 (never occurs) and 1 (always occurs) #pause

  $ 0 <= P(E) <= 1 $ #pause
  
  #side-by-side[Experiment][Probabilities] #pause

  #side-by-side[Flip a coin][$ P("heads") = 0.5 $] #pause

  #side-by-side[Walk outside][$ P("rain") = 0.15 $] #pause

  #side-by-side[Take from closet][$ P("Shirt") =  0.1 $]

==
When we define $P$ as a function over the sample space $S$, we call it a *Probability Distribution* #pause

For a discrete sample space $S$, the distribution is a *Probability Mass Function (PMF)* that maps each outcome to a probability #pause

$ P: S -> [0, 1] $ #pause

The distribution must sum to one over the sample space #pause

$ sum_(x in S) P(x) = 1 $  #pause

==
$ sum_(x in S) P(x) = 1 $  #pause

  #side-by-side[Flip a coin][$ {P("heads") = 0.5, P("tails") = 0.5} $] #pause

  #side-by-side[Take clothing from closet][$ {P("T-shirt") = 0.1, P("Trouser") = 0.08, \ P("Pullover") = 0.12, dots } $]

==
We can also define $P$ over the event space $E$ #pause

$ P: E -> [0, 1] $ #pause

*Question:* Does it still sum to one?

*Answer:* Not always!

#side-by-side[Flip a coin][
  $ P({"heads"}) = 0.5 $ 
  $ P({}) = 0 $
  $ P({"heads", "tails"}) = 1.0 $ 
  ] #pause



==
  $P$ is a function, so we can plot it

  #cimage("figures/lecture_5/pmf.jpg", width: 100%)

==
  Events can overlap with each other #pause
  - Disjoint events #pause
  - Conditionally dependent events

==
  Two events $A, B$ are *disjoint* if $A$ and $B$ cannot occur together #pause

  $ P(A, B) = 0 $ #pause

  #side-by-side[Flip a coin][
    $P("Heads") = 0.5, P("Tails") = 0.5$
    $P("Heads", "Tails") = 0$
  ] #pause

  Be careful!

  #side-by-side[Walk outside][
    $P("Rain") = 0.1, P("Cloud") = 0.2$
    $P("Rain" inter "Cloud") != 0$
  ]

==
  If $B$ provides information about $A$, they are *conditionally dependent* #pause

  $ P(A | B) != P(A) $ #pause

  $ P("cloud") = 0.2, P("rain") = 0.1 $ #pause

  $ P("rain" | "cloud") = 0.5 $ #pause

  $ P(A | B) = P(A, B) / P(B) $ #pause

  #side-by-side[Walk outside][
    $P("Rain", "Cloud") = 0.1 \ 
    P("Cloud") = 0.2$
    $P("Rain" | "Cloud") = 0.1 / 0.2 = 0.5$
  ]

==
  *Task:* Given a picture of clothes, predict the text description #pause

  $X: bb(Z)_(0,255)^(32 times 32) #image("figures/lecture_5/classify_input.svg", width: 80%)$ #pause

  $Y : & {"T-shirt, Trouser, Pullover, Dress, Coat,"\ 
    & "Sandal, Shirt, Sneaker, Bag, Ankle boot"}$ #pause

  *Approach:* Learn $bold(theta)$ that produce *conditional probabilities*  #pause

  $ f(bold(x), bold(theta)) = P(bold(y) | bold(x)) = P(vec("T-Shirt", "Trouser", dots.v) mid(|) #image("figures/lecture_5/shirt.png", height: 20%)) = vec(0.2, 0.01, dots.v) $

= Classification Model
==

  We will start with a multivariate linear model #pause
  
  $ f(bold(x), bold(theta)) = bold(theta)^top overline(bold(x)) $ #pause

  We want our model to predict the probability of each outcome #pause

  *Question:* What is the function signature of $f$? #pause

  *Answer:* $f: bb(R)^(d_x) times Theta |-> bb(R)^(d_y)$ #pause

  *Question:* Can we use this model to predict probabilities? #pause

  *Answer:* No! Outcome probabilities must be $in [0, 1]$ and sum to one 

  //*Question:* What is the range/co-domain of $f$? #pause

  //*Answer:* $[-oo, oo]$ #pause



  //However, the probabilities must sum to one #pause

  //We introduce the *softmax* operator to ensure all probabilities sum to 1 #pause

==
  How can we model a distribution over outcomes? #pause

  $ bold(v) = { vec(v_1, dots.v, v_(d_y)) mid(|) quad sum_(i=1)^(d_y) v_i = 1; quad v_i in [0, 1] } $ #pause

  There is special notation for this vector, called the *simplex* #pause

  $ Delta^(d_y - 1) $

==
  The simplex $Delta^(k-1)$ is an $k - 1$-dimensional triangle in $k$-dimensional space #pause

  #cimage("figures/lecture_5/simplex.svg", height: 70%)

  It has only $k - 1$ free variables, because $x_(k) = 1 - sum_(i=1)^(k - 1) x_i$ 

// 96:00

==
  $ f: bb(R)^(d_x) times Theta |-> bb(R)^(d_y) $ #pause

  So we need a function that maps to the simplex #pause

  $ g: bb(R)^(d_y) |-> Delta^(d_y - 1) $ #pause

  Then, we can combine $f$ and $g$

  $ g(f): bb(R)^(d_x) times Theta |-> Delta^(d_y - 1) $ 

==
  We need a function $g$

  $ g: bb(R)^(d_y) |-> Delta^(d_y - 1) $ #pause

  There are many functions that can do this #pause

  One example is dividing by the $L_1$ norm: 
  
  $ g(bold(x)) = bold(x) / (sum_(i=1)^(d_y) x_i) $ #pause

  In deep learning we often use the *softmax* function. When combined with the classification loss the gradient is linear, making learning faster

==
  The softmax function maps real numbers to the simplex (probabilities)

  $ "softmax": bb(R)^k |-> Delta^(k - 1) $ #pause

  $ "softmax"(vec(x_1, dots.v, x_k)) = (e^(bold(x))) / (sum_(i=1)^k e^(x_i)) = vec(
    e^(x_1) / (e^(x_1) + e^(x_2) + dots e^(x_k)),
    e^(x_2) / (e^(x_1) + e^(x_2) + dots e^(x_k)),
    dots.v,
    e^(x_k) / (e^(x_1) + e^(x_2) + dots e^(x_k)),
  ) $ #pause

  If we attach it to our linear model, we can output probabilities!

  $ f(bold(x), bold(theta)) = "softmax"(bold(theta)^top overline(bold(x))) $

==
  Naturally, we can use the same method with a deep neural network

  $ 
  f_1(bold(x), bold(phi)) = sigma(bold(phi)^top overline(bold(x))) \
  dots.v \

  f_ell (bold(x), bold(xi)) = "softmax"(bold(xi)^top overline(bold(x))) $ #pause

  Now, our neural network can output probabilities

  $ f(bold(x), bold(theta)) = vec(
    P("Shirt" | #image("figures/lecture_5/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_5/shirt.png", height: 10%)),
    dots.v
  ) = vec(0.25, 0.08, dots.v)
  $

==
  *Question:* Why do we output probabilities instead of a binary values

  $ f(bold(x), bold(theta)) = vec(
    P("Shirt" | #image("figures/lecture_5/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_5/shirt.png", height: 10%)),
    dots.v
  ) = vec(0.25, 0.08, dots.v); quad f(bold(x), bold(theta)) = vec(
    1,
    0,
    dots.v
  )
  $ #pause

  *Answer 1:* Outputting probabilities results in differentiable functions #pause
  
  *Answer 2:* We report uncertainty, which is useful in many applications 

==
    #cimage("figures/lecture_5/fashion_mnist_probs.png", height: 80%)

= Classification Loss

==
  We consider the label $bold(y)_[i]$ as a conditional distribution

  $ P(bold(y)_[i] | bold(x)_[i]) = vec(
    P("Shirt" | #image("figures/lecture_5/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_5/shirt.png", height: 10%))
  ) = vec(1, 0) $ #pause

  Our neural network also outputs some distribution

  $ f(bold(x)_[i], bold(theta)) = vec(
    P("Shirt" | #image("figures/lecture_5/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_5/shirt.png", height: 10%))
  ) = vec(0.6, 0.4) $ #pause

  What loss function should we use for classification?

==
  $ f(bold(x)_[i], bold(theta)) = vec(0.6, 0.4); quad bold(y)_[i] = vec(1, 0) $ #pause

  We could use the square error like linear regression #pause

  $ (0.6 - 1)^2 + (0.4 - 0)^2 $ #pause

  This can work, but in reality it does not work well #pause

  Distributions have special properties (positive, sum to one) that square error does not consider #pause

  Instead, we use the *cross-entropy loss* #pause

  Let us derive it

==
  We can model $f(bold(x), bold(theta))$ and $bold(y)$ as probability distributions #pause

  Loss should measure the error between distributions $f(bold(x), bold(theta))$ and $bold(y)$ #pause

  How do we measure the error between probability distributions? #pause

  We use the *Kullback-Leibler Divergence (KL)* #pause

  #cimage("figures/lecture_5/forwardkl.png", height: 50%)

==
  #cimage("figures/lecture_5/forwardkl.png", height: 50%)
  
  $ "KL"(P, Q) = sum_i P(i) log P(i) / Q(i) $

==
  First, write down KL-divergence

  $ "KL"(P, Q) = sum_i P(i) log P(i) / Q(i) $  #pause

  Plug in our two distributions $P = bold(y)$ and $Q = f$ 

 $ "KL"(P(bold(y) | bold(x)), f(bold(x), bold(theta))) = sum_(i=1)^(d_y) P(y_i | bold(x)) log P(y_i | bold(x)) / f(bold(x), bold(theta))_i $

==
 $ "KL"(P(bold(y) | bold(x)), f(bold(x), bold(theta))) = sum_(i=1)^(d_y) P(y_i | bold(x)) log P(y_i | bold(x)) / f(bold(x), bold(theta))_i $ #pause

 Rewrite the logarithm using the sum rule of logarithms

 $ "KL"(P(bold(y) | bold(x)), f(bold(x), bold(theta))) = sum_(i=1)^(d_y) P(y_i | bold(x)) (log P(y_i | bold(x)) - log f(bold(x), bold(theta))_i ) $ 

==
 $ "KL"(P(bold(y) | bold(x)), f(bold(x), bold(theta))) = sum_(i=1)^(d_y) P(y_i | bold(x)) (log P(y_i | bold(x)) - log f(bold(x), bold(theta))_i ) $ #pause

 Split the sum into two parts

 $ = sum_(i=1)^(d_y) P(y_i | bold(x)) log P(y_i | bold(x)) - sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i $

==
 $ = sum_(i=1)^(d_y) P(y_i | bold(x)) log P(y_i | bold(x)) - sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i $ #pause

 The first term is constant, and we will minimize the loss. So $argmin_bold(theta) cal(L) + k = argmin_bold(theta) cal(L)$. Therefore, we can ignore the first term.

 $ = - sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i $ #pause

  This is the loss for a classification task! We call this the *cross-entropy* loss function

// 114:00

==
  *Example:*  #pause

  $ cal(L)(bold(x), bold(y), bold(theta))= - sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i; quad

  f(bold(x), bold(theta)) = vec(0.6, 0.4); quad bold(y) = vec(1, 0) $ #pause

  $ = - [ 
    P(y_1 | bold(x)) log f(bold(x), bold(theta))_1
    + P(y_2 | bold(x)) log f(bold(x), bold(theta))_2
  ] $ #pause

  $ = - [ 
    1 dot log 0.6 +
    0 dot log 0.4 
  ] $ #pause

  $ = - [ -0.22 ] $ #pause

  $ = 0.22 $

==
  $ cal(L)(bold(x), bold(y), bold(theta)) = - sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i $

  By minimizing the loss, we make $f(bold(x), bold(theta)) =  P(bold(y) | bold(x))$ #pause

  $ argmin_theta cal(L)(bold(x), bold(y), bold(theta)) = argmin_theta [- sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i ] $ #pause

  $ f(bold(x), bold(theta)) = P(bold(y) | bold(x)) =
  P(vec("boot", "dress", dots.v) mid(|) #image("figures/lecture_5/shirt.png", height: 20%)) $ 

==
  Our loss was just for a single image #pause
  
  $ cal(L)(bold(x), bold(y), bold(theta)) = [- sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i ] $ #pause

  Find $bold(theta)$ that minimize the loss over the whole dataset #pause

  $ cal(L)(bold(x), bold(y), bold(theta)) = [- sum_(j=1)^n sum_(i=1)^(d_y) P(y_([j], i) | bold(x)_[j]) log f(bold(x)_[j], bold(theta))_i ] $ 

= Classification Optimization
==
  Find $bold(theta)$ just like before, using gradient descent #pause

  The gradients are the same as before except the last layer #pause

  I will not derive any more gradients, but the softmax gradient nearly identical to the sigmoid function gradient #pause

  $ (gradient_bold(theta) "softmax")(bold(z)) = "softmax"(bold(z)) dot.o (1 - "softmax"(bold(z))) $ #pause

  Softmax is a multi-class generalization of the sigmoid function

// 120:00
= Coding
==
  You have everything you need to solve deep learning tasks!
  + Regression #pause
  + Classification #pause

  Every interesting task (chatbot, self driving car, etc): #pause
  + Construct a deep neural network #pause
  + Compute regression or classification loss #pause
  + Optimize with gradient descent #pause

  The rest of this course will examine neural network architectures 

==
  https://colab.research.google.com/drive/1BGMIE2CjlLJOH-D2r9AariPDVgxjWlqG?usp=sharing

==
  Homework 3 is released, you have two weeks to complete it #pause

  https://colab.research.google.com/drive/1LainS20p6c3YVRFM4XgHRBODh6dvAaT2?usp=sharing#scrollTo=q8pJST5xFt-p 