#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

#set math.vec(delim: "[")
#set math.mat(delim: "[")
#let agenda(index: none) = {
  let ag = (
    [Review],
    [Torch optimization coding],
    [Classification task],
    [Probability review],
    [Define model $f$],
    [Define loss function $cal(L)$],
    [Find $bold(theta)$ that minimize $cal(L)$],
    //[Parameter initialization],
    //[Regularization],
    //[Residual networks],
    //[Adaptive optimization],
    //[Activation functions],
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

// classification
// other activation functions
// weight regularization and dropout
// weight initialization
// adam

#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 5: Classification"
)

#title-slide(
  // Section time: 34 mins at leisurely pace
  title: [Classification],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
  //logo: image("logo.jpg", width: 25%)
)

#slide(title: [Admin])[
  We will have a make-up lecture later on for the missed lecture #pause
  
  Assignment 1 grades were released on moodle #pause

  The scores were very good, with a mean of approximately 90/100 #pause

  I am still grading quiz 2, but I had a look at the responses to question 4
]

#slide(title: [Admin])[
  Some requests from students: #pause

  + More coding, less theory #pause
  + More math/theory #pause
  + Too easy, go faster #pause
  + Speak slower #pause
  + Course is too hard #pause
  + Course is perfect #pause
  + Move captions to top of screen #pause
  + Upload powerpoint before lecture #pause

  There are conflicting student needs
]
//11:30

#slide[#agenda(index: none)]
#slide[#agenda(index: 0)]

#slide(title: [Review])[
  Last time, we reviewed derivatives #pause

  $ f'(x) = d / (d x) f = (d f) / (d x) = lim_(h -> 0) (f(x + h) - f(x)) / h $ #pause

  and gradients #pause
  
  $ gradient_(bold(x)) f(mat(x_1, x_2, dots, x_n)^top) = mat((partial f) / (partial x_1), (partial f) / (partial x_2), dots, (partial f) / (partial x_n))^top $
]

#slide(title: [Review])[
  Gradients are important in deep learning for two reasons: #pause

  $ bold("Reason 1:") f(bold(x)) "has critical points at" gradient_bold(x) f(bold(x)) = 0 $ #pause

  #cimage("figures/lecture_5/saddle.png") #pause

  With optimization, we attempt to find minima of loss functions
]

#slide(title: [Review])[
  Gradients are important in deep learning for two reasons: #pause

  *Reason 2:* For problems without analytical solutions, the gradient (slope) is necessary for gradient descent #pause

  #cimage("figures/lecture_4/gradient_descent_3d.png", height: 60%)
]

#slide(title: [Review])[ 
  First, we derived the solution to linear regression #pause

  $ cal(L)(bold(X), bold(Y), bold(theta)) = sum_(i=1)^n ( f(bold(x)_[i], bold(theta)) - bold(y)_[i] )^2 $ #pause

  $ cal(L)(bold(X), bold(Y), bold(theta)) = ( bold(Y) - bold(X)_D bold(theta) )^top ( bold(Y) - bold(X)_D bold(theta) ) $  #pause

  $ cal(L)(bold(X), bold(Y), bold(theta)) = 
  underbrace(underbrace(( bold(Y) - bold(X)_D bold(theta) )^top, "Linear function of " theta quad)
  underbrace(( bold(Y) - bold(X)_D bold(theta) ), "Linear function of " theta), "Quadratic function of " theta) $
]

#slide(title: [Review])[ 
  #side-by-side[A quadratic function has a single critical point, which must be a global minimum][#cimage("figures/lecture_4/quadratic_parameter_space.png", height: 100%)]
]

#slide(title: [Review])[ 
  We found the analytical solution for linear regression by finding where the gradient was zero and solving for $bold(theta)$ #pause
  
  $ gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = 0 $ #pause

  $ bold(theta) = (bold(X)_D^top bold(X)_D)^(-1) bold(X)_D^top bold(Y) $ #pause

  Which solves
  $ argmin_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) $ 
]


#slide(title: [Review])[ 
  For neural networks, the square error loss is no longer quadratic #pause

  #side-by-side[$ cal(L)(x, y, bold(theta)) = (f(x, bold(theta)) - y)^2 $][Loss function] #pause
  #side-by-side[$ f(x, bold(theta)) = sigma(theta_0 + theta_1 x) $][Neural network model] #pause

  Now, we plug the model $f$ into the loss function #pause 

  $ cal(L)(x, y, bold(theta)) = (sigma(theta_0 + theta_1 x) - y)^2 $ #pause

  $ cal(L)(x, y, bold(theta)) = 
  underbrace((sigma(theta_0 + theta_1 x) - y), "Nonlinear function of" theta) 
  underbrace((sigma(theta_0 + theta_1 x) - y), "Nonlinear function of" theta) $ #pause

  There is no analytical solution for $bold(theta)$
]  

#slide(title: [Review])[ 
  Instead, we found the parameters of a neural network through gradient descent #pause

  Gradient descent is an optimization method for differentiable functions #pause

  We went over both the intuition and mathematical definitions
]

#slide(title: [Review])[ 
  #side-by-side[
    #cimage("figures/lecture_4/lightning.jpg", height: 100%) #pause
   ][
    #cimage("figures/lecture_4/hiking_slope.jpg", height: 100%)
   ] 
]

#slide(title: [Review])[ 
  #cimage("figures/lecture_4/parameter_space.png", height: 100%)
]

#slide(title: [Review])[ 
  The gradient descent algorithm: 
  #algorithm({
    import algorithmic: *

    Function("Gradient Descent", args: ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$), {

      Cmt[Randomly initialize parameters]
      Assign[$bold(theta)$][$cal(N)(0, 1)$] 

      For(cond: $i in 1 dots t$, {
        Cmt[Compute the gradient of the loss]        
        Assign[$bold(J)$][$gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta))$]
        Cmt[Update the parameters using the negative gradient]
        Assign[$bold(theta)$][$bold(theta) - alpha bold(J)$]
      })

    Return[$bold(theta)$]
    })
  })
]


#slide(title: [Review])[ 
  We derived the $gradient_bold(theta) cal(L)$ for deep neural networks using the chain rule #pause

  $ gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = sum_(i = 1)^n 2 ( f(bold(x)_[i], bold(theta)) - bold(y)_[i]) #redm[$gradient_bold(theta) f(bold(x)_[i], bold(theta))$] 
  $ #pause
   

  $ 
  #redm[$gradient_(bold(theta)) f(bold(x), bold(theta))$] = gradient_(bold(phi), bold(psi), dots, bold(xi)) f(bold(x), mat(bold(phi), bold(psi), dots, bold(xi))^top) = vec(
    gradient_bold(phi) f_1(bold(x), bold(phi)), 
    gradient_(bold(psi)) f_2(bold(z)_1, bold(psi)), 
    dots.v, 
    #redm[$gradient_(bold(xi)) f_(ell)(bold(z)_(ell - 1), bold(xi))$]
  ) 
  $ #pause

  $ #redm[$gradient_(bold(xi)) f_ell (bold(z)_(ell - 1), bold(xi))$] = (sigma(bold(xi)^top overline(bold(z))_(ell - 1)) dot.circle (1 - sigma(bold(xi)^top overline(bold(z))_(ell - 1)))) overline(bold(z))_(ell - 1)^top $ 
]


#slide(title: [Review])[
  We ran into issues computing the gradient of a layer because of the Heaviside step function #pause

  We replaced it with a differentiable (soft) approximation called the sigmoid function #pause

  #side-by-side[#cimage("figures/lecture_4/heaviside.svg")][#cimage("figures/lecture_4/sigmoid.svg")][ $ sigma(z) = 1 / (1 + e^(-z)) $]
]


#slide(title: [Review])[
  In `jax`, we compute the gradient using the `jax.grad` function #pause

  ```python
  import jax

  def L(theta, X, Y):
    ...

  # Create a new function that is the gradient of L
  # Then compute gradient of L for given inputs
  J = jax.grad(L)(X, Y, theta)
  # Update parameters
  alpha = 0.0001
  theta = theta - alpha * J
  ```
]


#slide(title: [Review])[ 
  In `torch`, we backpropagate through a graph of operations
  
  ```python
  import torch
  optimizer = torch.optim.SGD(lr=0.0001)

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
]

#slide[#agenda(index: 0)]
#slide[#agenda(index: 1)]

// 30:00

#slide[
  First, a video of one application of gradient descent 

  https://youtu.be/kGDO2e_qiyI?si=ZopZKy-6WQ4B0csX #pause

  #v(1em) 
  Time for some interactive coding

  https://colab.research.google.com/drive/1W8WVZ8n_9yJCcOqkPVURp_wJUx3EQc5w
]

// ~60:00

#slide[#agenda(index: 1)]
#slide[#agenda(index: 2)]

#slide(title: [Classification])[
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
]

#slide[
  *Task:* Given a picture of clothes, predict the text description #pause

  $X: bb(Z)_(0,255)^(32 times 32) #image("figures/lecture_5/classify_input.svg", width: 80%)$ #pause

  $Y : & {"T-shirt, Trouser, Pullover, Dress, Coat,"\ 
    & "Sandal, Shirt, Sneaker, Bag, Ankle boot"}$ #pause

  *Approach:* Learn $bold(theta)$ that produce *conditional probabilities*  #pause

  $ f(bold(x), bold(theta)) = P(bold(y) | bold(x)) = P(vec("T-Shirt", "Trouser", dots.v) mid(|) #image("figures/lecture_5/shirt.png", height: 20%)) = vec(0.2, 0.01, dots.v) $
]

#slide(title: [Agenda])[#agenda(index: 2)]
#slide(title: [Agenda])[#agenda(index: 3)]


#slide[
  Classification tasks require *probability theory*, so let us review #pause

  In probability, we have *experiments* and *outcomes* #pause

  An experiment yields one of many possible outcomes #pause

  #side-by-side[Experiment][Outcome] #pause
  #side-by-side[Flip a coin][Heads] #pause
  #side-by-side[Walk outside][Rain] #pause
  #side-by-side[Grab clothing from closest][Coat] #pause
  

]

#slide[
  The *sample space* $S$ defines all possible outcomes for an experiment #pause

  #side-by-side[Experiment][Sample Space $S$] #pause
  #side-by-side[Flip a coin][$ S = {"heads", "tails"} $] #pause

  #side-by-side[Walk outside][$ S = {"rain", "sun", "wind", "cloud"} $] #pause

  #side-by-side[Take clothing from closet][$ S = {"T-shirt", "Trouser", "Pullover", "Dress", \ 
  "Coat", "Sandal", "Shirt", "Sneaker", "Bag", \
  "Ankle boot"} $]
]

#slide[
  The *event space* $E$ is a specific subset of the sample space #pause

  #side-by-side[Experiment][Sample Space][Event]

  #side-by-side[Flip a coin][$ S = {"heads", "tails"} $][$ E = {"heads"} $] #pause

  #side-by-side[Walk outside][$ S = {"rain", "sun", "wind", "cloud"} $][$ E = {"rain", "wind"} $] #pause

  #side-by-side[Take from closet][$ {
    "T-shirt", "Trouser", \ 
    "Pullover",  "Dress", \
  "Coat", "Sandal", "Shirt", \ 
  "Sneaker", "Bag", "Ankle boot"} $][$ E = {"Shirt", "T-Shirt", "Coat"} $]
]

#slide[
  The *probability* measures how likely an event is to occur #pause

  The probability must be between 0 (never occurs) and 1 (always occurs) #pause

  $ 0 <= P(A) <= 1; quad forall A in S $ #pause
  
  #side-by-side[Experiment][Probabilities] #pause

  #side-by-side[Flip a coin][$ P("heads") = 0.5 $] #pause

  #side-by-side[Walk outside][$ P("rain") = 0.15 $] #pause

  #side-by-side[Take from closet][$ P("Shirt") =  0.1 $]

]

#slide[
  When we define $P$ as a function, we call it a *distribution* #pause

  $ P: E |-> (0, 1) $ #pause

  The probabilities (distribution) must sum to one #pause

  $ sum_(x in E) P(x) = 1 $  #pause

  #side-by-side[Flip a coin][$ {P("heads") = 0.5, P("tails") = 0.5} $] #pause

  //#side-by-side[Walk outside][$ {P("rain") = 0.15, P("sun") = 0.5, P("sun_and_rain") = 0.05, \ P("wind") = 0.1, dots } $] #pause

  #side-by-side[Take clothing from closet][$ {P("T-shirt") = 0.1, P("Trouser") = 0.08, \ P("Pullover") = 0.12, dots } $]
]

#slide[
  The distribution is a function, so we can plot it

  #cimage("figures/lecture_5/pmf.jpg", width: 100%)
]

#slide[
  Events can overlap with each other #pause
  - Disjoint events #pause
  - Conditionally dependent events
]

#slide[
  Two events $A, B$ are *disjoint* if either $A$ or $B$ occurs #pause
  
  With disjoint events, $P(A sect B) = 0$ #pause

  #side-by-side[Flip a coin][
    $P("Heads") = 0.5, P("Tails") = 0.5$
    $P("Heads" sect "Tails") = 0$
  ] #pause

  Be careful!

  #side-by-side[Walk outside][
    $P("Rain") = 0.05, P("Sun") = 0.4$
    $P("Rain" sect "Sun") != 0$
  ]
]

/*
#slide[
  Two events $A, B$ are independent events if the occurence of $A$ does not affect the occurence of $B$ #pause

  With independent events, $P(A union B) = P(A) dot P(B)$ #pause

  #side-by-side[Flip two coins][
    $ P("C1 Heads") = 0.5, P("C2 Tails") = 0.5 $
    $ P("Heads" union "Tails") = 0.25 dot 0.25 = 0.125 $
  ] 
]
*/

#slide[
  Events $A$ is *conditionally dependent* on $B$ if $B$ occuring tells us about the probability of $A$ #pause

  $ P("cloud") = 0.2, P("rain") = 0.05 $ #pause

  $ P("rain" | "cloud") = 0.7 $ #pause

  $ P(A | B) = P(A sect B) / P(B) $ #pause

  /*
  #side-by-side[Flip a coin][
    $P("Heads" sect "Tails") = 0 \ 
    P("Tails")=0.5$
    $P("Heads" | "Tails") = 0 / 0.5 = 0$
  ] #pause
  */

  #side-by-side[Walk outside][
    $P("Rain" sect "Cloud") = 0.2 \ 
    P("Cloud") = 0.4$
    $P("Rain" | "Cloud") = 0.2 / 0.4 = 0.5$
  ]
]

/*
#slide[
  For *mutually exclusive* events, we can sum together probabilities #pause

  $ P(A union B) = P(A) + P(B) $ #pause

  #v(1em)

  #side-by-side[Take clothing from closet][
    $ P("Shirt") = 0.1, P("Bag") = 0.05$
    $ P("Shirt" union "Bag") = 0.15 $
  ] #pause

  Be careful! Events must be mutually exclusive #pause

  #side-by-side[Walk outside][
    $P("Rain") = 0.05, P("Sun") = 0.4$
    $P("Rain" union "Sun") != 0.45$
  ]
]



#slide[
  If events are not mutually exclusive, but they are *independent*, then we can multiply their probabilities #pause

  $ P(A sect B) = P(A) dot P(B) $ #pause

  Be careful! Not all events are independent #pause

  #side-by-side[Flip a coin][
    $P("Heads") = 0.5, P("Tails")=0.5$
    $P("Heads" sect "Tails") != 0.25$
  ] #pause

  Such events are *dependent*
]

#slide[
  Events that are not independent are *conditionally dependent* 
  $ P(A | B) = P(A sect B) / P(B) $ #pause

  #side-by-side[Flip a coin][
    $P("Heads" sect "Tails") = 0 \ 
    P("Tails")=0.5$
    $P("Heads" | "Tails") = 0 / 0.5 = 0$
  ] #pause

  #v(1em)

  #side-by-side[Walk outside][
    $P("Rain" sect "Cloud") = 0.2 \ 
    P("Cloud") = 0.4$
    $P("Rain" | "Cloud") = 0.2 / 0.4 = 0.5$
  ]
]
*/

#slide[
  *Task:* Given a picture of clothes, predict the text description #pause

  $X: bb(Z)_(0,255)^(32 times 32) #image("figures/lecture_5/classify_input.svg", width: 80%)$ #pause

  $Y : & {"T-shirt, Trouser, Pullover, Dress, Coat,"\ 
    & "Sandal, Shirt, Sneaker, Bag, Ankle boot"}$ #pause

  *Approach:* Learn $bold(theta)$ that produce *conditional probabilities*  #pause

  $ f(bold(x), bold(theta)) = P(bold(y) | bold(x)) = P(vec("T-Shirt", "Trouser", dots.v) mid(|) #image("figures/lecture_5/shirt.png", height: 20%)) = vec(0.2, 0.01, dots.v) $
]

#slide(title: [Agenda])[#agenda(index: 3)]
#slide(title: [Agenda])[#agenda(index: 4)]

#slide[
  We will again start with a multivariate linear model #pause
  
  $ f(bold(x), bold(theta)) = bold(theta)^top bold(x) $ #pause

  We want our model to predict the probability of each item class #pause

  *Question:* What is the function signature of $f$? #pause

  *Answer:* $f: bb(R)^(d_x) times Theta |-> bb(R)^(d_y)$ #pause

  *Question:* Can we use this model to predict probabilities? #pause

  *Answer:* No! Because probabilities must sum to one 

  //*Question:* What is the range/co-domain of $f$? #pause

  //*Answer:* $[-oo, oo]$ #pause



  //However, the probabilities must sum to one #pause

  //We introduce the *softmax* operator to ensure all probabilities sum to 1 #pause
]

#slide[
  How can we represent a distribution as a vector? #pause

  $ bold(v) = { vec(v_1, dots.v, v_(d_y)) mid(|) quad sum_(i=1)^(d_y) v_i = 1 } $ #pause

  There is special notation for a vector that sums to one called the *simplex* #pause

  $ Delta^(d_y - 1) $

]

#slide[
  The simplex $Delta^k$ is an $k - 1$-dimensional triangle in $k$-dimensional space #pause

  #cimage("figures/lecture_5/simplex.svg", height: 70%)

  It has only $k - 1$ free variables, because $x_(k) = 1 - sum_(i=1)^(k - 1) x_i$ 


]

#slide[
  So we need a function that maps to the simplex #pause

  $ f: bb(R)^(d_y) times Theta |-> Delta^(d_y - 1) $ #pause

  There are many functions that can do this #pause

  One example is dividing by the $L_1$ norm: 
  
  $ f(bold(x)) = bold(x) / (sum_(i=1)^(d_y) x_i) $ #pause

  In deep learning we often use the *softmax* function. When combined with the classification loss the gradient is linear, making learning faster
]

#slide[
  The softmax function maps real numbers to the simplex (probabilities)

  $ "softmax": bb(R)^k |-> Delta^(k - 1) $ #pause

  $ "softmax"(vec(x_1, dots.v, x_k)) = (e^(bold(x))) / (sum_(i=1)^k e^(x_i)) = vec(
    e^(x_1) / (e^(x_1) + e^(x_2) + dots e^(x_k)),
    e^(x_2) / (e^(x_1) + e^(x_2) + dots e^(x_k)),
    dots.v,
    e^(x_k) / (e^(x_1) + e^(x_2) + dots e^(x_k)),
  ) $

  If we attach it to our linear model, we can output probabilities!

  $ f(bold(x), bold(theta)) = "softmax"(bold(theta)^top bold(x)) $
]

#slide[
  And naturally, we can use the same method for a deep neural network

  $ 
  f_1(bold(x), bold(phi)) = sigma(bold(phi)^top bold(x)) \
  dots.v \

  f_ell (bold(x), bold(xi)) = "softmax"(bold(xi)^top bold(x)) $ #pause

  Now, our neural network can output probabilities

  $ f(bold(x), bold(theta)) = vec(
    P("Ankle boot" | #image("figures/lecture_5/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_5/shirt.png", height: 10%)),
    dots.v
  )
  $
]

#slide[
  *Question:* Why do we output probabilities instead of a binary values

  $ f(bold(x), bold(theta)) = vec(
    P("Shirt" | #image("figures/lecture_5/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_5/shirt.png", height: 10%)),
  )
  $

  $ f(bold(x), bold(theta)) = vec(
    1,
    0
  )
  $

  *Answer 1:* Outputting probabilities results in differentiable functions 
  
  *Answer 2:* We report uncertainty, which is useful in many applications 
]

#slide[
    #cimage("figures/lecture_5/fashion_mnist_probs.png", height: 80%)
]


#slide(title: [Agenda])[#agenda(index: 4)]
#slide(title: [Agenda])[#agenda(index: 5)]

#slide[
  Consider the following prediction and truth

  $ f(bold(x)_[i], bold(theta)) = vec(
    P("Shirt" | #image("figures/lecture_5/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_5/shirt.png", height: 10%))
  ) = vec(0.6, 0.4) $ #pause

  $ bold(y)_[i] = vec(
    P("Shirt" | #image("figures/lecture_5/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_5/shirt.png", height: 10%))
  ) = vec(1, 0) $ #pause

  What loss function should we use for classification?

]

#slide[
  $ f(bold(x)_i, bold(theta)) = vec(0.6, 0.4), bold(y)_i = vec(1, 0) $ #pause

  We could use the square error like linear regression #pause

  $ (0.6 - 1)^2 + (0.4 - 0)^2 $ #pause

  This can work, but in reality it does not work well #pause

  Instead, we use the *cross-entropy loss* #pause

  Let us derive it
]

#slide[
  We can model $f(bold(x), bold(theta))$ and $bold(y)$ as probability distributions #pause

  How do we measure the difference between probability distributions? #pause

  We use the *Kullback-Leibler Divergence (KL)* #pause

  #cimage("figures/lecture_5/forwardkl.png", height: 50%)
]

#slide[

  #cimage("figures/lecture_5/forwardkl.png", height: 50%)
  
  $ "KL"(P, Q) = sum_i P(i) log P(i) / Q(i) $
]

#slide[
  First, write down KL-divergence

  $ "KL"(P, Q) = sum_i P(i) log P(i) / Q(i) $  #pause

  Plug in our two distributions $P = f$ and $Q = bold(y)$ 

 $ "KL"(P(bold(y) | bold(x)), f(bold(x), bold(theta))) = sum_(i=1)^(d_y) P(y_i | bold(x)) log P(y_i | bold(x)) / f(bold(x), bold(theta))_i $
]

#slide[
 $ "KL"(P(bold(y) | bold(x)), f(bold(x), bold(theta))) = sum_(i=1)^(d_y) P(y_i | bold(x)) log P(y_i | bold(x)) / f(bold(x), bold(theta))_i $ #pause

 Rewrite the logarithm using the sum rule of logarithms

 $ "KL"(P(bold(y) | bold(x)), f(bold(x), bold(theta))) = sum_(i=1)^(d_y) P(y_i | bold(x)) (log P(y_i | bold(x)) - log f(bold(x), bold(theta))_i ) $ #pause
]

#slide[
 $ "KL"(P(bold(y) | bold(x)), f(bold(x), bold(theta))) = sum_(i=1)^(d_y) P(y_i | bold(x)) (log P(y_i | bold(x)) - log f(bold(x), bold(theta))_i ) $ #pause

 Split the sum into two parts

 $ = sum_(i=1)^(d_y) P(y_i | bold(x)) log P(y_i | bold(x)) - sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i $

]

#slide[
 $ = sum_(i=1)^(d_y) P(y_i | bold(x)) log P(y_i | bold(x)) - sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i $ #pause

 The first term is constant, and we will minimize the loss. So $argmin_bold(theta) cal(L) + k = argmin_bold(theta) cal(L)$. Therefore, we can ignore the first term.

 $ = - sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i $ #pause

  This is the loss for a classification task! We call this the *cross-entropy* loss function
]

#slide[
  $ cal(L)(bold(x), bold(y), bold(theta)) = - sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i $

  By minimizing the loss, we make $f(bold(x), bold(theta)) =  bold(y)$ #pause

  $ min_theta cal(L)(bold(x), bold(y), bold(theta)) = min_theta [- sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i ] $ #pause

  $ f(bold(x), bold(theta)) = P(bold(y) mid(|) bold(x)) =
  P(vec("boot", "dress", dots.v) mid(|) #image("figures/lecture_5/shirt.png", height: 20%)) $ 
]

#slide[
  Our loss was just for a single image #pause
  
  $ min_theta cal(L)(bold(x), bold(y), bold(theta)) = min_theta [- sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i ] $ #pause

  Find $bold(theta)$ that minimize the loss over the whole dataset #pause

  $ min_theta cal(L)(bold(x), bold(y), bold(theta)) = min_theta [- sum_(j=1)^n sum_(i=1)^(d_y) P(y_([j], i) | bold(x)_[j]) log f(bold(x)_[j], bold(theta))_i ] $ #pause
]

#slide(title: [Agenda])[#agenda(index: 5)]
#slide(title: [Agenda])[#agenda(index: 6)]

#slide(title: [Classification])[
  Find $bold(theta)$ just like before, using gradient descent #pause

  The gradients are the same as before except the last layer #pause

  I will not derive any more gradients, but the softmax gradient nearly identical to the sigmoid function #pause

  $ gradient_bold(theta) "softmax"(bold(z)) = "softmax"(bold(z)) dot.circle (1 - "softmax"(bold(z))) $ #pause

  This is because softmax is a multi-class generalization of the sigmoid function
]

#slide(title: [Agenda])[#agenda(index: 6)]
#slide(title: [Agenda])[#agenda(index: 7)]

#slide(title: [Classification])[
  You have everything you need to solve any deep learning task!
  + Regression #pause
  + Classification #pause

  Every interesting task (chatbot, self driving car, etc): #pause
  + Train a deep neural network #pause
  + Using a regression or classification loss function #pause
  + That we optimize with gradient descent #pause

  The rest of this course will examine neural network architectures 
]

#slide(title: [Classification])[
]