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
  // Section time: 34 mins at leisurely pace
  title: [Classification],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
  //logo: image("logo.jpg", width: 25%)
)

#slide(title: [ML])[
  Many problems in ML can be reduced to *regression* or *classification* #pause

  *Regression* asks how many #pause
  - How much money will I make? #pause
  - How much rain will there be tomorrow? #pause
  - How far away is this object? #pause

  *Classification* asks which one #pause
  - Is this a dog or muffin? #pause
  - Will it rain tomorrow? Yes or no? #pause
  - What color is this object? #pause

  Now let us look at classification
]

#slide(title: [Classification])[
  + Define an example problem #pause
  + Primer on probability #pause
  + Define our machine learning model $f$ #pause
  + Define a loss function $cal(L)$ #pause
  + Use $cal(L)$ to learn the parameters $theta$ of $f$ 

  Does this look familiar?
]

#slide(title: [Classification])[
  + *Define an example problem*
  + Primer on probability 
  + Define our machine learning model $f$ 
  + Define a loss function $cal(L)$ 
  + Use $cal(L)$ to learn the parameters $theta$ of $f$ 
]

#slide[
  *Task:* Given an pictures of clothes, predict their text descriptions #pause

  $X: #image("figures/lecture_4/classify_input.svg", width: 80%)$ #pause

  $Y : & {"T-shirt, Trouser, Pullover, Dress, Coat,"\ 
    & "Sandal, Shirt, Sneaker, Bag, Ankle boot"}$

  *Approach:* Learn the parameters $theta$ that produce *class probabilities*  #pause

  $ f(x, theta) = P(y | x) = P("boot" | #image("figures/lecture_4/shirt.png", height: 20%)) $ #pause
]

#slide(title: [Classification])[
  + *Define an example problem*
  + Primer on probability 
  + Define our machine learning model $f$ 
  + Define a loss function $cal(L)$ 
  + Use $cal(L)$ to learn the parameters $theta$ of $f$ 
]

#slide(title: [Classification])[
  + Define an example problem
  + *Primer on probability*
  + Define our machine learning model $f$ 
  + Define a loss function $cal(L)$ 
  + Use $cal(L)$ to learn the parameters $theta$ of $f$ 
]

#slide[
  In probability, we have *experiments* and *outcomes* #pause

  An experiment yields one of many possible outcomes #pause

  #side-by-side[Flip a coin][Heads] #pause
  #side-by-side[Walk outside][Rain] #pause
  #side-by-side[Grab clothing from closest][Coat] #pause
  

]

#slide[
  The *sample space* $S$ defines all possible outcomes for an experiment #pause

  #side-by-side[Flip a coin][$ S = {"heads", "tails"} $] #pause

  #side-by-side[Walk outside][$ S = {"rain", "sun", "wind", "cloud"} $] #pause

  #side-by-side[Grab clothing from closet][$ S = {"T-shirt", "Trouser", "Pullover", "Dress", \ 
  "Coat", "Sandal", "Shirt", "Sneaker", "Bag", \
  "Ankle boot"} $]
]

#slide[
  An *event* is a subset of the sample space #pause

  #side-by-side[Flip a coin][$ {"heads"} $] #pause

  #side-by-side[Walk outside][$ {"rain", "cloud", "wind"} $] #pause

  #side-by-side[Grab clothing from closet][$ {"Sneaker"} $]
]

#slide[
  The *probability* measures how likely an event is to occur #pause

  The probability must be between 0 (never occurs) and 1 (always occurs) #pause

  $ 0 <= P(A) <= 1; quad forall A in S $ #pause

  The probabilities must sum to one #pause

  $ sum_(A in S) P(A) = 1 $
]

#slide[
  #side-by-side[Flip a coin][$ P("Heads")  = 1 / 2 $] #pause

  #side-by-side[Walk outside][$ P("Rain") = 0.05 $] #pause

  #side-by-side[Grab clothing from closet][$ P("Dress") = 0 $] #pause

]

#slide[
  For *mutually exclusive* events, we can sum together probabilities #pause

  $ P(A union B) = P(A) + P(B) $ #pause

  #v(1em)

  #side-by-side[Grab clothing from closet][
    $ P("Shirt") = 0.1, P("Bag") = 0.05$
    $ P("Shirt" union "Bag") = 0.15 $
  ]

  Be careful! #pause

  #side-by-side[Walk outside][
    $P("Rain") = 0.05, P("Sun") = 0.4$
    $P("Rain" union "Sun") != 0.45$
  ]
]

#slide[
  If events are *independent*, we can multiply their probabilities #pause

  $ P(A sect B) = P(A) dot P(B) $ #pause

  Be careful! #pause

  #side-by-side[Flip a coin][
    $P("Heads")  = 0.5, P("Tails")=0.5$
    $P("Heads" sect "Tails") != 0.25$
  ] #pause
]

#slide[
  Events can be *conditionally dependent* 
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

#slide[TODO: Random variable, distribution]

#slide(title: [Classification])[
  + Define an example problem
  + *Primer on probability*
  + Define our machine learning model $f$ 
  + Define a loss function $cal(L)$ 
  + Use $cal(L)$ to learn the parameters $theta$ of $f$ 
]

#focus-slide[Relax]

#slide[
  Back to the problem...
]

#slide[
  *Task:* Given a picture of clothes, predict the text description #pause

  $X: #image("figures/lecture_4/classify_input.svg", width: 80%)$ #pause

  $Y : & {"T-shirt, Trouser, Pullover, Dress, Coat,"\ 
    & "Sandal, Shirt, Sneaker, Bag, Ankle boot"}$

  *Approach:* Learn the parameters $theta$ that produce *event probabilities*  #pause

  $ f(x, theta) = P(y | x) = P("boot" | #image("figures/lecture_4/shirt.png", height: 20%)) $ #pause
]

#slide(title: [Classification])[
  + Define an example problem
  + Primer on probability
  + *Define our machine learning model $f$*
  + Define a loss function $cal(L)$ 
  + Use $cal(L)$ to learn the parameters $theta$ of $f$ 
]

#slide[
  We will again start with a linear model #pause
  
  $ f(x, bold(theta)) = f(x, vec(W, b)) = W x + b $ #pause

  However, the probabilities must sum to one which is nonlinear... #pause

  We introduce the *softmax* operator to ensure all probabilities sum to 1 #pause
]

#slide[
  The softmax operator is heavily used in machine learning, especially where probabilities pop up #pause

  It maps a vector of real numbers to a vector of probabilities #pause

  $ "softmax": bb(R)^n |-> Delta^(n-1) $ #pause

  $ Delta^(n-1) = { vec(p_1, p_2, dots.v, p_n) mid(|)  sum_(i=1)^n p_i = 1 } $

  The simplex operator $Delta$ just means that the outputs of softmax sum to 1 #pause
]

#slide[
  $ "softmax"(vec(x_1, dots.v, x_n)) = (e^(x_i)) / (sum_(j=1)^n e^(x_j)) = vec(
    e^(x_1) / (e^(x_1) + e^(x_2) + dots e^(x_n)),
    e^(x_2) / (e^(x_1) + e^(x_2) + dots e^(x_n)),
    dots.v,
    e^(x_n) / (e^(x_1) + e^(x_2) + dots e^(x_n)),
  ) $
]

#slide[
  Using the softmax function, we learn the probability for each class/event

  $ f(bold(x), bold(theta)): bb(Z)^n |->  Delta^(|Y| - 1) $

  $ f(x, bold(theta)) = f(x, vec(W, b)) = op("softmax")(W x + b) $ #pause

  Each output dimension determines a specific class/event probability

  $ f(x, bold(theta)) = vec(
    P("Ankle boot" | #image("figures/lecture_4/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_4/shirt.png", height: 10%)),
    dots.v
  )
  $
]

#slide[
    #cimage("figures/lecture_4/fashion_mnist_probs.png", height: 80%)
]

#slide[
  *Question:* Why do we output probabilities instead of just a one-hot vector

  $ f(bold(x), bold(theta)) = vec(
    P("Shirt" | #image("figures/lecture_4/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_4/shirt.png", height: 10%)),
  )
  $

  $ f(bold(x), bold(theta)) = vec(
    1,
    0
  )
  $

  *Answer:* We do not always know the correct answer. There is always uncertainty.
]

#focus-slide[Relax]

#slide(title: [Classification])[
  + Define an example problem
  + Primer on probability
  + *Define our machine learning model $f$*
  + Define a loss function $cal(L)$ 
  + Use $cal(L)$ to learn the parameters $theta$ of $f$ 
]

#slide(title: [Classification])[
  + Define an example problem
  + Primer on probability
  + Define our machine learning model $f$
  + *Define a loss function $cal(L)$* 
  + Use $cal(L)$ to learn the parameters $theta$ of $f$ 
]

#slide[
  We use squared error for regression, what about classification? #pause

  $ f(bold(x)_i, bold(theta)) = vec(
    P("Shirt" | #image("figures/lecture_4/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_4/shirt.png", height: 10%))
  ) = vec(0.6, 0.4) $ #pause

  $ bold(y)_i = vec(
    P("Shirt" | #image("figures/lecture_4/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_4/shirt.png", height: 10%))
  ) = vec(1, 0) $ #pause

  //In practice, square error works poorly for classification #pause

  //Does not take into account the nature of probabilities
  //We want to minimize the difference between $f(bold(x)_i, bold(theta))$ and $bold(y)_i$
]

#slide[
  $ f(bold(x)_i, bold(theta)) = vec(0.6, 0.4), bold(y)_i = vec(1, 0) $ #pause

  We could compute the sum of square errors #pause

  $ (0.6 - 1)^2 + (0.4 - 0)^2 $ #pause

  In practice, this does not work very well #pause

  Instead, we use the *cross-entropy loss* #pause

  Let us derive it
]

#slide[
  We can model $f(bold(x), bold(theta))$ and $bold(y)$ as probability distributions #pause

  How do we measure the difference between probability distributions? #pause

  We use the *Kullback-Leibler Divergence (KL)* #pause

  #cimage("figures/lecture_4/forwardkl.png", height: 50%)
]

#slide[

  #cimage("figures/lecture_4/forwardkl.png", height: 50%)
  
  $ "KL"(P, Q) = sum_i P(i) log P(i) / Q(i) $
]

#slide[

  #text(size: 22pt)[
    TODO: Should be $f(y_i | x, theta)$

  #side-by-side[ $ "KL"(P, Q) = $][$sum_i P(i) log P(i) / Q(i) $][KL divergence] #pause

  #side-by-side[$ "KL"(P(bold(y) | bold(x)), f(bold(x), bold(theta)))= $][
    $ sum_(y in Y) P(y | bold(x)) log P(y | bold(x)) / f(bold(x), bold(theta)) $][Plug in $f, y$] #pause

  #side-by-side(columns: (auto, auto, auto))[$ "KL"(P(bold(y) | bold(x)), f(bold(x), bold(theta)))= $][
    $ sum_(y in Y) P(y | bold(x)) [log P(y | bold(x)) - log f(bold(x), bold(theta)) ] $][Log rule] #pause

  
  #side-by-side(columns: (4em, auto, auto))[$ = $][
    $ sum_(y in Y) P(y | bold(x)) log P(y | bold(x)) - sum_(y in Y) P(y | bold(x)) log f(bold(x), bold(theta)) $][Split sum] #pause

  #side-by-side[$ = $][
    $ - sum_(y in Y) P(y | bold(x)) log f(bold(x), bold(theta)) $][First term constant] #pause

    #align(center)[This is the cross-entropy loss!]
  
]]

#slide[
  $ cal(L)(bold(x), bold(y), bold(theta)) = - sum_(y in Y) P(y | bold(x)) log f(bold(x), bold(theta)) $ #pause

  By minimizing the loss, we make $f(bold(x), bold(theta))$ output the same probability distribution as $bold(y)$ #pause

  $ min_theta cal(L)(bold(x), bold(y), bold(theta)) = min_theta [- sum_(y in Y) P(y | bold(x)) log f(bold(x), bold(theta)) ] $ #pause

  $ f(bold(x), bold(theta)) = P(bold(y) mid(|) bold(x)) =
    P(vec("boot", "dress", dots.v) mid(|) #image("figures/lecture_4/shirt.png", height: 20%)) $ #pause
]

#slide[
  Our loss was just for a single image #pause
  
  $ min_theta cal(L)(bold(x), bold(y), bold(theta)) = min_theta [- sum_(y in Y) P(y | bold(x)) log f(bold(x), bold(theta)) ] $ #pause

  Find the parameters over all images #pause

  $ min_theta cal(L)(bold(x), bold(y), bold(theta)) = min_theta [- sum_(i = 1)^n sum_(y_i in Y) P(y_i | bold(x)_i) log f(bold(x)_i, bold(theta)) ] $ #pause
]

#slide(title: [Classification])[
  + Define an example problem
  + Primer on probability
  + Define our machine learning model $f$
  + *Define a loss function $cal(L)$*
  + Use $cal(L)$ to learn the parameters $theta$ of $f$ 
]

#focus-slide[Relax]

#slide(title: [Classification])[
  + Define an example problem
  + Primer on probability
  + Define our machine learning model $f$
  + Define a loss function $cal(L)$ 
  + *Use $cal(L)$ to learn the parameters $theta$ of $f$*
]

#slide[
  Unlike linear regression, we use a softmax to model probabilities #pause

  $ f(x, bold(theta)) = f(x, vec(W, b)) = op("softmax")(W x + b) $ #pause

  There is no closed-form solution! #pause 
  
  We need to use iterative solvers to find theta #pause

  We will come back to this when discussing neural networks

]
