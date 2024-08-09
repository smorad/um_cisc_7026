#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.1.2": canvas, draw

// TODO handle x_i, y_i in the loss function, should be multiple x and y
// Rename design matrix to D instead of X

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
  title: [Regression],
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

  Let us start with regression
]

#slide(title: [Linear Regression])[
  + Define an example problem #pause
  + Define our machine learning model $f$ #pause
  + Define a loss function $cal(L)$ #pause
  + Use $cal(L)$ to learn the parameters $theta$ of $f$ #pause
  + Investigate the results
]

#slide(title: [Linear Regression])[
  + *Define an example problem*
  + Define our machine learning model $f$
  + Define a loss function $cal(L)$
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
  + Investigate the results
]

#slide(title: [Linear Regression])[
  *Task:* Given your education, predict your life expectancy #pause

  $X:$ Years in school #pause
  
  $Y:$ Age of death #pause

  *Approach:* Learn the parameters $theta$ such that 

  $ f(x, theta) = y $ #pause

  *Goal:* Given someone's education, you can predict how long they will live
]

#slide(title: [Linear Regression])[
  + *Define an example problem*
  + Define our machine learning model $f$
  + Define a loss function $cal(L)$
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
  + Investigate the results
]

#slide(title: [Linear Regression])[
  + Define an example problem
  + *Define our machine learning model $f$*
  + Define a loss function $cal(L)$
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
  + Investigate the results
]

#slide(title: [Linear Regression])[
  Soon, $f$ will be a deep neural network #pause

  For now, it is easier if we make $f$ a *linear function* #pause

  #align(center, grid(
    columns: 2,
    align: center,
    column-gutter: 2em,
    $ f(x, bold(theta)) = f(x, vec(theta_0, theta_1)) = theta_1 x + theta_0 $,
    cimage("figures/lecture_2/example_regression_graph.png", height: 50%)
  )) #pause

  Now, we need to find the parameters $bold(theta) = vec(theta_1, theta_0)$ that makes $f(x, bold(theta)) = y$
  //   To do this, we will need to find a *loss function* $cal(L)$

]

/*
#slide(title: [Linear Regression])[
  For simplicity, let $f$ be a linear function (we will make it deep later) #pause

  $ f(x, bold(theta)) = f(x, vec(theta_2, theta_1)) = W x + b $ #pause

  Now, we need to find the parameters $bold(theta) = vec(theta_2, theta_1)$ that makes $f(x, bold(theta)) = y$
  //   To do this, we will need to find a *loss function* $cal(L)$
]
*/

#slide(title: [Linear Regression])[
  + Define an example problem
  + *Define our machine learning model $f$*
  + Define a loss function $cal(L)$
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
  + Investigate the results
]

#slide(title: [Linear Regression])[
  + Define an example problem
  + Define our machine learning model $f$
  + *Define a loss function $cal(L)$*
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
  + Investigate the results
]


#slide(title: [Linear Regression])[
  Now, we need to find the parameters $bold(theta) = vec(theta_2, theta_1)$ that make $f(x, bold(theta)) = y$ #pause

  *Question:* How do we find $bold(theta)$? (Hint: We want $f(x, bold(theta)) = y$) #pause

  *Answer:* We will minimize the *loss* (error) between $f(x, bold(theta))$ and $y$ 

  #align(center, grid(
    columns: 2,
    align: center,
    column-gutter: 2em,
    [E.g., $ min_(bold(theta)) (f(x, bold(theta)) - y)^2 = 0 $], 
    cimage("figures/lecture_2/loss_function.png", height: 50%)
  ))
]

#slide(title: [Linear Regression])[
  We compute the loss using the *loss function* $cal(L)$ #pause

  $ cal(L)(x, y, bold(theta)) $ #pause

  The loss function tells us how close $f(x)$ is to $y$ #pause

  By *minimizing* the loss function, we make $f(x) = y$ #pause

  There are many possible loss functions, but for now we will use the *mean-square error* #pause

  $ "error"(y, hat(y)) = (y - hat(y))^2 $
]

#slide(title: [Linear Regression])[
  //$ f(x, vec(theta_2, theta_1)) = W x + b $ #pause

  Let's derive the error function #pause

  #side-by-side[
    $ f(x, bold(theta)) = y $ 
  ][
    f(x) should predict y
  ] #pause

  #side-by-side[
    $ f(x, bold(theta)) - y = 0 $
  ][
    Move y to LHS
  ] #pause

  #side-by-side[
    $ (f(x, bold(theta)) - y)^2 = 0 $
  ][ 
    Square for minimization
  ] #pause

  #side-by-side[
    $ "error"(f(x, bold(theta)), y) = (f(x, bold(theta)) - y)^2 $
  ][ 
  ] #pause

  /*
  #side-by-side[
    $ 1/2 (f(x, bold(theta)) - y)^2 = 0 $
  ][
    Scale by a constant
  ]
  #side-by-side[
    $ cal(L)(x, y, bold(theta)) =  1/2 (f(x,  bold(theta)) - y)^2 $
  ][
    This is our loss function!
  ] #pause
  */
  
]

#slide(title: [Linear Regression])[
  We can write the loss function for a single datapoint $x_i, y_i$ as

  $ cal(L)(x_i, y_i, bold(theta)) = "error"(f(x_i, bold(theta)),  y_i) = (f(x_i, bold(theta)) - y_i)^2 $ #pause

  By minimizing $cal(L)$, we can find the parameters $bold(theta)$ #pause

  $ 
   min_bold(theta) cal(L)(x_i, y_i, bold(theta)) = min_bold(theta) "error"(f(x_i, bold(theta)),  y_i) = min_bold(theta) (f(x_i, bold(theta)) - y_i)^2 
  $ #pause

  *Question:* Any issues with $cal(L)$? #pause

  *Answer:* We only consider a single datapoint! We want to learn $bold(theta)$ for the entire dataset
]

#slide(title: [Linear Regression])[
  For a single $x_i, y_i$:
  $ 
   min_bold(theta) cal(L)(x_i, y_i, bold(theta)) = min_bold(theta) "error"(f(x_i, bold(theta)),  y_i) = min_bold(theta) (f(x_i, bold(theta)) - y_i)^2 
  $ #pause

  For the entire dataset: 
  $ 
   min_bold(theta) cal(L)(x_i, y_i, bold(theta)) = min_bold(theta) sum_(i=1)^n "error"(f(x_i, bold(theta)),  y_i) = min_bold(theta) sum_(i=1)^n (f(x_i, bold(theta)) - y_i)^2 
  $ #pause

  Minimizing this loss function will give us the optimal parameters!
]

#slide(title: [Linear Regression])[
  + Define an example problem
  + Define our machine learning model $f$
  + *Define a loss function $cal(L)$*
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
  + Investigate the results
]

#slide(title: [Linear Regression])[
  + Define an example problem
  + Define our machine learning model $f$
  + Define a loss function $cal(L)$
  + *Use $cal(L)$ to learn the parameters $theta$ of $f$*
  + Investigate the results
]

#slide(title: [Linear Regression])[
  *Question:* How do we minimize:
  $ 
   min_bold(theta) cal(L)(x_i, y_i, bold(theta)) = min_bold(theta) sum_(i=1)^n "error"(f(x_i, bold(theta)),  y_i) = min_bold(theta) sum_(i=1)^n (f(x_i, bold(theta)) - y_i)^2 
  $ #pause

  *Answer:* For now, magic! We need more knowledge before we can derive this.
]


/*
#slide(title: [Linear Regression])[
  Back to the original problem... #pause
  
  Now, we need to find the parameters $bold(theta) = vec(theta_2, theta_1)$ that makes $f(x, vec(theta_2, theta_1)) = y$ #pause

  We will minimize this loss

  $
    min_bold(theta) sum_(i=1)^n cal(L) (f(x_i, bold(theta)), y_i)
  $ #pause

  But what is $cal(L)$? #pause

  $cal(L)$ should be smaller when $f(x, bold(theta)) = y$ and bigger when $f(x, bold(theta)) != y$
]



#slide(title: [Linear Regression])[



  #side-by-side[
    $ min_(theta_2, theta_1) cal(L)(x, y) = min_(theta_2, theta_1) 1/2 (f(x,  bold(theta)) - y)^2 $
  ][
    We want to minimize the loss function
  ] #pause

  #v(2em)
  #align(center)[How do we minimize $cal(L)$?]
]

#slide(title: [Linear Regression])[
    #side-by-side[
    $ min_(theta_2, theta_1) cal(L)(x, y) = min_(theta_2, theta_1) 1/2 (f(x, vec(theta_2, theta_1)) - y)^2 $
  ][
    How do we minimize this?
  ] #pause

  #v(2em)

  #align(center)[Magic! For, now you will have to trust me -- we will see later in the course how to derive this expression] 
]
*/

#slide(title: [Linear Regression])[
  #side-by-side[
    First, construct a *design matrix* $bold(X)$ containing input data $x$ and a constant 1 for the bias. Also construct a $bold(y)$ vector!
  ][  
  $ bold(X) = mat(x_1, 1; x_2, 1; dots.v, dots.v; x_n, 1), bold(y) = vec(y_1, y_2, dots.v, y_n) $
  ]
  
  #side-by-side[
    And remember the parameters $bold(theta)$
  ][ $ bold(theta) = vec(theta_1, theta_0),  $]

  #v(2em)

  #side-by-side[$ bold(theta) = (bold(X)^top bold(X) )^(-1) bold(X)^top bold(y) $ ][The solution to linear regression]
]

#slide(title: [Linear Regression])[
  + Define an example problem
  + Define our machine learning model $f$
  + Define a loss function $cal(L)$
  + *Use $cal(L)$ to learn the parameters $theta$ of $f$*
  + Investigate the results
]

#slide(title: [Linear Regression])[
  + Define an example problem
  + Define our machine learning model $f$
  + Define a loss function $cal(L)$
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
  + *Investigate the results*
]

#slide(title: [Example])[
  Back to the example... #pause

  *Task:* Given your education, predict your life expectancy #pause

  $X:$ Years in school #pause
  
  $Y:$ Age of death #pause

  *Goal:* Learn the parameters $bold(theta)$ such that 

  $ f(x, bold(theta)) = y $ #pause

  #v(1em)
  #align(center)[You will be doing this in your first assignment!]
]

#slide(title: [Example])[
  Back to the example...

  *Task:* Given your education, predict your life expectancy #pause

  #cimage("figures/lecture_2/linear_regression.png", height: 60%)
]

#slide[
  Tips for assignment 1 #pause

  ```py
  def f(theta, design): 
    # Linear function
    return theta @ design
  ``` #pause

  Not all matrices can be inverted! Ensure the matrices are square and the condition number is low

  ```py
  A.shape
  cond = jax.numpy.linalg.cond(A)
  ```
]

#slide(title: [Linear Regression])[
  + Define an example problem #pause
  + Define our machine learning model $f$ #pause
  + Define a loss function $cal(L)$ #pause
  + Use $cal(L)$ to learn the parameters $theta$ of $f$ #pause
  + Investigate the results
]

#focus-slide[Relax]


#slide[
  We figured out linear regression! #pause
  - Outliers #pause
  - Can we go beyond linear? #pause
  - Overfitting #pause
  - Test and train splits #pause
]

#slide(title: [Example])[
  Back to the example...

  *Task:* Given your education, predict your life expectancy #pause

  #cimage("figures/lecture_2/linear_regression.png", height: 60%) #pause

  #align(center)[Could we do better than a linear function $f$?]
]

#slide(title: [Example])[
  #align(center)[Could we do better than a linear function $f$?] #pause

  #cimage("figures/lecture_2/polynomial_regression_n2.png", height: 50%)

  What if we used a polynomial instead? #pause

  $ f(x) = theta_n x^n + theta_(n - 1) x^(n-1), dots, theta_1 + x^1 + theta_0 $
]

#slide(title: [Example])[
  $ f(x, bold(theta)) = f(x, vec(theta_0, theta_1, theta_2, dots.v, theta_n)) = theta_n x^n + theta_(n - 1) x^(n-1), dots, theta_1 + x^1 + theta_0 $ #pause

  //We can write this as a matrix product

  $ f(x, bold(theta)) = mat(
    theta_n, theta_(n-1), dots, theta_1, b
  )  mat(
    x^n;
    x^(n-1);
    dots.v;
    x^1;
    1
  )
  $
]

#slide(title: [Example])[
  $ f(x, bold(theta)) = theta_n x^n + theta_(n - 1) x^(n-1), dots, theta_1 x^1 + theta_0 $ #pause

  How do we choose $n$? Let us try different $n$ #pause

  #grid(
    columns: 3,
    row-gutter: 1em,
    image("figures/lecture_2/polynomial_regression_n2.png"),
    image("figures/lecture_2/polynomial_regression_n3.png"),
    image("figures/lecture_2/polynomial_regression_n5.png"),
    $ n = 2 $,
    $ n = 3 $,
    $ n = 5 $
  )
]

#slide(title: [Example])[
  $ f(x, bold(theta)) = theta_n x^n + theta_(n - 1) x^(n-1), dots, theta_1 + x^1 + theta_0 $ #pause

  #grid(
    columns: 3,
    row-gutter: 1em,
    image("figures/lecture_2/polynomial_regression_n2.png"),
    image("figures/lecture_2/polynomial_regression_n3.png"),
    image("figures/lecture_2/polynomial_regression_n5.png"),
    $ n = 2 $,
    $ n = 3 $,
    $ n = 5 $
  ) 

  *Question:* Which $n$ should we pick? Why?
]

#slide(title: [Example])[
  Data is inherently noisy #pause

  The world is governed by random processes #pause

  #cimage("figures/lecture_2/normal_dist.png", height: 50%),
]

#slide(title: [Example])[
  This is just an estimate
  
  #cimage("figures/lecture_2/polynomial_regression_n2.png", height: 60%) #pause

  Going to school for 20 years will not save you from a hungry bear
]

#slide(title: [Example])[
  #cimage("figures/lecture_2/polynomial_regression_n3.png", height: 60%) #pause


  When we fit to noise instead of the trend, we call it *overfitting* #pause

  Overfitting is bad because our predictions will be inaccurate
]

#slide(title: [Example])[
  How can we measure overfitting? #pause

  Learn our parameters from one subset of data: *training dataset* #pause

  Test our model on a different subset of data: *testing dataset* #pause

  #cimage("figures/lecture_2/train_test_split.png", height: 60%)

  //Training and testing dataset must come from the same distribution #pause
  //  - Select training and testing datasets randomly
]

#slide(title: [Example])[
  *Question:* How do we choose the training and testing datasets? #pause

  #align(center,grid(
    columns: 2,
    column-gutter: 20%,
    align: center,
    $ cal(D)_"train" &= vec(x_1, x_2, x_3) \ 
    cal(D)_"test" & = vec(x_4, x_5) $, 
    $ cal(D)_"train" &= vec(x_4, x_1, x_3) \ 
    cal(D)_"test" & = vec(x_2, x_5) $, 
  ))

  *Answer:* Always shuffle the data #pause

  ML relies on the *Independent and Identically Distributed (IID)* assumption 
]

// TODO we should formally define minimization of test loss as the objective


#slide(title: [Example])[
  - Overfitting
  - Outliers
  - Regularization
  - Etc
]