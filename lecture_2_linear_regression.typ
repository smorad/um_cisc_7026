#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *

// TODO handle x_i, y_i in the loss function, should be multiple x and y

// TODO: Motivate why we need a loss function
// TODO: Add review
// TODO: Slide titles


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

#let log_plot = canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: none,
    y-tick-step: none,
    {
      plot.add(
        domain: (0, 25), 
        x => calc.log(1 + x),
        label: $ log(1 + x) $
      )
    })
})

#title-slide(
  // Section time: 34 mins at leisurely pace
  title: [Regression],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
  //logo: image("logo.jpg", width: 25%)
)

#slide[
  Today, we will learn about linear regression #pause

  Probably the oldest method for machine learning (Gauss and Legendre) #pause

  #cimage("figures/lecture_1/timeline.svg")
]

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
  Today, we will come up with a regression problem and then solve it! #pause

  + Define an example problem #pause
  + Define our machine learning model $f$ #pause
  + Define a loss function $cal(L)$ #pause
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
]

#slide(title: [Linear Regression])[
  + *Define an example problem*
  + Define our machine learning model $f$
  + Define a loss function $cal(L)$
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
]


#slide(title: [Linear Regression])[
  The World Health Organization (WHO) has collected data on life expectancy #pause

  #cimage("figures/lecture_2/who.png", height: 60%)

  #text(size: 14pt)[Available for free at #link("https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates/ghe-life-expectancy-and-healthy-life-expectancy")
  ] 

]

#slide(title: [Linear Regression])[
  The WHO collected data from roughly 3,000 people from 193 countries #pause

  For each person, they recorded: #pause
  - Home country #pause
  - Alcohol consumption #pause
  - Education #pause
  - Gross domestic product (GDP) of the country #pause
  - Immunizations for Measles and Hepatitis B #pause
  - How long this person lived #pause

We can use this data to make future predictions
]

#slide(title: [Linear Regression])[
  Since everyone here is very educated, we will focus on how education affects life expectancy #pause

  There are studies showing a causal effect on education on health #pause
    - _The causal effects of education on health outcomes in the UK Biobank._ Davies et al. _Nature Human Behaviour_. #pause
    - By staying in school, you are likely to live longer 
]

#slide(title: [Linear Regression])[
  *Task:* Given your education, predict your life expectancy #pause

  $X in bb(R)_+:$ Years in school #pause
  
  $Y in bb(R)_+:$ Age of death #pause

  *Approach:* Learn the parameters $theta$ such that 

  $ f(x, theta) = y; quad x in X, y in Y $ #pause

  *Goal:* Given someone's education, predict how long they will live
]


// 16:00 very slow, no review

#slide(title: [Linear Regression])[
  + *Define an example problem*
  + Define our machine learning model $f$
  + Define a loss function $cal(L)$
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
]

#slide(title: [Linear Regression])[
  + Define an example problem
  + *Define our machine learning model $f$*
  + Define a loss function $cal(L)$
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
]

#slide(title: [Linear Regression])[
  Soon, $f$ will be a deep neural network #pause

  For now, it is easier if we make $f$ a *linear function* #pause

  #align(center, grid(
    columns: 2,
    align: center,
    column-gutter: 2em,
    $ f(x, bold(theta)) = f(x, vec(theta_1, theta_0)) = theta_1 x + theta_0 $,
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
]

#slide(title: [Linear Regression])[
  + Define an example problem
  + Define our machine learning model $f$
  + *Define a loss function $cal(L)$*
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
]


#slide(title: [Linear Regression])[
  Now, we need to find the parameters $bold(theta) = vec(theta_1, theta_0)$ that make $f(x, bold(theta)) = y$ #pause

  *Question:* How do we find $bold(theta)$? (Hint: We want $f(x, bold(theta)) = y$) #pause

  *Answer:* We will minimize the *loss* (error) between $f(x, bold(theta))$ and $y$, for all
  
  $ x in X, y in Y $

  /*
  #align(center, grid(
    columns: 2,
    align: center,
    column-gutter: 2em,
    [E.g., $ argmin_(bold(theta)) (f(x, bold(theta)) - y)^2 = 0 $], 
    cimage("figures/lecture_2/loss_function.png", height: 50%)
  ))
  */
]

#slide(title: [Linear Regression])[
  We compute the loss using the *loss function* $cal(L): X times Y times Theta |-> bb(R)$ #pause

  $ cal(L)(x, y, bold(theta)) $ #pause

  The loss function tells us how close $f(x, bold(theta))$ is to $y$ #pause

  By *minimizing* the loss function, we make $f(x, bold(theta)) = y$ #pause

  There are many possible loss functions, but for regression we often use the *square error* #pause

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
  ] 

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

  We want to find the parameters $bold(theta)$ that minimize $cal(L)$ #pause

  $ 
   argmin_bold(theta) cal(L)(x_i, y_i, bold(theta)) = argmin_bold(theta) "error"(f(x_i, bold(theta)),  y_i) = argmin_bold(theta) (f(x_i, bold(theta)) - y_i)^2 
  $ #pause

  *Question:* Any issues with $cal(L)$? Will it give us a good prediction for all $x$? #pause

  *Answer:* We only consider a single datapoint! We want to learn $bold(theta)$ for the entire dataset
]

#slide(title: [Linear Regression])[
  For a single $x_i, y_i$:
  $ 
   argmin_bold(theta) cal(L)(x_i, y_i, bold(theta)) = argmin_bold(theta) "error"(f(x_i, bold(theta)),  y_i) = argmin_bold(theta) (f(x_i, bold(theta)) - y_i)^2 
  $ #pause

  For the entire dataset: 
  $ bold(x) = mat(x_1, x_2, dots, x_n)^top, bold(y) = mat(y_1, y_2, dots, y_n)^top $
  $ 
   argmin_bold(theta) cal(L)(bold(x), bold(y), bold(theta)) = argmin_bold(theta) sum_(i=1)^n "error"(f(x_i, bold(theta)),  y_i) = argmin_bold(theta) sum_(i=1)^n (f(x_i, bold(theta)) - y_i)^2 
  $ #pause

  Minimizing this loss function will give us the optimal parameters!
]

// 28:30 slow, no review
#slide(title: [Linear Regression])[
  + Define an example problem
  + Define our machine learning model $f$
  + *Define a loss function $cal(L)$*
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
]

#slide(title: [Linear Regression])[
  + Define an example problem
  + Define our machine learning model $f$
  + Define a loss function $cal(L)$
  + *Use $cal(L)$ to learn the parameters $theta$ of $f$*
]

#slide(title: [Linear Regression])[
  *Question:* How do we minimize:
  $ 
   argmin_bold(theta) cal(L)(x_i, y_i, bold(theta)) &= argmin_bold(theta) sum_(i=1)^n "error"(f(x_i, bold(theta)),  y_i) \ &= argmin_bold(theta) sum_(i=1)^n (f(x_i, bold(theta)) - y_i)^2 
  $ #pause

  *Answer:* For now, magic! We need more knowledge before we can derive this.
]


/*
#slide(title: [Linear Regression])[
  Back to the original problem... #pause
  
  Now, we need to find the parameters $bold(theta) = vec(theta_2, theta_1)$ that makes $f(x, vec(theta_2, theta_1)) = y$ #pause

  We will minimize this loss

  $
    argmin_bold(theta) sum_(i=1)^n cal(L) (f(x_i, bold(theta)), y_i)
  $ #pause

  But what is $cal(L)$? #pause

  $cal(L)$ should be smaller when $f(x, bold(theta)) = y$ and bigger when $f(x, bold(theta)) != y$
]



#slide(title: [Linear Regression])[



  #side-by-side[
    $ argmin_(theta_2, theta_1) cal(L)(x, y) = argmin_(theta_2, theta_1) 1/2 (f(x,  bold(theta)) - y)^2 $
  ][
    We want to minimize the loss function
  ] #pause

  #v(2em)
  #align(center)[How do we minimize $cal(L)$?]
]

#slide(title: [Linear Regression])[
    #side-by-side[
    $ argmin_(theta_2, theta_1) cal(L)(x, y) = argmin_(theta_2, theta_1) 1/2 (f(x, vec(theta_2, theta_1)) - y)^2 $
  ][
    How do we minimize this?
  ] #pause

  #v(2em)

  #align(center)[Magic! For, now you will have to trust me -- we will see later in the course how to derive this expression] 
]
*/

#slide(title: [Linear Regression])[
  First, we will construct a *design matrix* $bold(X)_D$ containing input data $x$ #pause

  $ bold(X)_D = mat(x_1, 1; x_2, 1; dots.v, dots.v; x_n, 1) $
]

#slide(title: [Linear Regression])[  
  We add the column of ones so that we can multiply $bold(X)^top_D$ with $bold(theta)$ to get a linear function $theta_1 x + theta_0$ evaluated at each data point

  $ bold(X)_D bold(theta) = mat(x_1, 1; x_2, 1; dots.v, dots.v; x_n, 1) vec(theta_1, theta_0) = vec(theta_1 x_1 + theta_0, theta_1 x_2 + theta_0, dots.v, theta_1 x_n + theta_0) $
]

#slide(title: [Linear Regression])[
  #side-by-side[
    With our design matrix $bold(X)_D$ and desired output $bold(y),$
    
  ][  
  $ bold(X)_D = mat(x_1, 1; x_2, 1; dots.v, dots.v; x_n, 1), bold(y) = vec(y_1, y_2, dots.v, y_n) $
  ]
  
  #side-by-side[
    and our parameters $bold(theta),$
  ][ $ bold(theta) = vec(theta_1, theta_0),  $]

  #v(2em)

  #side-by-side[$ bold(theta) = (bold(X)_D^top bold(X)_D )^(-1) bold(X)_D^top bold(y) $ ][We can find the parameters that minimize $cal(L)$]
]

#slide(title: [Linear Regression])[
  + Define an example problem
  + Define our machine learning model $f$
  + Define a loss function $cal(L)$
  + *Use $cal(L)$ to learn the parameters $theta$ of $f$*
]

// 34 minutes no review, very slow
#slide(title: [Linear Regression])[
  + Define an example problem
  + Define our machine learning model $f$
  + Define a loss function $cal(L)$
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
]

#slide(title: [Example])[
  Back to the example... #pause

  *Task:* Given your education, predict your life expectancy #pause

  $X in bb(R)_+:$ Years in school #pause
  
  $Y in bb(R)_+:$ Age of death #pause

  *Approach:* Learn the parameters $theta$ such that 

  $ f(x, theta) = y; quad x in X, y in Y $ #pause

  *Goal:* Given someone's education, predict how long they will live #pause

  #align(center)[You will be doing this in your first assignment!]
]

/*
#slide(title: [Example])[
  Back to the example...

  *Task:* Given your education, predict your life expectancy #pause

  #cimage("figures/lecture_2/linear_regression.png", height: 60%)
]
*/

#slide[
  Tips for assignment 1 #pause

  ```py
  def f(theta, design): 
    # Linear function
    return design @ theta
  ``` #pause

  Not all matrices can be inverted! Ensure the matrices are square and the condition number is low

  ```py
  A.shape
  cond = jax.numpy.linalg.cond(A)
  ``` #pause

  Everything you need is in the lecture notes
]

// 38 mins very slow, no review
#slide(title: [Linear Regression])[
  + Define an example problem 
  + Define our machine learning model $f$ 
  + Define a loss function $cal(L)$ 
  + Use $cal(L)$ to learn the parameters $theta$ of $f$
]

#focus-slide[Relax]

#slide(title: [Example])[
  *Task:* Given your education, predict your life expectancy #pause

  Plot the datapoints $(x_1, y_1), (x_2, y_2), dots $ #pause

  Plot the curve $f(x, bold(theta)) = theta_1 x + theta_0; quad x in [0, 25]$ #pause

  #cimage("figures/lecture_2/linear_regression.png", height: 60%) #pause

  //We figured out linear regression! #pause

  //But can we do better?
]

#slide[
  #cimage("figures/lecture_2/linear_regression.png", height: 60%) 

  We figured out linear regression! #pause

  But can we do better?
]

#slide[
  + Beyond linear functions #pause
  + Overfitting #pause
  + Outliers #pause
  + Regularization 
]

#slide[
  + *Beyond linear functions*
  + Overfitting
  + Outliers
  + Regularization
]

#slide[
  *Question:* #pause
  #side-by-side[
    Does the data look linear? #pause
    #cimage("figures/lecture_2/linear_regression.png", height: 60%) #pause
  ][
    Or maybe more logarithmic? #pause
    #log_plot #pause
  ]

  However, linear regression must be linear! 
]

#slide[
  *Question:* What does it mean when we say linear regression is linear? #pause

  *Answer:* The function $f(x, theta)$ is a linear function of $x$ #pause

  *Trick:* Change of variables to make $f$ nonlinear: $x_"new" = log(1 + x_"data")$ #pause

  $ bold(X)_D = mat(x_1, 1; x_2, 1; dots.v, dots.v; x_n, 1) => bold(X)_D = mat(log(1 + x_1), 1; log(1 + x_2), 1; dots.v, dots.v; log(1 + x_n), 1) $

  Now, $f$ is a linear function of $log(1 + x)$ -- a nonlinear function of $x$!
]

#slide[
  New design matrix...
  $ bold(X)_D = mat(log(1 + x_1), 1; log(1 + x_2), 1; dots.v, dots.v; log(1 + x_n), 1) $

  #side-by-side[
  New function...
  $ f(x, vec(theta_1, theta_0)) = theta_1 log(1 + x) + theta_0 $#pause
  ][
  Same solution...
  $ bold(theta) = (bold(X)_D^top bold(X)_D )^(-1) bold(X)_D^top bold(y) $
  ]
]

#slide[
  #cimage("figures/lecture_2/log_regression.png", height: 60%) #pause

  Better, but still not perfect #pause

  Can we do even better?
]

#slide[
  What about polynomials? #pause

  $ f(x) = a x^n + b x^(n-1) + dots + c x + d $ #pause

  Polynomials can approximate *any* function (universal function approximator) #pause

  Can we extend linear regression to polynomials?

  //$ f(x, bold(theta)) = theta_n x^n + theta_(n - 1) x^(n-1) + dots + theta_1 + x^1 + theta_0 $
]

#slide[
  Expand $x$ to a multi-dimensional input space... #pause

  $ bold(X)_D = mat(x_1, 1; x_2, 1; dots.v, dots.v; x_n, 1) => bold(X)_D = mat(
    x_1^n, x_1^(n-1), dots, x_1, 1; 
    x_2^n, x_2^(n-1), dots, x_2, 1; 
    dots.v, dots.v, dots.down; 
    x_n, x_n^(n-1), dots, x_n, 1
    ) $

  And add some new parameters...
  $ bold(theta) = mat(theta_1, theta_0)^top => bold(theta) =  mat(theta_n, theta_(n-1), dots, theta_1, theta_0)^top $
]

#slide[
  $ bold(X)_D bold(theta) = mat(
    x_1^n, x_1^(n-1), dots, x_1, 1; 
    x_2^n, x_2^(n-1), dots, x_2, 1; 
    dots.v, dots.v, dots.down; 
    x_n, x_n^(n-1), dots, x_n, 1
    ) vec(theta_n, theta_(n-1), dots.v, theta_0) = 
    vec(
      theta_n x_1^n + theta_(n-1) x_1^(n-1) + dots + theta_0,
      theta_n x_2 + theta_(n-1) x_2^(n-1) + dots + theta_0,
      dots.v,
      theta_n x_n^n + theta_(n-1) x_n^(n-1) + dots + theta_0
    )
   $ #pause

  New function...
  $ f(x, bold(theta)) = theta_n x^n + theta_(n - 1) x^(n-1), dots, theta_1 + x^1 + theta_0 $ #pause

  Same solution...
  $ bold(theta) = (bold(X)_D^top bold(X)_D )^(-1) bold(X)_D^top bold(y) $

  //By changing the input space, we can fit a polynomial to the data!
]

#slide[
  $ f(x, bold(theta)) = theta_n x^n + theta_(n - 1) x^(n-1), dots, theta_1 + x^1 + theta_0 $ #pause

  *Summary:* By changing the input space, we can fit a polynomial to the data using a linear fit!
]

#slide[
  + *Beyond linear functions*
  + Overfitting
  + Outliers
  + Regularization
]
#slide[
  + Beyond linear functions
  + *Overfitting*
  + Outliers
  + Regularization
]

#slide[
  $ f(x, bold(theta)) = theta_n x^n + theta_(n - 1) x^(n-1), dots, theta_1 + x^1 + theta_0 $ #pause

  How do we choose $n$ (polynomial order) that provides the best fit? #pause

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

#slide[
  How do we choose $n$ (polynomial order) that provides the best fit? 

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

  #side-by-side[Pick the $n$ with the smallest loss][
    $ argmin_(bold(theta), n) cal(L)(bold(x), bold(y), (bold(theta), n)) $]
]

#slide[
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

  *Question:* Which $n$ do you think has the smallest loss? #pause

  *Answer:* $n=5$, but intuitively, $n=5$ does not seem very good...
]

#slide[
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

  More specifically, $n=5$ will not generalize to new data #pause

  We will only use our model for new data (we already have the $y$ for a known $x$)! #pause
]

#slide[
  When our model has a small loss but does not generalize to new data, we call it *overfitting* #pause

  The model has fit too closely to the sampled data points, rather than the trend #pause

  Models that overfit are not useful for making predictions #pause

  Back to the question... #pause
  
  *Question:* How do we choose $n$ such that our polynomial model works for unseen/new data? #pause

  *Answer:* Compute the loss on unseen data!
]

#slide[
  To compute the loss on unseen data, we will need unseen data #pause

  Let us create some unseen data! #pause

    #cimage("figures/lecture_2/train_test_split.png", height: 60%)
]

#slide(title: [Example])[
  *Question:* How do we choose the training and testing datasets? #pause

  $ "Option 1:" bold(x)_"train" &= vec(
    x_1, x_2, x_3
  ) bold(y)_"train" &= vec(
    y_1, y_2, y_3
  ); quad
  bold(x)_"test" &= vec(
    x_4, x_5
  ); bold(y)_"test" &= vec(
    y_4, y_5
  ) 
  $ #pause

  $ "Option 2:" bold(x)_"train" &= vec(
    x_4, x_1, x_3
  ) bold(y)_"train" &= vec(
    y_4, y_1, y_3
  ); quad
  bold(x)_"test" &= vec(
    x_2, x_5
  ); bold(y)_"test" &= vec(
    y_2, y_5
  ) 
  $ #pause

  *Answer:* Always shuffle the data #pause

  *Note:* The model must never see the testing dataset during training. This is very important!
]

#slide[
  We can now measure how the model generalizes to new data #pause

  #cimage("figures/lecture_2/train_test_regression.png", height: 60%)

  Learn parameters from the train dataset, evaluate on the test dataset #pause

  #side-by-side[
    $cal(L)(bold(X)_"train", bold(y)_"train", bold(theta))$
  ][
    $cal(L)(bold(X)_"test", bold(y)_"test", bold(theta))$
  ]
]

#slide[
  We use separate training and testing datasets on *all* machine learning models, not just linear regression #pause
]




//

// Cast this as an improvement step by step

// Breakpoint after
/*
#slide(title: [Example])[
  Back to the example...

  *Task:* Given your education, predict your life expectancy #pause

  #cimage("figures/lecture_2/linear_regression.png", height: 60%) #pause

  #align(center)[Could we do better than a linear function $f$?]
]

#slide(title: [Example])[
  #align(center)[Could we do better than a linear function $f$?] #pause

  #cimage("figures/lecture_2/polynomial_regression_n2.png", height: 50%)

  #align(center)[What if we used a polynomial instead?] #pause

  $ f(x, bold(theta)) = theta_n x^n + theta_(n - 1) x^(n-1), dots, theta_1 + x^1 + theta_0 $
]

#slide(title: [Example])[
  But we said we were using a linear model, how can we come up with a nonlinear polynomial? #pause
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
    image("figures/lecture_2/polynomial_regression_n2.png", height: 45%),
    image("figures/lecture_2/polynomial_regression_n3.png", height: 45%),
    image("figures/lecture_2/polynomial_regression_n5.png", height: 45%),
    $ n = 2 $,
    $ n = 3 $,
    $ n = 5 $
  ) 

  *Question:* Which $n$ should we pick? #pause

  *Answer:* $n=2$ feels right, but why?
]

#slide(title: [Example])[
  #grid(
    columns: 3,
    row-gutter: 1em,
    image("figures/lecture_2/polynomial_regression_n2.png", height: 45%),
    image("figures/lecture_2/polynomial_regression_n3.png", height: 45%),
    image("figures/lecture_2/polynomial_regression_n5.png", height: 45%),
    $ n = 2 $,
    $ n = 3 $,
    $ n = 5 $
  ) 

  Data can be noisy and we want to fit the trend, not the noise
]

#slide(title: [Example])[
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

  Overfitting is bad because new predictions will be inaccurate
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
    $ (D)_"train" &= mat(x_1, y_1; x_2, y_2; x_3, y_3) \ 
    cal(D)_"test" & = mat(x_4, y_4; x_5, y_5) $, 
    $ cal(D)_"train" &= mat(x_4, y_4; x_1, y_1; x_3, y_3) \ 
    cal(D)_"test" & = mat(x_2, y_2; x_5, y_5) $, 
  ))

  *Answer:* Always shuffle the data #pause

  *Note:* The model must never see the testing dataset during training. This is very important!

  //ML relies on the *Independent and Identically Distributed (IID)* assumption 
]

#slide(title: [Conclusion])[
  Today we: #pause
  - Came up with a linear regression task #pause
  - Proposed a linear model #pause
  - Defined the square error loss function #pause
  - Found $bold(theta)$ that minimized the loss
  - Used a trick to extend linear regression to nonlinear functions #pause
  - Discussed overfitting and test/train splits
]
*/