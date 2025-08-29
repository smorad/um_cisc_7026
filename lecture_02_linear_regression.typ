#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.0"
#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "common.typ": *
#import "@preview/pinit:0.2.2": *

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
    title: [Linear Regression],
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

#let argmin_plot = canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 1,
    y-tick-step: 2,
    x-label: $ x $,
    y-label: $ f(x) $,
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


= Announcements
==
Homework 0 was ok? #pause

Homework 1 released, due in ~2 weeks (see Moodle) #pause
- Discuss more at the end of class #pause

I am getting married, and will be away 09.12 and 09.19 #pause
- Yutao will lecture 09.12 #pause
- My students will proctor exam 1 on 09.19 #pause

==

Currently writing exam 1, probably 6 questions: #pause
- 1 question function notation
- 1 question set notation
- 2 questions linear regression (make sure you can invert matrices)
- 1 question neural networks (neurons)
- 1 question gradient descent (know how to take derivatives, no need to memorize formulas) #pause

Bring a pen/pencil/eraser to exam, you need nothing else #pause

You will have 3 hours to finish the exam #pause
- Probably takes most students 1-1.5 hours #pause
- No rush, take as long as you need

= Review
==
  We often know *what* we want, but we do not know *how* #pause

  We have many pictures of either dogs or muffins $x in X$ #pause

  We want to know if the picture is [dog | muffin] $y in Y$ #pause

  We learn a function or mapping from $X$ to $Y$

  $ f: X times |-> Y $ 

==

Usually, functions are defined once and static: $f(x) = x^2$ #pause

But in machine learning, we must *learn* the function #pause

To avoid confusion, we introduce the *function parameters* 

#side-by-side[
  $ theta in Theta $ #pause
][
  $ f: X times Theta |-> Y $ #pause
]

= Math Notation

= Math Notation - Sets <touying:hidden>

==
Before we go any futher, we need to agree on math notation #pause

If you ever get confused, come back to these slides #pause

#side-by-side(align: horizon)[
  Vectors
][bold small characters][
  $ bold(x) = vec(x_1, x_2, dots.v, x_n) $
] #pause

#side-by-side(align: horizon)[
  Matrices
][bold big characters][
  $ bold(X) = mat(
    x_(1,1), x_(1,2), dots, x_(1,n); 
    x_(2,1), x_(2,2), dots, x_(2,n); 
    dots.v, dots.v, dots.down, dots.v;
    x_(m,1), x_(m,2), dots, x_(m,n); 
  ) $
]

==
We will represent *tensors* as nested vectors or matrices #pause

#side-by-side(align: horizon)[
  Tensor
][
  $ bold(x) = vec(bold(x)_1, bold(x)_2, dots.v, bold(x)_n) $
] #pause

Each $bold(x)_i$ is a vector 

==
Same for matrices

#side-by-side(align: horizon)[
  Tensor of matrices
][
  $ bold(X) = mat(
    bold(x)_(1,1), bold(x)_(1,2), dots, bold(x)_(1,n); 
    bold(x)_(2,1), bold(x)_(2,2), dots, bold(x)_(2,n); 
    dots.v, dots.v, dots.down, dots.v;
    bold(x)_(m,1), bold(x)_(m,2), dots, bold(x)_(m,n); 
  ) $
] #pause

I use square brackets for data index #pause

$x_([i], j, k)$ indexes a 3D tensor, where the first dimension is the dataset #pause
-  Dataset of matrices (2D)

==
*Question:* What is the difference between the following?

$ bold(X) = mat(
  x_(1,1), x_(1,2), dots, x_(1,n); 
  x_(2,1), x_(2,2), dots, x_(2,n); 
  dots.v, dots.v, dots.down, dots.v;
  x_(m,1), x_(m,2), dots, x_(m,n); 
) $

$ bold(X) = mat(
  bold(x)_(1,1), bold(x)_(1,2), dots, bold(x)_(1,n); 
  bold(x)_(2,1), bold(x)_(2,2), dots, bold(x)_(2,n); 
  dots.v, dots.v, dots.down, dots.v;
  bold(x)_(m,1), bold(x)_(m,2), dots, bold(x)_(m,n); 
) $

==
*Exception:* I will always write the parameters $theta$ lowercase #pause
- We introduce a scalar parameter $theta$ #pause
- Then introduce a parameter vector $bold(theta)$ #pause
- But later it becomes a matrix $bold(theta)$ #pause
- And then a tensor (vector of matrices) $bold(theta)$

==
Capital letters will often refer to *sets* #pause

$ X = {1, 2, 3, 4} $ #pause

We will represent important sets with blackboard font #pause

#side-by-side[$ bb(R) $][Set of all real numbers ${-1, 2.03, pi, dots}$] #pause
#side-by-side[$ bb(Z) $][Set of all integers ${-2, -1, 0, 1, 2, dots}$] #pause
#side-by-side[$ bb(Z)_+ $][Set of all *positive* integers ${1, 2, dots}$]

==
#side-by-side[
  $ [0, 1] $
][
  Closed interval $0.0, 0.01, 0.00 dots 1, 0.99, 1.0$
] #pause
#side-by-side[
  $ (0, 1) $
][
  Open interval $0.01, 0.00 dots 1, 0.99$
] #pause
#side-by-side[
  $ {0, 1} $
][
  Set of two numbers (boolean)
] #pause

#side-by-side[
  $ [0, 1]^k $
][
  A vector of $k$ numbers between 0 and 1
] #pause

#side-by-side[
  $ {0, 1}^(k times k) $
][
  A matrix of boolean values of shape $k$ by $k$
]

==
We will use various set operations #pause

#side-by-side[$ A subset.eq B $][$A$ is a subset of $B$] #pause
#side-by-side[$ A subset B $][$A$ is a strict subset of $B$] #pause
#side-by-side[$ a in A $][$a$ is an element of $A$] #pause
#side-by-side[$ b in.not A $][$b$ is not an element of $A$] #pause
#side-by-side[$ A union B $][The union of sets $A$ and $B$] #pause
#side-by-side[$ A inter B $][The intersection of sets $A$ and $B$] 

==
We will often use *set builder* notation #pause

$ { #pin(1) x + 1 #pin(2) | #pin(3) x in {1, 2, 3, 4} #pin(4) } $ #pause

#pinit-highlight(1, 2)
#pinit-point-from((1,2), pin-dx: 0pt, offset-dx: 0pt)[Function]

#pinit-highlight(3, 4, fill: blue.transparentize(80%))
#pinit-point-from((3,4),)[Domain] #pause

#v(2em)

You can think of this as a for loop 

```python
  output = {} # Set
  for x in {1, 2, 3, 4}:
    output.insert(x + 1)
```  #pause


#v(2em)

```python
  output = {x + 1 for x in {1, 2, 3, 4}}
```

= Math Notation - Functions <touying:hidden>

==
We define *functions* or *maps* between sets

$ #pin(1) f #pin(2) : #pin(3) bb(R) #pin(4) |-> #pin(5) bb(Z) #pin(6) $ #pause

#pinit-highlight(1, 2)
#pinit-point-from((1,2), pin-dx: 0pt, offset-dx: 0pt)[Name] #pause

#pinit-highlight(3, 4, fill: blue.transparentize(80%))
#pinit-point-from((3,4),)[Input] #pause

#pinit-highlight(5, 6, fill: green.transparentize(80%))
#pinit-point-from((5,6),)[Output] #pause

#v(2em)

This function $f$ maps a real number to an integer #pause

*Question:* What functions could $f$ be? #pause

$ "round": bb(R) |-> bb(Z) $ 

==

Functions can have multiple inputs

$ f: X times Theta |-> Y  $ #pause

The function $f$ maps elements from sets $X$ and $Theta$ to set $Y$ #pause

I will define variables when possible 

#side-by-side[$ X = bb(R)^n  $][$ Theta = bb(R)^(m times n) $][$ Y = [0, 1]^(n times m) $] 

==
Finally, functions can have a function as input or output #pause

*Question:* Any examples? #pause

$ dif / (dif x): underbrace((f: bb(R) |-> bb(R)), "Input function") |-> underbrace((f': bb(R) |-> bb(R)), "Output function") $ #pause

$ dif / (dif x) [x^2] = 2x $

/*
== // 15:00
The $max$ function returns the maximum of a function over a domain #pause

$ max: (f: X |-> Y) times (Z subset.eq X) |-> Y $ #pause

$ max_(x in Z) f(x) $ #pause


The $argmax$ operator returns the input that maximizes a function #pause

$ argmax: (f: X |-> Y) times (Z subset.eq X) |-> Z $ #pause

$ argmax_(x in Z) f(x) $ 

==

We also have the $min$ and $argmin$ operators, which minimize $f$ 

$ min: (f: X |-> Y) times (Z subset.eq X) |-> Y $ #pause

$ min_(x in Z) f(x) $ #pause

$ argmin: (f: X |-> Y) times (Z subset.eq X) |-> Z $ #pause

$ argmin_(x in Z) f(x) $ #pause

We want to make optimal decisions, so we will often take the minimum or maximum of functions
*/


= Notation Exercises
== // 20:00 + 2

#side-by-side[$ bb(R)^n $ #pause][Set of all vectors containing $n$ real numbers #pause]
#side-by-side[$ {3, 4, dots, 31} $ #pause][Set of all integers between 3 and 31 #pause]
#side-by-side[$ [0, 1]^n $ #pause][Set of all vectors of length $n$ with values between 0 and 1 #pause] 
#side-by-side[$ {0, 1}^n $ #pause][Set of all boolean vectors of length $n$]

/*
==
#side-by-side[
$ f(x) = -(x + 1)^2 $ #pause
][
    #align(center)[
        #canvas(length: 1cm, {
        plot.plot(size: (8, 6),
            x-tick-step: 1,
            y-tick-step: 1,
            y-min: -4,
            y-max: 1,
            y-label: $ f(x) $,
            {
            plot.add(
                domain: (-3, 3), 
                style: (stroke: (thickness: 5pt, paint: red)),
                x => -calc.pow(x + 1, 2)
            )
            })
        })
    ] #pause
]
#side-by-side[
  $ max_(x in bb(R)) f(x) ? $ #pause
][
  $ argmax_(x in bb(R)) f(x) ? $ #pause
][
  $ argmax_(x in bb(Z)_+) f(x) ? $ #pause
]

#side-by-side[$ 0 $ #pause][$ -1 $ #pause][$ 1 $]
*/
==

$ {x^(1/2) | x in bb(R)_+} $ #pause

*Question:* What is this? #pause

*Answer:* The results of evaluating $f(x) = sqrt(x)$ for all positive real numbers #pause

$ {2x | x in bb(Z)_+} $ #pause

*Question:* What is this? #pause

*Answer:* Set of all positive even integers

// Review 8 mins


// 15 + 15 = 30 mins no questions

= Linear Regression
==
Today, we will learn about linear regression #pause

Probably the oldest method for machine learning (Gauss and Legendre) #pause

#cimage("figures/lecture_1/timeline.svg", height: 55%) #pause

Neural networks share many similarities with linear regression

==
  Many problems in ML can be reduced to *regression* or *classification* #pause

  *Regression* asks how many #pause
  - Given my parents height, How tall will I be? #pause
  - Given the rain today, how much rain will there be tomorrow? #pause
  - Given a camera image, how far away is this object? #pause

  *Classification* asks which one #pause
  - Is this image of a dog or muffin? #pause
  - Given the rain today, will it rain tomorrow? Yes or no? #pause
  - Given a camera image, what color is this object? Yellow, blue, red? #pause

  Let us start with regression

==
  Today, we will come up with a regression problem and then solve it! #pause

  Remember the four steps of machine learning #pause
  + Define an example problem and dataset #pause
  + Define our linear model $f$ #pause
  + Define a loss function $cal(L)$ #pause
  + Find parameters using $cal(L)$ (optimization) #pause

  We will combine these to solve the example problem

= Linear Regression - Example Problem <touying:hidden>

==
  The World Health Organization (WHO) collected data on life expectancy #pause

  #cimage("figures/lecture_2/who.png", height: 60%)

  #text(size: 14pt)[Available for free at #link("https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates/ghe-life-expectancy-and-healthy-life-expectancy")
  ] 

==
  The data comes from roughly 3,000 people from 193 countries #pause

  For each person, they recorded: #pause
  - Home country #pause
  - Alcohol consumption #pause
  - Education #pause
  - Gross domestic product (GDP) of the country #pause
  - Immunizations for Measles and Hepatitis B #pause
  - How long this person lived #pause

We can use this data to make future predictions

==
Since everyone here is very educated, we will focus on how education affects life expectancy #pause

There are studies showing a causal effect of education on health #pause
  - _The causal effects of education on health outcomes in the UK Biobank._ Davies et al. _Nature Human Behaviour_. #pause
  - By staying in school, you are likely to live longer 

==
*Task:* Predict life expectancy #pause

$X = bb(R)_+:$ Years in school #pause

$Y = bb(R)_+:$ Age of death #pause

Each $x in X$ and $y in Y$ represent a single person #pause

*Approach:* Learn the parameters $theta$ such that 

$ f(x, theta) = y; quad x in X, y in Y $ #pause

*Goal:* Given someone's education, predict how long they will live
==

// 40:00 
= Linear Regression - Model <touying:hidden>
==

Soon, $f$ will be a deep neural network #pause

The core of all neural networks are *linear functions* #pause

For now, we let $f$ be a linear function #pause

#align(center, grid(
  columns: 2,
  align: center,
  column-gutter: 2em,
  $ f(x, bold(theta)) = f(x, vec(theta_0, theta_1)) = theta_0 + theta_1 x $,
  cimage("figures/lecture_2/example_regression_graph.png", height: 50%)
)) #pause

Now, we need to find the parameters $bold(theta) = vec(theta_0, theta_1)$ that makes $f(x, bold(theta)) = y$
//   To do this, we will need to find a *loss function* $cal(L)$

= Linear Regression - Loss Function <touying:hidden>
==
  Now, we need to find the parameters $bold(theta) = vec(theta_0, theta_1)$ that make $f(x, bold(theta)) = y$ #pause

  *Question:* How do we choose $bold(theta)$? #pause
  
  *Answer:* $bold(theta)$ that makes $f(x, bold(theta)) = y; quad x in X, y in Y$ #pause

  + Find a loss function #pause
  + Choose $bold(theta)$ with the smallest loss #pause

==
  The loss function computes the loss for the given parameters and data #pause
  
  $ cal(L): X^n times Y^n times Theta |-> bb(R) $ #pause

  // $ cal(L)(x, y, bold(theta)) $ #pause

  The loss function should tell us how close $f(x, bold(theta))$ is to $y$ #pause

  By *minimizing* the loss function, we make $f(x, bold(theta)) = y$ #pause

  There are many possible loss functions, but for regression we often use the *square error* #pause

  $ "error"(y, hat(y)) = (y - hat(y))^2 $
/*
==

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
*/
==
  We can write the loss function for a single datapoint $x_[i], y_[i]$ as

  $ cal(L)(x_[i], y_[i], bold(theta)) = "error"(f(x_[i], bold(theta)),  y_[i]) = (f(x_[i], bold(theta)) - y_[i])^2 $ #pause

  *Question:* Will this $cal(L)$ give us a good prediction for all possible $x$? #pause

  *Answer:* No! We only consider a single datapoint $x_[i], y_[i]$. We want to learn $bold(theta)$ for the entire dataset, for all $x in X, y in Y$

==
  For a single $x_[i], y_[i]$:
  $ 
   cal(L)(x_[i], y_[i], bold(theta)) = "error"(f(x_[i], bold(theta)),  y_[i]) = (f(x_[i], bold(theta)) - y_[i])^2 
  $ #pause

  What about the entire dataset?
  $ bold(x) = mat(x_[1], x_[2], dots, x_[n])^top, bold(y) = mat(y_[1], y_[2], dots, y_[n])^top $ #pause

  #text(size: 22pt)[
    $ 
    cal(L)(bold(x), bold(y), bold(theta)) = sum_(i=1)^n "error"(f(x_[i], bold(theta)),  y_[i]) = sum_(i=1)^n (f(x_[i], bold(theta)) - y_[i])^2 
    $
  ] #pause

  When $cal(L)(bold(x), bold(y), bold(theta))$ is small, then $f(x, bold(theta)) approx y$ for the whole dataset!

// 50 mins

= Linear Regression - Optimization <touying:hidden>
==
  Here is our loss function:

  $ cal(L)(bold(x), bold(y), bold(theta)) = sum_(i=1)^n "error"(f(x_[i], bold(theta)),  y_[i]) = sum_(i=1)^n (f(x_[i], bold(theta)) - y_[i])^2 $ #pause

  We want to find parameters $bold(theta)$ that make the loss small #pause

  We call this search for $bold(theta)$ *optimization* #pause

  Let us state this more formally
==

  Our objective is to *minimize* the loss, using $argmin$ #pause

  #side-by-side[$ argmin_x f(x) $][Find $x$ that makes $f(x)$ smallest] #pause

  #side-by-side[*Question:* $ "What is " argmin_(x) (x + 1)^2 $ ][  #argmin_plot] #pause

  *Answer:* $argmin_(x) (x + 1)^2 = -1$, where $f(x) = 0$


==
  *Optimization objective:* Find the $bold(theta)$ that minimizes the loss #pause
  $ 
   argmin_bold(theta) cal(L)(bold(x), bold(y), bold(theta)) &= argmin_bold(theta) sum_(i=1)^n "error"(f(x_[i], bold(theta)),  y_[i]) \ &= argmin_bold(theta) sum_(i=1)^n (f(x_[i], bold(theta)) - y_[i])^2 
  $  #pause

  How do we optimize $bold(theta)$? #pause

  There is an analytical solution we will derive next lecture #pause

  For now, just follow these steps 

==
  First, we will construct a *design matrix* $overline(bold(X))$ containing input data $x$ #pause

  $ overline(bold(X)) = mat(bold(1), bold(x))  = mat(1, x_[1]; 1, x_[2]; dots.v, dots.v; 1, x_[n]) $

  *Question:* Why two columns? #pause *Hint:* How many parameters? #pause

==
  #side-by-side[
    With our design matrix $overline(bold(X))$ and desired output $bold(y)$,
  ][  
  $ overline(bold(X)) = mat(1, x_[1]; 1, x_[2]; dots.v, dots.v; 1, x_[n]), bold(y) = vec(y_[1], y_[2], dots.v, y_[n]) $
  ] #pause
  
  #side-by-side[
    and our parameters $bold(theta),$
  ][ $ bold(theta) = vec(theta_0, theta_1),  $] #pause

  #v(2em)

  $ argmin_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = (overline(bold(X))^top overline(bold(X)) )^(-1) overline(bold(X))^top bold(y) $

==
  To summarize: #pause

  The $bold(theta)$ given by

  $ bold(theta) = (overline(bold(X))^top overline(bold(X)) )^(-1) overline(bold(X))^top bold(y) $ #pause

  Provide the solution to 
  $ 
   argmin_bold(theta) cal(L)(bold(x), bold(y), bold(theta)) &= argmin_bold(theta) sum_(i=1)^n "error"(f(x_[i], bold(theta)),  y_[i]) \ &= argmin_bold(theta) sum_(i=1)^n (f(x_[i], bold(theta)) - y_[i])^2 
  $ 

==
  Multiply $overline(bold(X))$ and $bold(theta)$ to predict labels #pause

  $ overline(bold(X)) bold(theta) = mat(1, x_[1]; 1, x_[2]; dots.v, dots.v; 1, x_[n]) vec(theta_0, theta_1) = underbrace(vec(#pin(1)theta_0 + theta_1 x_[1]#pin(2), theta_0 + theta_1 x_[2], dots.v, theta_0 + theta_1 x_[n]), y "prediction") $

  #pinit-highlight-equation-from((1,2), (2,2), fill: red, pos: top, height: 1.1em)[$f(x_[1], bold(theta))$] #pause

  We can also evaluate our model for new datapoints #pause

  $ mat(1, x_"Steven") vec(theta_0, theta_1) = underbrace(vec(theta_0 + theta_1 x_"Steven"), y "prediction") $ 
// 55:00, maybe 60:00 with more slides?

= Linear Regression - Example Problem <touying:hidden>
==
  Back to the example... #pause

  *Task:* Predict life expectancy #pause

  $X = bb(R)_+:$ Years in school #pause
  
  $Y = bb(R)_+:$ Age of death #pause

  *Approach:* Learn the parameters $theta$ such that 

  $ f(x, theta) = y; quad x in X, y in Y $ #pause

  *Goal:* Given someone's education, predict how long they will live #pause

  #align(center)[You will be doing this in your first assignment!]

==
  Plot the datapoints $(x_[1], y_[1]), (x_[2], y_[2]), dots $ #pause

  Plot the curve $f(x, bold(theta)) = theta_0 + theta_1 x; quad x in [0, 25]$ #pause

  #cimage("figures/lecture_2/linear_regression.png", height: 60%) 

= Linear Regression - Solve the Problem <touying:hidden>

// 70:00 maybe 75:00

==
  *Goal:* Given someone's education, predict how long they will live #pause

  Plot the datapoints $(x_[1], y_[1]), (x_[2], y_[2]), dots $ #pause

  Plot the curve $f(x, bold(theta)) = theta_0 + theta_1 x; quad x in [0, 25]$ #pause

  #cimage("figures/lecture_2/linear_regression.png", height: 60%) 

==
  #cimage("figures/lecture_2/linear_regression.png", height: 60%) 

  We figured out linear regression! #pause

  But can we do better?

= Polynomial Regression

==
  *Question:* 
  #side-by-side[
    Does the data look linear? 
    #cimage("figures/lecture_2/linear_regression.png", height: 60%) #pause
  ][
    Or maybe more logarithmic? #pause
    #log_plot #pause
  ]

  However, linear regression must be linear! 
==

  *Question:* What does it mean when we say linear regression is linear? #pause

  *Answer:* The function $f(x, theta)$ is a linear function of $x$ and $theta$ #pause

  *Trick:* Change of variables to make $f$ nonlinear: $x_"new" = log(1 + x)$ #pause

  $ overline(bold(X)) = mat(1, x_[1]; 1, x_[2]; dots.v, dots.v; 1, x_[n]) => overline(bold(X)) = mat(1, log(1 + x_[1]); 1, log(1 + x_[2]); dots.v, dots.v; 1, log(1 + x_[n])) $

  Now, $f$ is a linear function of $log(1 + x)$ -- a nonlinear function of $x$!

==


  #side-by-side(columns: (0.3fr, 0.5fr))[
  New design matrix...
  $ overline(bold(X)) = mat(1, log(1 + x_[1]); dots.v, dots.v; 1, log(1 + x_[n])) $][ #pause
  New *nonlinear* function...

     $ overline(bold(X)) bold(theta) = mat(1, log(1 + x_[1]); dots.v, dots.v; 1, log(1 + x_[n])) vec(theta_0, theta_1) = vec(theta_0 + theta_1 x_[1], dots.v, theta_0 + theta_1 x_[n]) $ #pause
  ]
  Same solution...
  $ bold(theta) = (overline(bold(X))^top overline(bold(X)) )^(-1) overline(bold(X))^top bold(y) $

==
  #cimage("figures/lecture_2/log_regression.png", height: 60%) #pause

  Better, but still not perfect #pause

  Can we do even better?

==
  What about polynomials? #pause

  //$ f(x) = a x^m + b x^(m-1) + dots + c x + d $ #pause
  $ f(x) = a + b x + c x^2 + dots + d x^m $ #pause

  Polynomials are *universal function approximators* #pause
  - Can approximate *any* function #pause

  Can we extend linear regression to polynomials?

  //$ f(x, bold(theta)) = theta_m x^m + theta_(m - 1) x^(m-1) + dots + theta_1 + x^1 + theta_0 $

==
  Expand $x$ to a multi-dimensional input space... #pause

  $ overline(bold(X)) = mat(1, x_[1]; 1, x_[2]; dots.v, dots.v; 1, x_[n]) => overline(bold(X)) = mat(
    1, x_[1], x_[1]^2, dots, x^m_[1]; 
    1, x_[2], x_[2]^2, dots, x^m_[2]; 
    dots.v, dots.v, dots.down; 
    1, x_[n], x_[n]^2, dots, x^m_[n]
    ) $ #pause

  Remember, $n$ datapoints and $m + 1$ polynomial terms #pause

  And add some new parameters...
  $ bold(theta) = mat(theta_0, theta_1)^top => bold(theta) =  mat(theta_0, theta_1, dots, theta_m)^top $

==
  New function... #pause
  $ overline(bold(X)) bold(theta) = underbrace(mat(
    1, x_[1], x_[1]^2, dots, x_[1]^m, 1; 
    1, x_[2], x_[2]^2, dots, x_[2]^m, 1; 
    dots.v, dots.v, dots.down; 
    1, x_[n], x_[n]^2, dots, x_[n]^m, 1
    ), n times (m + 1)) 
    underbrace(vec(theta_0, theta_1, dots.v, theta_m), (m + 1) times 1) = 
    underbrace(vec(
      theta_0 + theta_1 x_[1] + dots + theta_(m) x_[1]^(m),
      theta_0 + theta_1 x_[2] + dots + theta_(m) x_[2]^(m),
      dots.v,
      theta_0 + theta_1 x_[n] + dots + theta_(m) x_[n]^(m),
    ), "y prediction, " n times 1)
   $ #pause

  //$ "Polynomial function: " quad f(x, bold(theta)) = theta_m x^m + theta_(m - 1) x^(m-1), dots, theta_1 + x^1 + theta_0 $ #pause

  $ "Same solution... "quad bold(theta) = (overline(bold(X))^top overline(bold(X)) )^(-1) overline(bold(X))^top bold(y) $


==
  $ f(x, bold(theta)) = f(x, vec(theta_0, theta_(1), dots.v, theta_m)) = theta_0 + theta_1 x + dots + theta_m x^m $ #pause

  *Summary:* By changing the input space, we can fit a polynomial to the data using a linear fit!

// TODO: Continue updating here fall 2025
= Overfitting

==
  $ f(x, bold(theta)) = theta_0 + theta_1 x + dots + theta_m x^m $ #pause

  How do we choose $m$ (polynomial order) that provides the best fit? #pause

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
  How do we choose $n$ (polynomial order) that provides the best fit? 

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

  #side-by-side[Pick the $m$ with the smallest loss][
    $ argmin_(bold(theta), m) cal(L)(bold(x), bold(y), (bold(theta), m)) $]

==
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

  *Question:* Which $m$ do you think has the smallest loss? #pause

  *Answer:* $m=5$, but intuitively, $m=5$ does not seem very good...

==
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

  More specifically, $m=5$ will not generalize to new data #pause

  We will only use our model for new data (we already have the $y$ for a known $x$)! #pause

==
  Model has a small loss but does not generalize to new data #pause

  We call this issue *overfitting* #pause

  The model fit too closely to data noise, rather than the trend #pause

  Models that overfit are not useful for making predictions #pause

  Back to the question... #pause
  
  *Question:* How do we choose $m$ such that our polynomial model works for unseen/new data? #pause

  *Answer:* Compute the loss on unseen data!

==
  To compute the loss on unseen data, we will need unseen data #pause

  Let us create some unseen data! #pause

    #cimage("figures/lecture_2/train_test_split.png", height: 60%)

==
  *Question:* How do we choose the training and testing datasets? #pause

  $ "Option 1:" bold(x)_"train" &= vec(
    x_[1], x_[2], x_[3]
  ) bold(y)_"train" &= vec(
    y_[1], y_[2], y_[3]
  ); quad
  bold(x)_"test" &= vec(
    x_[4], x_[5]
  ) bold(y)_"test" &= vec(
    y_[4], y_[5]
  ) 
  $ #pause

  $ "Option 2:" bold(x)_"train" &= vec(
    x_[4], x_[1], x_[3]
  ) bold(y)_"train" &= vec(
    y_[4], y_[1], y_[3]
  ); quad
  bold(x)_"test" &= vec(
    x_[2], x_[5]
  ) bold(y)_"test" &= vec(
    y_[2], y_[5]
  ) 
  $ #pause

  *Answer:* Always shuffle the data #pause

  *Note:* The model must never see the testing dataset during training. This is very important!

==
  We can now measure how the model generalizes to new data #pause

  #cimage("figures/lecture_2/train_test_regression.png", height: 60%)

  Learn parameters from the train dataset, evaluate on the test dataset #pause

  #side-by-side[
    $cal(L)(bold(X)_"train", bold(y)_"train", bold(theta))$
  ][
    $cal(L)(bold(X)_"test", bold(y)_"test", bold(theta))$
  ]

==
  We use separate training and testing datasets on *all* machine learning models, not just linear regression #pause

// 55:00 + 20 break + 40 = 1h55m

/*
==
  *Q:* When should we use test and train splits? #pause

  *Q:* Are neural networks more powerful than linear regression? #pause

  *Q:* Why would we want to use linear regression instead of neural networks? #pause

  *Q:* What are interesting problems that we can apply linear regression to? #pause

  *Q:* We use a squared error loss. What effect does this have on outliers? #pause
*/

= Homework

==
Homework 1 is released, due in two weeks #pause

You will predict life expectancy based on education #pause
- Maybe this convinces you to do a PhD


==
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

==
https://colab.research.google.com/drive/1I6YgapkfaU71RdOotaTPLYdX9WflV1me
