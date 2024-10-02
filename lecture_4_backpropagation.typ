#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

// TODO: Should talk more about forward/backward pass
// TODO: Add break
// TODO: Use different brackets for gradient_theta (f(theta)) to differentiate between products and input/arguments to grad function

#set math.vec(delim: "[")
#set math.mat(delim: "[")
#let ag = (
  [Review],
  [Quiz],
  [Optimization],
  [Calculus review],
  [Deriving linear regression],
  [Gradient descent],
  [Backpropagation],
  [Coding]
)

#let local_optima_plot = canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 2,
    y-tick-step: 30,
    {
      plot.add(
        domain: (-2, 2), 
        x => - 2 * calc.pow(x, 3) - 2 * calc.pow(x, 4) + calc.pow(x, 5) + calc.pow(x, 6),
      )
      plot.add(((-1.4,0), (1.2,-2)),
        mark: "o",
        mark-style: (stroke: none, fill: black),
        style: (stroke: none))
    })
})


#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 1: Introduction"
)

#title-slide(
  title: [Optimization],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
  //logo: image("logo.jpg", width: 25%)
)

#slide(title: [Admin])[
  Quiz 1 grades are on moodle (mean 2.75 / 4) #pause

  When I have a function called $g$ that maps some inputs $a in A, b in B, c in C$ to outputs $d in D, e in E$ I would write

  $ g: A times B times C |-> D times E $ 

  or 

  $ g: A, B, C |-> D, E $ #pause

  In my code, I would write

  $ d, e = g(a, b, c) $
]

#slide(title: [Admin])[
  $bb(R)$ is the set of real numbers #pause

  $5, pi, e, -3.2, 3 / 2, 1.3333...$ are all real numbers #pause

  Unless the number is $sqrt(-1)$, it is probably a real number #pause

  When I write $bb(R)^(d_x)$, this corresponds to $d_x$ real numbers #pause

  So $f: bb(R)^(d_x) |-> bb(R)^(d_y)$ is a function that maps $d_x$ numbers to $d_y$ numbers
]

#slide(title: [Admin])[
  Assignment one should be graded next week #pause

  Today's lecture will be very theoretical, but I must teach it #pause

  I understand the homework might be difficult for some #pause

  Assignment 2 is *optional* (it is *not* required) #pause

  I will give $n$ assignments and you will receive the highest $n - 1$ grades #pause

  E.g., $"asg1" = 60, #redm[$"asg2" = 90$], #redm[$"asg3" = 70$]$, $"total score" =  160 / 200$ #pause

  E.g., $#redm[$"asg1" = 60$], "asg2" = 0, #redm[$"asg3" = 70$]$, $"total score" =  130 / 200$ 
]

/*
#sslide[

  I have two types of students in this class #pause

  + Computer/information science MS and PhD students #pause
  + Students from other departments #pause

  Many computer science students take this course to prepare for PhD research #pause

  They must understand theory to do good research #pause

  Maybe other students care more about application than theory #pause

  It is challenging to teach both groups at once #pause

  I can only change the course if you give me feedback
]
*/

// 4:00

#aslide(ag, none)
#aslide(ag, 0)

#sslide[
  In lecture 1, we assumed a single-input system #pause

  Years of education: $X in bb(R)$ #pause

  But sometimes we want to consider multiple input dimensions #pause

  Years of education, BMI, GDP: $X in bb(R)^3$ #pause

  We can solve these problems using linear regression too
]

#sslide[
  For multivariate problems, we will define the input dimension as $d_x$ #pause

  $ bold(x) in X; quad X in bb(R)^(d_x) $ #pause

  We will write the vectors as

  $ bold(x)_[i] = vec(
    x_([i], 1),
    x_([i], 2),
    dots.v,
    x_([i], d_x)
  ) $
]

#sslide[
  The design matrix for a *multivariate* linear system is

  $ bold(X)_D = mat(
    x_([1], d_x), x_([1], d_x - 1),  dots, x_([1], 1), 1; 
    x_([2], d_x), x_([2], d_x - 1), dots, x_([2], 1), 1; 
    dots.v, dots.v, dots.down, dots.v; 
    x_([n], d_x), x_([n], d_x - 1), dots, x_([n], 1), 1
  ) $ #pause

  The solution is the same as before

  $ bold(theta) = (bold(X)_D^top bold(X)_D )^(-1) bold(X)_D^top bold(y) $ 
]

#sslide[
  We combined *polynomial* and *multivariate* design matrices: #pause

  #side-by-side[
    One-dimensional polynomial functions
    $ bold(X)_D = mat(
      x_[1]^m, x_[1]^(m-1), dots, x_[1], 1; 
      x_[2]^m, x_[2]^(m-1), dots, x_[2], 1; 
      dots.v, dots.v, dots.down; 
      x_[n]^m, x_[n]^(m-1), dots, x_[n], 1
      ) $ #pause][
    Multi-dimensional linear functions
    $ bold(X)_D = mat(
      x_([1], d_x), x_([1], d_x - 1),  dots, 1; 
      x_([2], d_x), x_([2], d_x - 1), dots, 1; 
      dots.v, dots.v, dots.down, dots.v; 
      x_([n], d_x), x_([n], d_x - 1), dots, 1
    ) $ 
  ]
]

#sslide[
  $ bold(X)_D = mat(bold(x)_(D, [1]), dots, bold(x)_(D, [n]))^top $ #pause

  $ &bold(x)_(D, [i]) = \ &mat(
    underbrace(x_([i], d_x)^m x_([i], d_x - 1)^m dots x_([i], 1)^m, (d_x => 1, x^m)),
    underbrace(x_([i], d_x)^m x_([i], d_x - 1)^m dots x_([i], 2)^m, (d_x => 2, x^m)),
    dots,
    underbrace(x_([i], d_x)^(m-1) x_([i], d_x - 1)^(m-1) dots x_([i], 1)^m, (d_x => 1, x^(m-1))),
    dots,
  )
  $ #pause

  The resulting design matrix is too large to solve #pause

  We introduced neural networks because they scale to larger problems
]

#sslide[
  Brains and neural networks rely on *neurons* #pause

  *Brain:* Biological neurons $->$ Biological neural network #pause

  *Computer:* Artificial neurons $->$ Artificial neural network 
]

#sslide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  Neurons send messages based on messages received from other neurons
]

#sslide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  Incoming electrical signals travel along dendrites
]

#sslide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  Electrical charges collect in the Soma (cell body)
]

#sslide[
  #cimage("figures/lecture_3/neuron_anatomy.jpg") 
  The axon outputs an electrical signal to other neurons
]

#sslide[
  How does a neuron decide to send an impulse ("fire")? #pause

  Dendrites form a parallel circuit #pause

  In a parallel circuit, we can sum voltages together #pause

  #side-by-side[Incoming impulses (via dendrites) change the electric potential of the neuron][  #cimage("figures/lecture_3/bio_neuron_activation.png", height: 50%)] #pause


  Many active dendrites will add together and trigger an impulse
]

#sslide[
  We model the neuron "firing" using an activation function $sigma$ #pause

  Last time, we used the heaviside step function as the activation function

  $ sigma(x) = H(x) = cases(0 "if" x <= 0, 1 "if" x > 0) $ #pause

  #cimage("figures/lecture_4/heaviside.svg", height: 50%)
]

#sslide[
  We modeled a neuron mathematically, creating an artificial neuron #pause

  $ f(bold(x), bold(theta)) = sigma(bold(theta)^top overline(bold(x))); quad overline(bold(x)) = vec(1, bold(x)) $ #pause

  $ f(bold(x), vec(b, bold(w))) = sigma(b + bold(w)^top bold(x)) $ #pause

  An artificial neuron is a linear model with an activation function $sigma$ #pause
  
  $ f(bold(x), bold(theta)) = sigma( underbrace(theta_0 1 + theta_1 x_1 + dots + theta_(d_x) x_(d_x), "Linear model") ) $ 
]

#sslide[
  We can represent AND and OR boolean operators using a neuron #pause

  But a neuron cannot represent many other functions #pause

  So we take many neurons and create a neural network #pause

  We discussed *wide* neural networks and *deep* neural networks
]

#sslide[
  How do we express a *wide* neural network mathematically? #pause

  A single neuron:

  $ f: bb(R)^(d_x) times Theta |-> bb(R) $ 
  
  $ Theta in bb(R)^(d_x + 1) $ #pause

  $d_y$ neurons (wide):

  $ f: bb(R)^(d_x) times Theta |-> bb(R)^(d_y) $ 
  
  $ Theta in bb(R)^((d_x + 1) times d_y) $
]

#sslide[
  // Must be m by n (m rows, n cols)
  #text(size: 24pt)[
  For a wide network (also called a layer):
  $ f(vec(x_1, x_2, dots.v, x_(d_x)), mat(theta_(0,1), theta_(0,2), dots, theta_(0,d_y); theta_(1,1), theta_(1,2), dots, theta_(1, d_y); dots.v, dots.v, dots.down, dots.v; theta_(d_x, 1), theta_(d_x, 2), dots, theta_(d_x, d_y)) ) = vec(
    sigma(sum_(i=0)^(d_x) theta_(i,1) overline(x)_i ),
    sigma(sum_(i=0)^(d_x) theta_(i,2) overline(x)_i ),
    dots.v,
    sigma(sum_(i=0)^(d_x) theta_(i,d_y) overline(x)_i ),
  )
  $  

  $ f(bold(x), bold(theta)) = 
    sigma(bold(theta)^top overline(bold(x))); quad bold(theta) in bb(R)^( (d_x + 1) times d_y ) 
  $ #pause

  $
    f(bold(x), vec(bold(b), bold(W))) = sigma( bold(b) + bold(W)^top bold(x) ); quad bold(b) in bb(R)^(d_y), bold(W) in bb(R)^( d_x times d_y ) 
  $
  ]
]

#sslide[
  A *wide* neural network is also called a *layer* #pause

  A layer is a linear operation and an activation function

  $ f(bold(x), vec(bold(b), bold(W))) = sigma(bold(b) + bold(W)^top bold(x)) $

  #side-by-side[Many layers makes a deep neural network][
  #text(size: 22pt)[
    $ bold(z)_1 &= f(bold(x), vec(bold(b)_1, bold(W)_1)) \
    bold(z)_2 &= f(bold(z)_1, vec(bold(b)_2, bold(W)_2)) \ quad bold(y) &= f(bold(z)_2, vec(bold(b)_2, bold(W)_2)) $
  ]
  ]
]


// 18:00
#aslide(ag, 0)
#aslide(ag, 1)
#sslide[
  Put away your phones and laptops #pause

  Take out a paper and pen, write your name and student ID #pause

  I will take away your quiz, give zero points, and refer you to the Dean if: #pause
  - Your phone or laptop is open during the quiz #pause
  - You talk to your neighbor #pause
  - You look at your neighbor's quiz #pause

  After I explain the questions, you will have 15 minutes to finish the quiz
]

#aslide(ag, 1)
#aslide(ag, 2)
#sslide[
  We previously introduced neural networks as universal function approximators #pause

  A deep neural network $f$ can approximate *any* continuous function $g$ to infinite precision #pause

  $ f(x, bold(theta)) = g(x) + epsilon; quad epsilon -> 0 $ #pause

  $g$ can be a mapping from pictures to text, English to Chinese, etc #pause

  But how do we find the $bold(theta)$ that makes $f(x, bold(theta)) = g(x) + epsilon$? #pause

  We said this $bold(theta)$ exists, but never said how to find it #pause

  *Goal:* Find the parameters $bold(theta)$ a neural network
]


#sslide[
  When we search for $bold(theta)$, we call it *optimization* or *training* #pause
  
  We optimize a loss function by computing

  $ argmin_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) $ #pause

  This expression looks very simple, but it can be very hard to evaluate

]

#sslide[
  #cimage("figures/lecture_1/timeline.svg", height: 60%)

  Neural networks were discovered in 1943, but we could not train them until 1982! #pause

  This is why theory is important
]


#sslide[
  Recall how we found $bold(theta)$ in the linear regression problem #pause

  We define the square error loss function #pause

  $ argmin_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = argmin_bold(theta) sum_(i=1)^n ( f(bold(x)_[i], bold(theta)) - bold(y)_[i] )^2  $ #pause

  Then, I gave you a magical solution to this optimization problem #pause

  $ bold(theta) = (bold(X)_D^top bold(X)_D )^(-1) bold(X)_D^top bold(Y) $ #pause

  Where does this solution come from? Can we do the same for neural networks?
]

//22:00 + 15:00 quiz = 37:00

#sslide[
  The solution for linear regression and neural networks comes from *calculus* #pause

  The solution for neural networks also comes from calculus #pause

  Let us review basic calculus concepts
]

#aslide(ag, 2)
#aslide(ag, 3)

#sslide[
  We write the *derivative* of a function $f$ with respect to an input $x$ as 

  $ f'(x) = d / (d x) f = (d f) / (d x) = lim_(h -> 0) (f(x + h) - f(x)) / h $ #pause

  The derivative is the slope of a function #pause

  #cimage("figures/lecture_4/slope.svg", height: 30%) #pause

  $ f(x), #redm[$f'(x=a)$] $
]

//26:00 + 15 == 41:00
#sslide[
  It is easiest if you treat the derivative as a *function of functions* #pause

  $ "derivative"(f) = f' $ #pause

  $ d / (d x): f |-> f' $ #pause

  The derivative takes a function $f$ and outputs a new function $f'$ #pause

  $ d / (d x): [f: X |-> Y] |-> [f': X |-> Y] $
]

#sslide[
  There are formulas for computing the derivative of various operations #pause

  #side-by-side(align: left + horizon)[Constant][$ d / (d x) c = 0 $] #pause

  #side-by-side(align: left + horizon)[Power][$ d / (d x) x^n = n x^(n-1) $] 
]

#sslide[
  #side-by-side(align: left + horizon)[Sum/Difference][$ d / (d x) [f(x) + g(x)] = f'(x) + g'(x) $] #pause

  #side-by-side(align: left + horizon)[Product][$ d / (d x) [f(x) g(x)] = f'(x) g(x) + g'(x) f(x) $] #pause

  #side-by-side(align: left + horizon)[Chain][$ d / (d x) [f (g (x))] = f'(g(x)) g'(x) $]
]

#sslide[
  For example, consider the function

  $ f(x) = x^2 - 3x $ #pause

  We can write the derivative as

  $ d / (d x) [f(x)] = 2x - 3  $ #pause

  We can evaluate the derivative at specific points #pause

  $ d / (d x) [f] (1) = 2 dot 1 - 3 = -1 $
]

#sslide[
  #side-by-side[#cimage("figures/lecture_4/quadratic.png", height: 35%)][$ f(x) = x^2 - 3x $] #pause
  
  #side-by-side[#cimage("figures/lecture_4/quadratic_with_derivative.png", height: 35%)][$ (d f) / (d x) = 2x - 3 $] #pause

  $ 0 = 2x - 3 quad => quad x = 3 / 2 $
]

#sslide[
  We can expand the definition of derivative to multivariate functions. We call this the *gradient*

  $ gradient_(bold(x)) f(mat(x_1, x_2, dots, x_n)^top) = mat((partial f) / (partial x_1), (partial f) / (partial x_2), dots, (partial f) / (partial x_n))^top $ #pause

  Partial derivatives behave similarly to standard derivatives #pause

  $ (partial) / (partial x_1) f(x_1, dots, x_n) approx d / (d x_1) f(x_1, dots, x_n) $ #pause

  When computing $partial / (partial x_i) f(x_1, dots, x_n)$, we treat $x_1, dots, x_(i-1), x_(i+1), dots, x_n$ as constant
]

#sslide[
  For example, consider the function
  $ f(x_1, x_2) = x_1^2 - 3 x_1 x_2 $ #pause

  We can write the gradient as 
  $ gradient_bold(x) f(bold(x)) = 
    gradient_(x_1, x_2) f(vec(x_1, x_2)) =
   vec(
    partial  / (partial x_1) f(x_1, x_2), 
    partial  / (partial x_2) f(x_1, x_2)
  ) = vec(
    2 x_1 - 3 x_2,
    -3 x_1
  ) $ 
]

#sslide[
  $ gradient_bold(x) f(bold(x)) = 
    gradient_(x_1, x_2) f(vec(x_1, x_2)) =
   vec(
    partial  / (partial x_1) f(x_1, x_2), 
    partial  / (partial x_2) f(x_1, x_2)
  ) = vec(
    2 x_1 - 3 x_2,
    -3 x_1
  ) $ #pause

  We can evaluate the gradient at specific points #pause

  $ gradient_bold(x) f(vec(1, 0)) = 
    gradient_(x_1, x_2) f(vec(1, 0)) =
   vec(
    partial  / (partial x_1) f(1, 0), 
    partial  / (partial x_2) f(1, 0)
  ) = vec(
    2 dot 1 - 3 dot 0,
    -3 dot 1
  ) = vec(-1, -3) $ 
]
// 54:00 with quiz
#sslide[
  In calculus, we can find the local extrema of a function $f(x)$ by finding where the derivative is zero
  
  $ f'(x) = d / (d x) f (x)= 0 $ #pause

  With a multivariate function, the extrema lies where the gradient is zero

  //In multivariable calculus, we use the gradient instead

  $ gradient_bold(x) f(bold(x)) = mat((partial f) / (partial x_1), (partial f) / (partial x_2), dots, (partial f) / (partial x_n))^top = mat(0, 0, dots, 0)^top $
]

#aslide(ag, 3)
#aslide(ag, 4)

#sslide[
  Now that we remember calculus, let us revisit linear regression #pause

  If we can derive the solution for linear regression, maybe we can apply it to deep neural networks
]


#sslide[
  In linear regression, our loss function is 

  $ cal(L)(bold(X), bold(Y), bold(theta)) = sum_(i=1)^n ( f(bold(x)_[i], bold(theta)) - bold(y)_[i] )^2 $ #pause

  We can write the square error loss in matrix form as

  $ cal(L)(bold(X), bold(Y), bold(theta)) = ( bold(Y) - bold(X)_D bold(theta) )^top ( bold(Y) - bold(X)_D bold(theta) )  $ #pause

  $ cal(L)(bold(X), bold(Y), bold(theta)) = 
  underbrace(underbrace(( bold(Y) - bold(X)_D bold(theta) )^top, "Linear function of " theta quad)
  underbrace(( bold(Y) - bold(X)_D bold(theta) ), "Linear function of " theta), "Quadratic function of " theta)  $ 
]

#sslide[
  $ cal(L)(bold(X), bold(Y), bold(theta)) = 
  underbrace(underbrace(( bold(Y) - bold(X)_D bold(theta) )^top, "Linear function of " theta quad)
  underbrace(( bold(Y) - bold(X)_D bold(theta) ), "Linear function of " theta), "Quadratic function of " theta)  $ #pause

  #side-by-side[A quadratic function has a single minima! The minima must be at $gradient_bold(theta) cal(L) = 0$][#cimage("figures/lecture_4/quadratic_parameter_space.png", height: 65%)]
]

#sslide[
  Therefore, we know that the $bold(theta)$ that solves

  $ gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = 0 $ #pause

  Also solves
  
  $ argmin_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) $

]

#sslide[
  Using calculus, let us derive the solution to linear regression #pause

  $ cal(L)(bold(X), bold(Y), bold(theta)) = ( bold(Y) - bold(X)_D bold(theta) )^top ( bold(Y) - bold(X)_D bold(theta) ) $ #pause

  $ gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = gradient_bold(theta) [( bold(Y) - bold(X)_D bold(theta) )^top ( bold(Y) - bold(X)_D bold(theta) )] $ #pause

  $ = gradient_bold(theta) [bold(Y)^top bold(Y) - bold(Y)^top bold(X)_D bold(theta) - (bold(X)_D bold(theta))^top bold(Y) + (bold(X)_D bold(theta))^top bold(X)_D bold(theta)] $ #pause

  $ = bold(0) - bold(Y)^top bold(X)_D bold(I) - (bold(X)_D bold(I))^top bold(Y) + (bold(X)_D bold(I))^top bold(X)_D bold(theta) + (bold(X)_D bold(theta))^top bold(X)_D bold(I) $
]
#sslide[
  $ = bold(0) - bold(Y)^top bold(X)_D bold(I) - (bold(X)_D bold(I))^top bold(Y) + (bold(X)_D bold(I))^top bold(X)_D bold(theta) + (bold(X)_D bold(theta))^top bold(X)_D bold(I) $ #pause

  $ = - bold(Y)^top bold(X)_D - bold(X)_D^top bold(Y) + bold(X)_D^top bold(X)_D bold(theta) + (bold(X)_D bold(theta))^top bold(X)_D $ #pause
  
  Remember, $(bold(A B))^top = bold(B)^top bold(A)^top$, and so $bold(Y)^top bold(X)_D = bold(Y)^top (bold(X)_D^top)^top = bold(X)_D^top bold(Y)$ #pause

  $ = - bold(Y)^top bold(X)_D - bold(Y)^top bold(X)_D + bold(X)_D^top bold(X)_D bold(theta) + bold(X)_D^top bold(X)_D bold(theta) $ #pause

  $ = - 2 bold(X)_D^top bold(Y) + 2 bold(X)_D^top bold(X theta) $ #pause

  And so, the gradient of the loss is

  $ gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = - 2 bold(X)_D^top bold(Y) + 2 bold(X)_D^top bold(X)_D bold(theta) $
]
#sslide[
  $ gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = - 2 bold(X)_D^top bold(Y) + 2 bold(X)_D^top bold(X)_D bold(theta) $ #pause

  We want to find the $bold(theta)$ that makes the gradient of the loss zero #pause

  $ bold(0) = - 2 bold(X)_D^top bold(Y) + 2 bold(X)_D^top bold(X)_D bold(theta) $  #pause

  $ 2 bold(X)_D^top bold(Y) = 2 bold(X)_D^top bold(X)_D bold(theta) $  #pause

  $ bold(X)_D^top bold(Y) = bold(X)_D^top bold(X theta) $ #pause

  $ (bold(X)_D^top bold(X)_D)^(-1) bold(X)_D^top bold(Y) = bold(theta) $
]

#sslide[
  $ (bold(X)_D^top bold(X)_D)^(-1) bold(X)_D^top bold(Y) = bold(theta) $ #pause

  This was the "magic" solution I gave you for linear regression #pause

  $ bold(theta) = (bold(X)_D^top bold(X)_D)^(-1) bold(X)_D^top bold(Y) $ 
]

// 68:00 with quiz

#sslide[
  Great! We derived the solution to linear regression #pause

  Now, we will do the same approach for neural networks #pause

  To make it simple, we assume $d_x = 1, d_y = 1, n = 1$ #pause

  One input dimension, one output dimension, one datapoint #pause

  *Step 1*: Write the loss function for a neural network
]

#sslide[
  Like linear regression, we can use square error for a neural network #pause

  $ cal(L)(x, y, bold(theta)) = (f(x, bold(theta)) - y)^2 $ #pause

  All that changes is the model $f$ #pause

  #side-by-side[Linear regression:][$ f(x, y, bold(theta)) = theta_0 + theta_1 x $] #pause

  #side-by-side[Perceptron:][$ f(x, y, bold(theta)) = sigma(theta_0 + theta_1 x) $] 
]

#sslide[
  #side-by-side[$ cal(L)(x, y, bold(theta)) = (f(x, bold(theta)) - y)^2 $][Loss function] #pause
  #side-by-side[$ f(x, bold(theta)) = sigma(theta_0 + theta_1 x) $][Neural network model] #pause

  Now, we plug the model $f$ into the loss function #pause 

  $ cal(L)(x, y, bold(theta)) = (sigma(theta_0 + theta_1 x) - y)^2 $ #pause

  Rewrite #pause

  $ cal(L)(x, y, bold(theta)) = 
  underbrace((sigma(theta_0 + theta_1 x) - y), "Nonlinear function of" theta) 
  underbrace((sigma(theta_0 + theta_1 x) - y), "Nonlinear function of" theta) $
]

// 70:00

#sslide[
  #side-by-side[Linear regression loss function was quadratic with one minima][#cimage("figures/lecture_4/quadratic_parameter_space.png", height: 35%)] #pause

  With a neural network, this is our loss function

  $ cal(L)(x, y, bold(theta)) = 
  underbrace((sigma(theta_0 + theta_1 x) - y), "Nonlinear function of" theta) 
  underbrace((sigma(theta_0 + theta_1 x) - y), "Nonlinear function of" theta) $ #pause

  *Question:* How many minima does this function have? #pause

  *Answer:* We do not know
]

#sslide[
  The nonlinearity/activation function $sigma$ in the neural network means we cannot find an analytical solution #pause

  $ f(x, bold(theta)) = sigma(theta_0 + theta_1 x) $ #pause

  *Question:* Can we remove the activation function $sigma$? #pause

  *Answer:* Yes, but the result is linear regression #pause

  $ f(x, bold(theta)) = theta_0 + theta_1 x $ #pause

  Activation functions make the neural network powerful 
]

#sslide[
  Linear regression: analytical solution for $bold(theta)$ #pause

  Neural network: no analytical solution for $bold(theta)$ #pause

  So how to find $bold(theta)$ for a neural network?
]

#aslide(ag, 4)
#aslide(ag, 5)

#sslide[
  To find $bold(theta)$ for a neural network, we use *gradient descent* #pause

  Gradient descent optimizes *differentiable* functions #pause

  We must be able to take the derivative or gradient of the loss function to use gradient descent #pause

  How does gradient descent work?
]

#sslide[
  #side-by-side[A differentiable loss function produces a manifold][#cimage("figures/lecture_4/parameter_space.png", height: 70%)] #pause
  
  Our goal is to find the lowest point on this manifold #pause

  The lowest point solves $argmin_bold(theta) cal(L)(bold(X), bold(Y), bold(theta))$
]

#sslide[
  *Note:* Gradient descent provides a *local* optima, not necessarily a *global* optima #pause

  #align(center, local_optima_plot) #pause

  In practice, a local optima provides a good enough model
]

#focus-slide[Relax]

#sslide[
  Let us define gradient descent without math #pause

  You are on the top of a mountain and there is lightning storm #pause

  #cimage("figures/lecture_4/lightning.jpg", height: 60%) #pause

  For safety, you should walk down the mountain to escape the lightning
]

#sslide[
  But you do not know the path down! #pause

  #cimage("figures/lecture_4/hiking_slope.jpg", height: 70%)

  You see this, which way do you walk next?
]

#sslide[
  #cimage("figures/lecture_4/hiking_slope.jpg", height: 80%)
  This is gradient descent
]

#sslide[
  In gradient descent, we look at the *slope* of the loss function #pause

  And we walk in the steepest direction #pause

  #cimage("figures/lecture_4/gradient_descent_3d.png", height: 60%) #pause

  And then we repeat 
]


#sslide[
  #cimage("figures/lecture_4/gradient_descent_3d.png", height: 70%) #pause

  We find the gradient $gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta))$ #pause

  And update $bold(theta)$ in the steepest direction
]

#sslide[
  #cimage("figures/lecture_4/gradient_descent_3d.png", height: 70%)

  Eventually, we arrive at the bottom
]

#sslide[
  With gradient descent, the loss function must be differentiable #pause

  If we cannot compute the derivative/gradient, then we do not know which way to walk!
]

#sslide[
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
// 84:00

#sslide[
  #cimage("figures/lecture_4/parameter_space.png", height: 100%)
]

#sslide[
  Two main steps in gradient descent: #pause

  Step 1: Compute the gradient of the loss #pause

  Step 2: Update the parameters using the gradient #pause

  Let us start with step 1
]

#aslide(ag, 5)
#aslide(ag, 6)

#sslide[
  *Goal:* Compute the gradient of the loss $gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta))$ #pause

  We call this process *backpropagation* #pause

  We propagate errors from the loss function *backward* through each layer of the neural network #pause

  #cimage("figures/lecture_1/optimizer_nn.png")


  //Let us propagate errors through a deep neural network
]

#sslide[
  Forward propagation 
  #cimage("figures/lecture_4/forward.svg", height: 80%)
]
#sslide[
  Backward propagation
  #cimage("figures/lecture_4/backward.svg", height: 80%)
]

#sslide[
  Finding the gradient is necessary to use gradient descent! #pause

  First, we will find the gradient of a neural network layer #pause

  Then, we will find the gradient of a deep neural network #pause

  Finally, we will find the gradient of the loss function
]

#sslide[
  Start with the equation of a neural network layer 

  $ f(bold(x), bold(theta)) = sigma( bold(theta)^top overline(bold(x)) ) $ #pause

  Take the gradient of both sides 
  $ gradient_bold(theta) f(bold(x), bold(theta)) = gradient_bold(theta) sigma( bold(theta)^top overline(bold(x)) ) $ #pause

  $ "Chain: " d / (d x) f(g(x)) = f'(g(x)) dot g'(x) $ #pause

  $ gradient_(bold(theta)) f(bold(x), bold(theta)) = gradient_bold(theta) [sigma] (bold(theta)^top overline(bold(x))) dot gradient_(bold(theta)) (bold(theta)^top overline(bold(x))) $ 
]

#sslide[
  $ gradient_(bold(theta)) f(bold(x), bold(theta)) = gradient_bold(theta) [sigma] (bold(theta)^top overline(bold(x))) dot gradient_(bold(theta)) (bold(theta)^top overline(bold(x))) $ 

  $ "What is " gradient_bold(theta) sigma "?" $ #pause

  #cimage("figures/lecture_4/heaviside.svg") #pause

  Derivative is zero everywhere and infinity at $x=0$, so the derivative for a layer is either infinity or zero 
]

#sslide[
  We use a differentiable approximation of the heaviside step function #pause

  $ sigma(z) = 1 / (1 + e^(-z)) $ #pause

  We call this approximation the *sigmoid function* #pause

  #side-by-side[#cimage("figures/lecture_4/heaviside.svg")][#cimage("figures/lecture_4/sigmoid.svg")]

  The sigmoid function has finite and nonzero derivative everywhere
]

#sslide[
  #side-by-side[#cimage("figures/lecture_4/heaviside.svg")][#cimage("figures/lecture_4/sigmoid.svg")][ $ sigma(z) = 1 / (1 + e^(-z)) $]

  The derivative of the sigmoid function is

  $ d / (d z) sigma(z) = sigma(z) dot (1 - sigma(z)) $ #pause

  $ gradient_bold(z) sigma(bold(z)) = sigma(bold(z)) dot.circle (1 - sigma(bold(z))) $ 
]

// 97:00

#sslide[
  Back to our layer #pause

  $ gradient_(bold(theta)) f(bold(x), bold(theta)) = gradient_bold(theta) [sigma] (bold(theta)^top overline(bold(x))) dot gradient_(bold(theta)) (bold(theta)^top overline(bold(x))) $ #pause

  Plug in the gradient of our new activation function 
  
  $ gradient_bold(z) sigma(bold(z)) = sigma(bold(z)) dot.circle (1 - sigma(bold(z))) $ #pause

  $ gradient_(bold(theta)) f(bold(x), bold(theta)) = (sigma(bold(theta)^top overline(bold(x))) dot.circle (1 - sigma(bold(theta)^top overline(bold(x))))) gradient_(bold(theta)) (bold(theta)^top overline(bold(x))) $ #pause

  Evalute the final term #pause

  $ gradient_(bold(theta)) f(bold(x), bold(theta)) = (sigma(bold(theta)^top overline(bold(x))) dot.circle (1 - sigma(bold(theta)^top overline(bold(x))))) overline(bold(x))^top $ 
]

#sslide[
  $ gradient_(bold(theta)) f(bold(x), bold(theta)) = (sigma(bold(theta)^top overline(bold(x))) dot.circle (1 - sigma(bold(theta)^top overline(bold(x))))) overline(bold(x))^top $ #pause

  This is the gradient for the layer of a neural network! #pause

  We will use this to compute the gradient of a deep neural network
]

#sslide[
  Recall the deep neural network has many layers

    $ f_1(bold(x), bold(phi)) = sigma(bold(phi)^top overline(bold(x))) quad

    f_2(bold(x), bold(psi)) = sigma(bold(psi)^top overline(bold(x))) quad

    dots quad

    f_(ell)(bold(x), bold(xi)) = sigma(bold(xi)^top overline(bold(x))) $ #pause

    And that we call them in series

    $ bold(z)_1 &= f_1(bold(x), bold(phi)) \ 
      bold(z)_2 &= f_2(bold(z)_1, bold(psi)) \
      dots.v \
      bold(z)_(ell) &= f_(ell)(bold(z)_(ell - 1), bold(xi))
    $ #pause

  //$ f(bold(x), bold(theta)) = f_(ell) (dots f_2(f_1(bold(x), bold(phi)), bold(psi)) dots bold(xi) ) $
]

#sslide[
  #side-by-side[Take the gradient of both sides][
  $  gradient_(bold(phi), bold(psi), dots, bold(xi)) bold(z)_1 &=  gradient_(bold(phi), bold(psi), bold(xi)) f_1(bold(x), bold(phi)) \ 
  gradient_(bold(phi), bold(psi), dots, bold(xi)) bold(z)_2 &=  gradient_(bold(phi), bold(psi), bold(xi)) f_2(bold(z)_1, bold(psi)) \
  dots.v \
   gradient_(bold(phi), bold(psi), dots, bold(xi)) bold(z)_(ell) &=  gradient_(bold(phi), bold(psi), bold(xi)) f_(ell)(bold(z)_(ell - 1), bold(xi))
  $] #pause

  #v(1em)

  #side-by-side[Each layer only uses one set of parameters][

  $  gradient_(bold(phi)) bold(z)_1 &=  gradient_(bold(phi)) f_1(bold(x), bold(phi)) \ 
  gradient_(bold(psi)) bold(z)_2 &=  gradient_(bold(psi)) f_2(bold(z)_1, bold(psi)) \
  dots.v \
  gradient_(bold(xi)) bold(z)_(ell) &=  gradient_(bold(xi)) f_(ell)(bold(z)_(ell - 1), bold(xi))
  $] #pause
]

#sslide[
  The gradient of a deep neural network is 

  $ 
  gradient_(bold(theta)) f(bold(x), bold(theta)) = gradient_(bold(phi), bold(psi), dots, bold(xi)) f(bold(x), mat(bold(phi), bold(psi), dots, bold(xi))^top) = vec(
    gradient_bold(phi) f_1(bold(x), bold(phi)), 
    gradient_(bold(psi)) f_2(bold(z)_1, bold(psi)), 
    dots.v, 
    gradient_(bold(xi)) f_(ell)(bold(z)_(ell - 1), bold(xi))
  ) 
  $ #pause

  Where each layer gradient is 

  $ gradient_(bold(xi)) f_ell (bold(z)_(ell - 1), bold(xi)) = (sigma(bold(xi)^top overline(bold(z))_(ell - 1)) dot.circle (1 - sigma(bold(xi)^top overline(bold(z))_(ell - 1)))) overline(bold(z))_(ell - 1)^top $ 
]

#sslide[
  We computed the gradient of a neural network layer #pause

  We computed the gradient of the neural network #pause

  Now, we must compute gradient of the loss function #pause
]

#sslide[
  $ cal(L)(bold(X), bold(Y), bold(theta)) = sum_(i = 1)^n (f(bold(x)_[i], bold(theta)) - bold(y)_[i])^2 $

  $ gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = gradient_bold(theta) sum_(i = 1)^n (f(bold(x)_[i], bold(theta)) - bold(y)_[i])^2 $ #pause

  //Use the sum rule $d / (d x) (f(x) + g(x)) = f'(x) + g'(x)$

  $ gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = sum_(i = 1)^n gradient_bold(theta) ( f(bold(x)_[i], bold(theta)) - bold(y)_[i] )^2 $ #pause

  $ gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = sum_(i = 1)^n 2 ( f(bold(x)_[i], bold(theta)) - bold(y)_[i]) gradient_bold(theta) f(bold(x)_[i], bold(theta)) $ 
]

#sslide[
  To summarize: #pause

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

#sslide[
  *Question:* Why did we spend all this time deriving gradients? #pause

  *Answer:* The gradient is necessary for gradient descent #pause

  #algorithm({
    import algorithmic: *

      For(cond: $i in 1 dots t$, {
        Cmt[Compute the gradient of the loss]        
        Assign[$bold(J)$][$gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta))$]
        Cmt[Update the parameters using the negative gradient]
        Assign[$bold(theta)$][$bold(theta) - alpha bold(J)$]
      })
    }) #pause

    $ bold(theta)_(t + 1) = bold(theta)_t - alpha gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)_t) $ 
]

#aslide(ag, 6)
#aslide(ag, 7)

#slide[
  How do gradients work in `jax` or `torch`? #pause

  The libraries compute the gradients using *autograd* #pause


  Autograd differentiates nested functions using the chain rule #pause

  $ gradient_bold(theta) f(g(h(x, bold(theta)))) = f'(g(h(x, bold(theta)))) dot g'(h(x, bold(theta))) dot h'(x, bold(theta)) $ #pause

  Engineers derived gradients for hundreds of functions $f, g, h, dots$ #pause

  Researchers derive their own analytical gradients like we did today #pause

  Now, let us look at `jax` and `torch` optimization code
]

#slide[

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

#slide[
  ```python
  import torch
  optimizer = torch.optim.SGD(lr=0.0001)

  def L(model, X, Y):
    ...
  # Pytorch will record a graph of all operations
  # Everytime you do theta @ x, it stores inputs and outputs 
  loss = L(X, Y, model) # compute gradient
  # Traverse the graph backward and compute the gradient
  loss.backward() # Sets .grad attribute on each parameter
  optimizer.step() # Update the parameters using .grad
  optimizer.zero_grad() # Set .grad to zero, DO NOT FORGET!!
  ```
]

#slide[
  Time for some interactive coding

  https://colab.research.google.com/drive/1W8WVZ8n_9yJCcOqkPVURp_wJUx3EQc5w
]