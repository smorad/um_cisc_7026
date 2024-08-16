#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw
#import "common.typ": *


#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 1: Introduction"
)

#title-slide(
  title: [Optimization and Backpropagation],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
  //logo: image("logo.jpg", width: 25%)
)

#slide[
  - Introduce Backpropagation
    - Requirement of differentiable activation functions

  - Introduce SGD
    - Gradient descent
    - Stochastic gradient descent
    - Batch gradient descent

  - Introduce SGD + momentum
]


#slide[
  - Talk about model, how do we train it?
  - Talk about linear regression with closed form solution
    - Review calculus
    - Draw space
    - Quadratic, one maxima
  - Show parameter space and minimization of loss
  - Gradient descent?
  - Talked about how a neural network works
    - Also talked about loss functions?
  - Hebbian learning and biology
  - But how do we learn the parameters that minimize the loss?
  - Use gradient descent
  - Activation function we used before (heaviside) is not continuous
  - Introduce new activation function
]

#slide[
  We previously introduced neural networks as universal function approximators #pause

  A deep neural network $f$ can approximate a function $g$ to infinite precision #pause

  $ f(x, bold(theta)) approx g(x) $ #pause

  $g$ can be a mapping from pictures to text, English to Chinese, etc #pause

  But how do we find the parameters $bold(theta)$ that satisfy this mapping? #pause

  *Goal:* Find a way to determine the parameters $bold(theta)$ of $f$
]

#slide[
  Recall how we solved the linear regression problem #pause

  We minimized a square error loss function #pause

  $ min_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = min_bold(theta) sum_(i=1)^n ( f(x_i, bold(theta)) - y_i )^2  $ #pause

  I then gave you a magical solution to this optimization problem #pause

  $ bold(theta) = (bold(X)^top bold(X) )^(-1) bold(X)^top bold(Y) $ #pause

  Where does this solution come from? Can we do the same for neural networks?
]

#slide[
  The solution for linear regression comes from calculus #pause

  We will briefly review basic calculus concepts
]

#slide[
  We write the *derivative* of a function $f$ with respect to an input $x$ as 

  $ f'(x) = (d f) / (d x) $ #pause

  $ f(x) = x^2 - 3x, quad (d f) / (d x) = 2x - 3  $ #pause

  We can evaluate the derivative at specific points #pause

  $ (d f) / (d x)(1) = 2 dot 1 - 3 = -1 $

]

#slide[
  We can expand the definition of derivative to multivariate functions. We call this the *gradient*

  $ gradient_(bold(x)) f = mat((partial f) / (partial x_1), (partial f) / (partial x_2), dots, (partial f) / (partial x_n))^top $ #pause

  $ f(x_1, x_2) = x_1^2 - 3 x_1 x_2, quad gradient_(x_1, x_2) f = mat(2x_1 + 3 x_2, 3 x_1 ) $ #pause

  We can evaluate the gradient at specific points #pause

  $ (gradient_(x_1, x_2) f) (1, 0) = mat((partial f) / (partial x_1) (1, 0), (partial f) / (partial x_2) (3, 0))^top = mat(2 dot 1 + 3 dot 0, 3 dot 1) = mat(2, 3)^top  $


  
]

#slide[

  In calculus, we can find the extrema of a function $f(x)$ by finding where the derivative or gradient is zero
  
  $ f'(x) = (d f) / (d x) = 0 $ #pause

  //In multivariable calculus, we use the gradient instead

  //$ gradient_(x_1, x_2, dots, x_n) = mat((partial f) / (partial x_1), (partial f) / (partial x_2), dots, (partial f) / (partial x_n))^top $
]

#slide[
  #side-by-side[#cimage("figures/lecture_4/quadratic.png", height: 35%)][$ f(x) = x^2 - 3x $] #pause
  
  #side-by-side[#cimage("figures/lecture_4/quadratic_with_derivative.png", height: 35%)][$ (d f) / (d x) = 2x - 3 $] #pause

  $ 0 = 2x - 3 quad => quad x = 3 / 2 $
]

#slide[
  Now that we remember calculus, let us revisit linear regression #pause

  If we can derive the solution for linear regression, maybe we can apply it to deep neural networks
]

#slide[
  In linear regression, our loss function is 

  $ cal(L)(bold(X), bold(Y), bold(theta)) = sum_(i=1)^n ( f(x_i) - y_i )^2 $

  We can write the square error loss in matrix form as

  $ cal(L)(bold(X), bold(Y), bold(theta)) = ( bold(Y) - bold(X) bold(theta) )^top ( bold(Y) - bold(X) bold(theta) )  $ #pause

  $ cal(L)(bold(X), bold(Y), bold(theta)) = 
  underbrace(underbrace(( bold(Y) - bold(X) bold(theta) )^top, "Linear function of " theta quad)
  underbrace(( bold(Y) - bold(X) bold(theta) ), "Linear function of " theta), "Quadratic function of " theta)  $ #pause
]

#slide[
  $ cal(L)(bold(X), bold(Y), bold(theta)) = 
  underbrace(underbrace(( bold(Y) - bold(X) bold(theta) )^top, "Linear function of " theta quad)
  underbrace(( bold(Y) - bold(X) bold(theta) ), "Linear function of " theta), "Quadratic function of " theta)  $ #pause

  #side-by-side[A quadratic function has a single minima! The minima must be at $gradient_bold(theta) cal(L) = 0$][#cimage("figures/lecture_4/quadratic_parameter_space.png", height: 65%)]
]

#slide[
  For posterity, let us derive the solution to linear regression
  $ gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = gradient_bold(theta) [( bold(Y) - bold(X) bold(theta) )^top ( bold(Y) - bold(X) bold(theta) )] $ #pause

  $ = gradient_bold(theta) [ bold(Y)^top bold(Y) - 2 bold(Y)^top bold(X) bold(theta) + bold(theta)^top bold(X)^top bold(X) bold(theta) ] $ #pause

  $ = -2 bold(X)^top bold(Y) + 2 bold(X)^top bold(X) bold(theta) $ #pause

  Set equal to zero and solve for $bold(theta)$ #pause

  $ bold(X)^top bold(X) bold(theta) = bold(X)^top bold(Y)  $ #pause

  $ bold(theta) = ( bold(X)^top bold(X) )^(-1) bold(X)^top bold(Y)  $ #pause
]

#slide[
  Great! We derived the solution to linear regression #pause
  + Write down the loss function
  + Find the gradient/derivative of the loss function
  + Set the gradient equal to zero
  + Solve for $bold(theta)$

  Let us apply the same approach to neural networks #pause
]

#slide[
  Square error loss again, this time $f$ is a neural network

  $ cal(L)(bold(X), bold(Y), bold(theta)) = sum_(i=1)^n ( f(bold(x)_i, bold(theta)) - bold(y)_i )^2  $ #pause

  Let us just focus on a single $x, y$ each of one dimension (one neuron)

  $ cal(L)(x, y, bold(theta)) = ( f(x, bold(theta)) - y )^2  $
]

#slide[
  $ min_bold(theta) cal(L)(x, y, bold(theta)) = min_bold(theta) ( f(x, bold(theta)) - y )^2  $ #pause

  Write out the neuron fully #pause

  $ min_bold(theta) cal(L)(x, y, bold(theta)) = min_bold(theta) [sigma( theta_0 + theta_1 x ) - y ]^2, quad sigma(x) = cases(0 "if" x <= 0, 1 "if" x > 0) $ #pause

  Let us differentiate and set it equal to zero #pause

  $ 0 = gradient_bold(theta) ([sigma( theta_0 + theta_1 x ) - y ]^2  ), quad sigma(x) = cases(0 "if" x <= 0, 1 "if" x > 0) $ #pause
]

#slide[
  $ 0 = gradient_bold(theta) ([sigma( theta_0 + theta_1 x ) - y ]^2  ), quad sigma(x) = cases(0 "if" x <= 0, 1 "if" x > 0) $ #pause



  *Question:* Does anybody see any issues with computing this gradient? #pause

  + Our current $sigma$ is not differentiable, $sigma'(x) = 0, quad forall x$ #pause
    - We can fix this by choosing a continuous activation function #pause
  + Since $sigma$ is nonlinear (not quadratic), we have many local extrema #pause
  + Since $sigma$ is nonlinear, there is no analytical method to find extrema #pause

  There is no analytical method to find the parameters of a neural network (or even a single neuron)

]

#slide[
  The representational power of neural networks comes from the nonlinear activation function $sigma$ #pause

  Unfortunately, this also means we do not have a closed-form solution to find the parameters $bold(theta)$ #pause

  We need to find another way to find $bold(theta)$ #pause

  Instead, we can use *gradient descent*
]