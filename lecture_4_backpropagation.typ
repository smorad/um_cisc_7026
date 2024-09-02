#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

#set math.vec(delim: "[")
#set math.mat(delim: "[")
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
  title: [Optimization and Backpropagation],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
  //logo: image("logo.jpg", width: 25%)
)

#slide[
  + Review
  + Optimization
  + Calculus review
  + Deriving linear regression
  + Gradient descent
  + Autograd
]

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

  $ argmin_bold(theta) cal(L)(bold(X), bold(Y), bold(theta)) = argmin_bold(theta) sum_(i=1)^n ( f(x_i, bold(theta)) - y_i )^2  $ #pause

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

  $ argmin_bold(theta) cal(L)(x, y, bold(theta)) = argmin_bold(theta) [sigma( theta_0 + theta_1 x ) - y ]^2, quad sigma(x) = cases(0 "if" x <= 0, 1 "if" x > 0) $ #pause

  Let us differentiate and set it equal to zero #pause

  $ 0 = gradient_bold(theta) ([sigma( theta_0 + theta_1 x ) - y ]^2  ), quad sigma(x) = cases(0 "if" x <= 0, 1 "if" x > 0) $
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

  We will use *gradient descent* to find $bold(theta)$
]

#slide[
  Gradient descent is an optimization method for *differentiable* functions #pause

  More formally, gradient descent approximates the $theta$ that solves

  $ argmin_theta cal(L)(x, y, theta) $ #pause

]

#slide[

  Gradient descent provides a *local* optima, not necessarily a *global* optima #pause

  #align(center, local_optima_plot) #pause

  In practice, a local optima provides a good enough model


  //Given a loss function, we can take a step in the direction of the *negative gradient* to decrease the loss #pause
]

#slide[
  Gradient descent relies on the *gradient* of $cal(L)$, therefore $cal(L)$ must be differentiable #pause

  In gradient descent, we update the parameters in the direction of the negative gradient of the loss 
]

#slide[
  The gradient descent algorithm is as follows:

  #algorithm({
    import algorithmic: *

    Function("Gradient Descent", args: ($bold(x)$, $bold(y)$, $cal(L)$, $t$, $alpha$), {

      Cmt[Randomly initialize parameters]
      Assign[$bold(theta)$][$cal(N)(0, 1)$] 

      For(cond: $i in 1 dots t$, {
        Cmt[Compute the gradient of the loss]        
        Assign[$bold(J)$][$partial cal(L)(bold(x), bold(y), bold(theta)) "/ " partial bold(theta)$]
        Cmt[Update the parameters using the negative gradient]
        Assign[$bold(theta)$][$bold(theta) - alpha bold(J)$]
      })

    Return[$bold(theta)$]
    })
  })
]

#slide[
  We can visualize gradient descent of a bivariate function #pause

  #cimage("figures/lecture_4/gradient_descent_3d.png", height: 80%)
]

#slide[
  Let us do a very simple example #pause

  *Goal:* Solve $argmin_bold(theta) cal(L)(x, y, bold(theta)) = theta_1^2 x + theta_0$ using gradient descent

  *Dataset:* One item in the dataset, $x = 1, y = 2$

  1. Randomly initialize $bold(theta)$ #pause

  $ theta = mat(2.5, 1.1)^top $ #pause

  2. Compute the gradient for our dataset

  $ J = (partial cal(L)(x, y, theta)) / (partial theta) $
]

#slide[
  #side-by-side[$ bold(J) = (partial cal(L)(x, y, theta)) / (partial theta) $][From the last slide] #pause

  #side-by-side[$ bold(J) = mat(partial / (partial theta_1) cal(L)(x, y, theta), partial / (partial theta_0) cal(L)(x, y, theta) )^top $][Write out gradient w.r.t. all parameters] #pause

  #side-by-side[$ bold(J) = mat(partial / (partial theta_1) (theta_1^2 x + theta_0), partial / (partial theta_0) (theta_1^2 x + theta_0))^top $][Plug in $cal(L)$]


  #side-by-side[$ bold(J) = mat(2 theta_1 x, 1)^top $][Differentiate]

  #side-by-side[$ bold(J) = mat(2 dot 2.5 dot 1, 1)^top = mat(5, 1)^top $][Plug in $x, y, bold(theta)$]
]

#slide[

  #text(size: 24pt)[
  3. Update $bold(theta)$ using $bold(J)$

  #side-by-side[$ bold(theta) <- bold(theta) - alpha bold(J) $][From algorithm]

  #side-by-side[$ vec(theta_1, theta_0) <- vec(2.5, 1.1) - alpha vec(5, 1) $][Plug in $bold(theta)$ and $bold(J)$]

  #side-by-side[$ vec(theta_1, theta_0) <- vec(2.5, 1.1) - 0.1 vec(5, 1) $][Let $alpha = 0.1$]

  #side-by-side[$ vec(theta_1, theta_0) <- vec(2, 1) $][Evaluate]
  ]
]

#slide[
  4. Repeat this process until convergence (loss no longer decreases)

  $ bold(J) &= (partial cal(L)(x, y, bold(theta))) / (partial bold(theta)) \

  bold(theta) & <- bold(theta) - alpha bold(J)
  $
]

#slide[
  #cimage("figures/lecture_4/gradient_descent_3d.png", height: 80%)
]

#slide[
  TODO: Summary
]

#slide[
  The are different types of gradient descent, depending on how much training data we use:
 
  *Batch gradient descent:* uses the entire dataset $cal(L)(vec(bold(x)_1, dots.v, bold(x)_n), vec(bold(y)_1, dots.v, bold(y)_n), bold(theta))$

  *Stochastic gradient descent:* Gradient descent over one datapoint at a time $cal(L)(bold(x)_i, bold(y)_i, bold(theta))$

  *Minibatch gradient descent:* Gradient descent over many (but not all) datapoints$cal(L)(vec(bold(x)_i, dots.v, bold(x)_j), vec(bold(y)_i, dots.v, bold(y)_j), bold(theta))$
]

#slide[
  In practice, we do not have enough computer memory for batch gradient descent #pause

  Instead, we use stochastic gradient descent or minibatch gradient descent #pause
]

