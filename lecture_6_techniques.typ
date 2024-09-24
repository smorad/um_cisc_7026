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
    [Dirty secret of deep learning], 
    [Optimization is hard],
    [Deeper neural networks],
    [Activation functions],
    [Parameter initialization],
    //[Regularization],
    //[Residual networks],
    [Batch optimization],
    [Modern optimization],
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

#let sigmoid = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 2,
    y-tick-step: none,
    y-ticks: (0, 1),
    y-min: 0,
    y-max: 1,
    {
      plot.add(
        domain: (-5, 5), 
        style: (stroke: (thickness: 5pt, paint: red)),
        label: $ sigma(x) $,
        x => 1 / (1 + calc.pow(2.718, -x)),
      )
      plot.add(
        domain: (-5, 5), 
        style: (stroke: (thickness: 3pt, paint: blue)),
        label: $ gradient sigma(x)$,
        x => (1 / (1 + calc.pow(2.718, -x))) * (1 - 1 / (1 + calc.pow(2.718, -x))),
      )
    })
})}

#let relu = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 2.5,
    //y-tick-step: 1,
    y-tick-step: none,
    y-ticks: (1, 3, 5),
    y-min: 0,
    y-max: 5,
    {
      plot.add(
        domain: (-5, 5), 
        style: (stroke: (thickness: 5pt, paint: red)),
        label: $ sigma(x) $,
        line: (type: "linear"),
        x => calc.max(0, x)
      )
      plot.add(
        domain: (-5, 0), 
        style: (stroke: (thickness: 3pt, paint: blue)),
        x => 0,
      )
      plot.add(
        domain: (0, 5), 
        style: (stroke: (thickness: 3pt, paint: blue)),
        label: $ gradient sigma(x)$,
        x => 1,
      )
    })
})}

#let lrelu = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 2.5,
    //y-tick-step: 1,
    y-tick-step: none,
    y-ticks: (-0.1, 3, 5),
    y-min: -1,
    y-max: 5,
    {
      plot.add(
        domain: (-5, 5), 
        style: (stroke: (thickness: 5pt, paint: red)),
        label: $ sigma(x) $,
        line: (type: "linear"),
        x => calc.max(0.1 * x, x)
      )
      plot.add(
        domain: (-5, 0), 
        style: (stroke: (thickness: 3pt, paint: blue)),
        x => -0.1,
      )
      plot.add(
        domain: (0, 5), 
        style: (stroke: (thickness: 3pt, paint: blue)),
        label: $ gradient sigma(x)$,
        x => 1,
      )
    })
})}

#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 6: Techniques"
)

#title-slide(
  title: [Modern Techniques],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
)

#slide(title: [Agenda])[#agenda(index: none)]
#slide(title: [Agenda])[#agenda(index: 0)]
// TODO: Review
#slide(title: [Agenda])[#agenda(index: 0)]
#slide(title: [Agenda])[#agenda(index: 1)]

#slide(title: [Dirty Secret of Deep Learning])[
    So far, I gave you the impression that deep learning is rigorous #pause

    Biological inspiration, theoretical bounds and mathematical guarantees #pause

    For complex neural networks, deep learning is a *science* not *math* #pause

    There is no widely-accepted theory for why deep neural networks are so effective #pause

    In modern deep learning, we progress using trial and error more often than theory #pause

    Today we experiment, and maybe tomorrow we discover the theory 
]

#slide(title: [Dirty Secret of Deep Learning])[
    Similar to using neural networks for 40 years without knowing how to train them #pause

    #cimage("figures/lecture_1/timeline.svg", width: 90%) #pause

    Are modern networks are too complex for humans to understand?
]

#slide(title: [Dirty Secret of Deep Learning])[
    Scientific method: #pause
        + Collect observations #pause
        + Form hypothesis #pause
        + Run experiment #pause
        + Publish theory #pause

    However, there is a second part: #pause
        + Find theory #pause
        + Find counterexample #pause
        + Publish counterexample
        + Falsify theory

    Deep learning is new, so much of part 2 has not happened yet!

]

#slide(title: [Dirty Secret of Deep Learning])[
    For many concepts, the *observations* are stronger than the *theory* #pause

    Researchers observed that this concept improves many types of neural networks #pause

    Then, they try to create a theory #pause

    Often, these theories are incomplete #pause

    If you do not believe the theory, prove it wrong and be famous! #pause

    Even if we do not agree on *why* a concept works, if we *observe* that it helps, we can still use it #pause

    This is how medicine works (e.g., Anasthetics)!

    //In practice, these concepts usually result in more powerful networks, even if we do not agree on *why* #pause
]

#slide(title: [Agenda])[#agenda(index: 1)]
#slide(title: [Agenda])[#agenda(index: 2)]

#slide(title: [Optimization is Hard])[
    A 2-layer neural network can represent *any* continuous function to arbitrary precision #pause

    $ | f(x, bold(theta)) - g(x) | < epsilon $ #pause

    $ lim_(d_h -> oo) epsilon = 0 $ #pause

    However, finding such $bold(theta)$ is a much harder problem
]

#slide(title: [Optimization is Hard])[
    Gradient descent only guarantees convergence to a *local* optima #pause

    #cimage("figures/lecture_6/poor_minima.png", height: 80%)
]

#slide(title: [Optimization is Hard])[
    #cimage("figures/lecture_6/poor_minima.png", height: 75%) #pause

    Harder tasks can have millions of local optima, and many of the local optima are not very good!
]

#slide(title: [Optimization is Hard])[
    Many of the concepts today create a *flat* loss landscape #pause

    #cimage("figures/lecture_6/skip_connection_img.png", height: 70%)

    Gradient descent reaches a better optimum more quickly in these cases 
]

#slide(title: [Agenda])[#agenda(index: 2)]
#slide(title: [Agenda])[#agenda(index: 3)]

#slide(title: [Deeper Neural Networks])[
    A two-layer neural network is sufficient to approximate any continuous function to arbitrary precision #pause

    But only with infinite width $d_h -> oo$ #pause

    For certain problems, adding one more layer is equivalent to *exponentially* increasing the width
    #text(size: 18pt)[- Eldan, Ronen, and Ohad Shamir. "The power of depth for feedforward neural networks." Conference on learning theory. PMLR, 2016.]


    $ 2 times 32 times 32 => 2^(2 times 32 times 32) approx 10^616; quad "universe has " 10^80 "atoms"$ #pause

    We need more layers for harder problems 
]

#slide(title: [Deeper Neural Networks])[
    In fact, we do not just need *deeper* networks, but also *wider* networks #pause

    The number of neurons in a deep neural network affects the quality of local optima #pause

    From Choromanska, Anna, et al. "The loss surfaces of multilayer networks.": #pause

    - "For large-size networks, most local minima are equivalent and yield similar performance on a test set." #pause

    - "The probability of finding a “bad” (high value) local minimum is non-zero for small-size networks and decreases quickly with network size"
]

#slide(title: [Deeper Neural Networks])[
    To summarize, deeper and wider neural networks tend to produce better results #pause

    Add more layers to your network #pause

    Increase the width of each layer
]

#slide(title: [Deeper Neural Networks])[
    #side-by-side(align: top)[
    ```python
    # Deep neural network
    from torch import nn

    d_x, d_y, d_h = 1, 1, 16
    # Linear(input, output)
    l1 = nn.Linear(d_x, d_h) 
    l2 = nn.Linear(d_h, d_y) 
    ``` #pause
    ][
    ```python
    # Deeper and wider neural network
    from torch import nn
    
    d_x, d_y, d_h = 1, 1, 256
    # Linear(input, output)
    l1 = nn.Linear(d_x, d_h)
    l2 = nn.Linear(d_h, d_h) 
    l3 = nn.Linear(d_h, d_h) 
    ...
    l6 = nn.Linear(d_h, d_y) 
    ```
    ]
]

#slide(title: [Deeper Neural Networks])[
    ```python
    import torch
    d_x, d_y, d_h = 1, 1, 256
    net = torch.nn.Sequential([
        torch.nn.Linear(d_x, d_h),
        torch.nn.Sigmoid(),
        torch.nn.Linear(d_h, d_h),
        torch.nn.Sigmoid(),
        ...
        torch.nn.Linear(d_h, d_y),
    ])

    x = torch.ones((d_x,))
    y = net(x)
    ```
]

#slide(title: [Deeper Neural Networks])[
    ```python
    import jax, equinox
    d_x, d_y, d_h = 1, 1, 256
    net = equinox.nn.Sequential([
        equinox.nn.Linear(d_x, d_h),
        equinox.nn.Lambda(jax.nn.sigmoid), 
        equinox.nn.Linear(d_h, d_h),
        equinox.nn.Lambda(jax.nn.sigmoid), 
        ...
        equinox.nn.Linear(d_h, d_y),
    ])

    x = jax.numpy.ones((d_x,))
    y = net(x)
    ```
]


#slide(title: [Agenda])[#agenda(index: 3)]
#slide(title: [Agenda])[#agenda(index: 4)]

#slide(title: [Activation Functions])[
    The sigmoid function was the standard activation function until \~ 2012 #pause

    #cimage("figures/lecture_1/timeline.svg", width: 90%) #pause

    In 2012, people realized that ReLU activation performed much better
]

#slide(title: [Activation Functions])[
    #side-by-side[#sigmoid #pause][
        The sigmoid function can result in a *vanishing gradient* #pause
        
        $ f(bold(x), bold(theta)) = sigma(bold(theta)_3^top sigma(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x))))) $ #pause
    ] 


    #only(4)[
    $ gradient_bold(theta) f(bold(x), bold(theta)) = 
    gradient [sigma](bold(theta)_3^top sigma(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x)))))

    dot gradient[sigma](bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x))))

    dot gradient[sigma](bold(theta)_1^top overline(bold(x)))
    $ 

    ]

    #only((5,6))[
    $ gradient_bold(theta) f(bold(x), bold(theta)) = 
    underbrace(gradient [sigma](bold(theta)_3^top sigma(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x))))), < 1)

    dot underbrace(gradient[sigma](bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x)))), < 1)

    dot underbrace(gradient[sigma](bold(theta)_1^top overline(bold(x))), < 1)
    $ 
    ]

    #only(6)[
    $ gradient_bold(theta) f(bold(x), bold(theta)) approx 0 $
    ]

]

#slide(title: [Activation Functions])[
    #only((1,2))[To fix the vanishing gradient, researchers use the *rectified linear unit (ReLU)*]
    #side-by-side[$ sigma(x) = max(0, x) \ gradient sigma(x) = cases(0 "if" x < 0, 1 "if" x >= 0) $ #pause][#relu #pause]

    #only((3,4,5))[Looks nothing like a biological neuron]

    #only((4,5,6))[However, it works much better than sigmoid in practice] 

    #only((5,6,7))[Via chain rule, gradient is $1 dot 1 dot 1 dots$ which does not vanish]
    
    #only((6,7,8))[The gradient is constant, resulting in easier optimization] 

    #only((7,8,9))[*Question:* Any problems?]

    #only((8,9,10))[*Answer:* Zero gradient region!]

    #only((9, 10))[Neurons can get "stuck", always output 0]

    #only((10))[These neurons cannot recover, they are *dead neurons*]
]

#slide(title: [Activation Functions])[
    #side-by-side[$ sigma(x) = max(0, x) \ gradient sigma(x) = cases(0 "if" x < 0, 1 "if" x >= 0) $ ][#relu]

    These neurons cannot recover, they are *dead neurons* #pause

    Training for longer results in more dead neurons #pause

    Dead neurons hurt your network!
]

#slide(title: [Activation Functions])[
    To fix dying neurons, use *leaky ReLU* #pause

    #side-by-side[$ sigma(x) = max(0.1 x, x) \ gradient sigma(x) = cases(0.1 "if" x < 0, 1 "if" x >= 0) $ #pause][#lrelu #pause]

    Small negative slope allows dead neurons to recover
]


#slide(title: [Activation Functions])[
    #side-by-side[
    There are other activation functions that are better than leaky ReLU #pause
    - Mish #pause
    - Swish #pause
    - ELU #pause
    - GeLU #pause
    - SeLU #pause
    ][
        #cimage("figures/lecture_6/activations.png")
    ] #pause

    They are all very similar #pause

    I usually use leaky ReLU because it works well enough
]

#slide(title: [Activation Functions])[
    https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

    https://jax.readthedocs.io/en/latest/jax.nn.html#activation-functions
]


#slide(title: [Agenda])[#agenda(index: 4)]
#slide(title: [Agenda])[#agenda(index: 5)]

#slide(title: [Parameter Initialization])[
  Recall the gradient descent algorithm

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

#slide(title: [Parameter Initialization])[
    Initial $bold(theta)$ is starting position for gradient descent #pause

    #cimage("figures/lecture_6/poor_minima.png", height: 70%) #pause

    Pick $bold(theta)$ that results in good local minima
]

#slide(title: [Parameter Initialization])[
    Start simple, initialize all parameters to 0

    $ bold(theta) = mat(
        0, dots, 0; 
        dots.v, dots.down, dots.v;
        0, dots, 0
    ), mat(
        0, dots, 0; 
        dots.v, dots.down, dots.v;
        0, dots, 0
    ), dots $ #pause
    
    *Question:* Any issues? #pause

    *Answer:* The gradient will always be zero

    $ gradient_bold(theta)_1 f = sigma(bold(theta)^top_2 sigma( bold(theta)_1^top overline(bold(x)))) space sigma( bold(theta)_1^top overline(bold(x))) space  overline(bold(x)) $ #pause

    $ gradient_bold(theta)_1 f = sigma(bold(0)^top sigma( bold(theta)_1^top overline(bold(x)))) space sigma( bold(theta)_1^top overline(bold(x))) space  overline(bold(x)) = 0 $
]

#slide(title: [Parameter Initialization])[
    Ok, so initialize $bold(theta) = bold(1)$ #pause

    $ bold(theta) = mat(
        1, dots, 1; 
        dots.v, dots.down, dots.v;
        1, dots, 1
    ), mat(
        1, dots, 1; 
        dots.v, dots.down, dots.v;
        1, dots, 1
    ), dots $ #pause
    
    *Question:* Any issues? #pause

    All neurons in a layer will have the same gradient, and so they will always be the same (useless)

    $ z_i = sigma(sum_(j=1)^d_x theta_j dot overline(x)_j)  = sigma(sum_(j=1)^d_x overline(x)_j) $
]

#slide(title: [Parameter Initialization])[
    $bold(theta)$ must be randomly initialized for neurons

    $ bold(theta) = mat(
        -0.5, dots, 2; 
        dots.v, dots.down, dots.v;
        0.1, dots, 0.6
    ), mat(
        1.3, dots, 1.2; 
        dots.v, dots.down, dots.v;
        -0.8, dots, -1.1
    ), dots $ #pause

    But what scale? If $bold(theta) << 0$ the gradients will vanish to zero, if $bold(theta) >> 0$ the gradients explode to infinity #pause

    Almost everyone initializes following a single paper from 2010: #pause
        - Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." #pause
        - Maybe there are better options?
]

#slide(title: [Parameter Initialization])[
    Here is the magic equation, given the input and output size of the layer is $d_h$

    $ bold(theta) tilde cal(U)[ - sqrt(6) / sqrt(2 d_h), sqrt(6) / sqrt(2 d_h)] $ #pause

    If you have different input or output sizes, such as $d_x, d_y$, then the equation is

    $ bold(theta) tilde cal(U)[ - sqrt(6) / sqrt(d_x + d_y), sqrt(6) / sqrt(d_x + d_y)] $
]

#slide(title: [Parameter Initialization])[
    These equations are designed for ReLU and similar activation functions #pause

    They prevent vanishing or exploding gradients
]


#slide(title: [Parameter Initialization])[
    Usually `torch` and `jax/equinox` will automatically use this initialization when you create `nn.Linear`

    ```python
    layer = nn.Linear(d_x, d_h) # Uses Glorot init
    ```

    You can find many initialization functions at https://pytorch.org/docs/stable/nn.init.html

    For JAX it is https://jax.readthedocs.io/en/latest/jax.nn.initializers.html
]

#slide(title: [Parameter Initialization])[
    ```python

    import torch
    d_h = 10
    # Manually
    theta = torch.zeros((d_h + 1, d_h))
    torch.nn.init.xavier_uniform_(theta) 
    theta = torch.nn.Parameter(theta)

    # Using nn.Linear
    layer = torch.nn.Linear(d_h, d_h)
    # USe .data, to bypass autograd security
    torch.nn.init.xavier_uniform_(layer.weight.data)
    torch.nn.init.xavier_uniform_(layer.bias.data)

    ```
]

#slide(title: [Parameter Initialization])[
    ```python
    import jax
    d_h = 10

    init = jax.nn.initializers.glorot_uniform()
    theta = init(jax.random.key(0), (d_h + 1, d_h))
    ```
]



#slide(title: [Parameter Initialization])[
    ```python
    import jax, equinox
    d_h = 10
    
    layer = equinox.nn.Linear(d_h, d_h, key=jax.random.key(0))
    # Create new bias and weight
    new_weight = init(jax.random.key(1), (d_h, d_h))
    new_bias = init(jax.random.key(2), (d_h,))

    # Use a lambda function to save space
    # tree_at creates a new layer with the new weight
    layer = equinox.tree_at(lambda l: l.weight, layer, new_weight)
    layer = equinox.tree_at(lambda l: l.bias, layer, new_weight)
    ```
]

#slide(title: [Parameter Initialization])[
    Remember, in `equinox` and `torch`, `nn.Linear` will already be initialized correctly! 
]


#slide(title: [Agenda])[#agenda(index: 5)]
#slide(title: [Agenda])[#agenda(index: 6)]

#slide(title: [Batch Optimization])[
    Stochastic gradient descent
]

#slide(title: [Agenda])[#agenda(index: 6)]
#slide(title: [Agenda])[#agenda(index: 7)]


#slide(title: [Modern Optimization])[
    Gradient descent is a powerful tool, but it has issues #pause
    + It can be slow to converge #pause
    + It can get stuck in poor local optima #pause

    Many researchers work on improving gradient descent to converge more quickly, while also preventing premature convergence #pause

    It is hard to teach adaptive optimization through math #pause

    So first, I want to show you a video to prepare you

    https://www.youtube.com/watch?v=MD2fYip6QsQ&t=77s
]

#slide(title: [Modern Optimization])[
    The video simulations provide an intuitive understanding of adaptive optimizers #pause

    The key behind modern optimizers is two concepts:
    - Momentum #pause
    - Adaptive learning rate #pause

    Let us discuss the algorithms more slowly
]

#slide(title: [Modern Optimization])[

    Review gradient descent again, because we will be making changes to it #pause

    #algorithm({
    import algorithmic: *

    Function("Gradient Descent", args: ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$), {

      Cmt[Randomly initialize parameters]
      Assign[$bold(theta)$][$"Glorot"()$] 

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


#slide(title: [Modern Optimization])[
    Introduce *momentum* first #pause

    #algorithm({
    import algorithmic: *

    Function(redm[$"Momentum"$] + " Gradient Descent", args: ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$, redm[$beta$]), {

      Assign[$bold(theta)$][$"Glorot"()$] 
      Assign[#redm[$bold(M)$]][#redm[$bold(0)$] #text(fill:red)[\# Init momentum]]

      For(cond: $i in 1 dots t$, {
        Assign[$bold(J)$][$gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta))$ #text(fill: red)[\# Represents acceleration]]
        Assign[#redm[$bold(M)$]][#redm[$beta dot bold(M) + (1 - beta) dot bold(J)$] #text(fill: red)[\# Momentum and acceleration]]
        Assign[$bold(theta)$][$bold(theta) - alpha #redm[$bold(M)$]$]
      })

    Return[$bold(theta)$]
    })
  })
]

#slide(title: [Modern Optimization])[
    Now *adaptive learning rate* #pause

    #algorithm({
    import algorithmic: *

    Function(redm[$"RMSProp"$], args: ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$, $beta$, redm[$epsilon$]), {

      Assign[$bold(theta)$][$"Glorot"()$] 
      Assign[#redm[$bold(V)$]][#redm[$bold(0)$] #text(fill: red)[\# Init variance]] 

      For(cond: $i in 1 dots t$, {
        Assign[$bold(J)$][$gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta))$ \# Represents acceleration]
        Assign[#redm[$bold(V)$]][#redm[$beta dot bold(V) + (1 - beta) dot bold(J) dot.circle bold(J) $] #text(fill: red)[\# Magnitude]]
        Assign[$bold(theta)$][$bold(theta) - alpha #redm[$bold(J) ⊘ root(dot.circle, bold(V) + epsilon)$]$ #text(fill: red)[\# Rescale grad by prev updates]]
      })

    Return[$bold(theta)$]
    })
  })
]


#slide(title: [Adaptive Optimization])[
    Combine *momentum* and *adaptive learning rate* to create *Adam* #pause

  #algorithm({
    import algorithmic: *

    Function("Adaptive Moment Estimation", args: ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$, greenm[$beta_1$], bluem[$beta_2$], bluem[$epsilon$]), {
      Assign[$bold(theta)$][$"Glorot"()$] 
      Assign[$#greenm[$bold(M)$], #bluem[$bold(V)$]$][$bold(0)$] 

      For(cond: $i in 1 dots t$, {
        Assign[$bold(J)$][$gradient_bold(theta) cal(L)(bold(X), bold(Y), bold(theta))$]
        Assign[#greenm[$bold(M)$]][#greenm[$beta_1 bold(M) + (1 - beta_1) bold(J)$] \# Compute momentum]
        Assign[#bluem[$bold(V)$]][#bluem[$beta_2 dot bold(V) + (1 - beta_2) dot bold(J) dot.circle bold(J)$] \# Magnitude]
        //Assign[$hat(bold(M))$][$bold(M)  "/" (1 - beta_1)$ \# Bias correction]
        //Assign[$hat(bold(V))$][$bold(V) "/" (1 - beta_2)$ \# Bias correction]

        Assign[$bold(theta)$][$bold(theta) - alpha #greenm[$bold(M)$] #bluem[$⊘ root(dot.circle, bold(V) + epsilon)$]$ \# Adaptive param update]
      })

    Return[$bold(theta)$ \# Note, we use biased $bold(M), bold(V)$ for clarity]
    })
  }) 
]

#slide(title: [Adaptive Optimization])[
    ```python
    import torch
    betas = (0.9, 0.999)
    net = ...
    theta = net.parameters()

    sgd = torch.optim.SGD(theta, lr=alpha)
    momentum = torch.optim.SGD(
        theta, lr=alpha, momentum=betas[0]) 
    rmsprop = torch.optim.RMSprop(
        theta, lr=alpha, momentum=betas[1])
    adam = torch.optim.Adam(theta, lr=alpha, betas=betas)
    ...
    sgd.step(), momentum.step(), rmsprop.step(), adam.step()
    ```
]

#slide(title: [Adaptive Optimization])[
    ```python
    import optax
    betas = (0.9, 0.999)
    theta = ...

    sgd = optax.sgd(lr=alpha)
    momentum = optax.sgd(lr=alpha, momentum=betas[0]) 
    rmsprop = optax.rmsprop(lr=alpha, decay=betas[1])
    adam = optax.adam(lr=alpha, b1=betas[0], b2=betas[1])

    v = rmsprop.init(theta)
    theta, v = rmsprop.update(J, v, theta)
    mv = adam.init(theta) # contains M and V
    theta, mv = mv.update(J, mv, theta)
    ```
]

#slide(title: [Agenda])[#agenda(index: 7)]
#slide(title: [Agenda])[#agenda(index: 8)]


// First requirement: break symmetry (explain why)
// Second requirement: normalized/scaled activation/gradients (explain why)