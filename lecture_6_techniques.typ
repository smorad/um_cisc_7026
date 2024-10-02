#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm


// FUTURE TODO: Swap order, modern techniques -> classification

#set math.vec(delim: "[")
#set math.mat(delim: "[")

#let ag = (
  [Review],
  [Dirty Secret of Deep Learning], 
  [Optimization is Hard],
  [Deeper Neural Networks],
  [Activation Functions],
  [Parameter Initialization],
  //[Regularization],
  //[Residual networks],
  [Stochastic Gradient Descent],
  [Modern Optimization],
  [Coding]
)

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


#aslide(ag, none)
#aslide(ag, 0)
// 4:30

#sslide[
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

#sslide[
  *Task:* Given a picture of clothes, predict the text description #pause

  $X: bb(Z)_(0,255)^(32 times 32) #image("figures/lecture_5/classify_input.svg", width: 80%)$ #pause

  $Y : & {"T-shirt, Trouser, Pullover, Dress, Coat,"\ 
    & "Sandal, Shirt, Sneaker, Bag, Ankle boot"}$ #pause

  *Approach:* Learn $bold(theta)$ that produce *conditional probabilities*  #pause

  $ f(bold(x), bold(theta)) = P(bold(y) | bold(x)) = P(vec("T-Shirt", "Trouser", dots.v) mid(|) #image("figures/lecture_5/shirt.png", height: 20%)) = vec(0.2, 0.01, dots.v) $
]

#sslide[
  If events $A, B$ are not disjoint, they are *conditionally dependent* #pause

  $ P("cloud") = 0.2, P("rain") = 0.1 $ #pause

  $ P("rain" | "cloud") = 0.5 $ #pause

  $ P(A | B) = P(A sect B) / P(B) $ #pause

  #side-by-side[Walk outside][
    $P("Rain" sect "Cloud") = 0.1 \ 
    P("Cloud") = 0.2$
    $P("Rain" | "Cloud") = 0.1 / 0.2 = 0.5$
  ]
]

#sslide[
  How can we represent a probability distribution for a neural network? #pause

  $ bold(v) = { vec(v_1, dots.v, v_(d_y)) mid(|) quad sum_(i=1)^(d_y) v_i = 1; quad v_i in (0, 1) } $ #pause

  There is special notation for this vector, called the *simplex* #pause

  $ Delta^(d_y - 1) $
]

#sslide[
  The simplex $Delta^k$ is an $k - 1$-dimensional triangle in $k$-dimensional space #pause

  #cimage("figures/lecture_5/simplex.svg", height: 70%)

  It has only $k - 1$ free variables, because $x_(k) = 1 - sum_(i=1)^(k - 1) x_i$ 
]

#sslide[
  The softmax function maps real numbers to the simplex (probabilities)

  $ "softmax": bb(R)^k |-> Delta^(k - 1) $ #pause

  $ "softmax"(vec(x_1, dots.v, x_k)) = (e^(bold(x))) / (sum_(i=1)^k e^(x_i)) = vec(
    e^(x_1) / (e^(x_1) + e^(x_2) + dots e^(x_k)),
    e^(x_2) / (e^(x_1) + e^(x_2) + dots e^(x_k)),
    dots.v,
    e^(x_k) / (e^(x_1) + e^(x_2) + dots e^(x_k)),
  ) $ #pause

  If we attach it to our linear model, we can output probabilities!

  $ f(bold(x), bold(theta)) = "softmax"(bold(theta)^top bold(x)) $
]

#sslide[
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
]

#sslide[
    #cimage("figures/lecture_5/fashion_mnist_probs.png", height: 80%)
]

#sslide[
  We consider the label $bold(y)_[i]$ as a conditional distribution

  $ P(bold(y)_[i] | bold(x)_[i]) = vec(
    P("Shirt" | #image("figures/lecture_5/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_5/shirt.png", height: 10%))
  ) = vec(1, 0) $ #pause

  What loss function should we use for classification?
]


#sslide[
  Since $f(bold(x), bold(theta))$ and $P(bold(y) | bold(x))$ are both distributions, we want to measure the difference between distributions #pause

  One measurement is the *Kullback-Leibler Divergence (KL)* #pause

  #cimage("figures/lecture_5/forwardkl.png", height: 50%)
]

#sslide[
  From the KL divergence, we derived the *cross-entropy loss* function, which we use for classification #pause

  $ = - sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i $ #pause

  $ cal(L)(bold(x), bold(y), bold(theta)) = [- sum_(j=1)^n sum_(i=1)^(d_y) P(y_([j], i) | bold(x)_[j]) log f(bold(x)_[j], bold(theta))_i ] $ 

]

#sslide[
  Finish coding exercise

  https://colab.research.google.com/drive/1BGMIE2CjlLJOH-D2r9AariPDVgxjWlqG#scrollTo=AnHP-PHVhpW_
]

#aslide(ag, 0)
#aslide(ag, 1)

#sslide[
    So far, I gave you the impression that deep learning is rigorous #pause

    Biological inspiration, theoretical bounds and mathematical guarantees #pause

    For complex neural networks, deep learning is a *science* not *math* #pause

    There is no widely-accepted theory for why deep neural networks are so effective #pause

    In modern deep learning, we progress using trial and error #pause

    Today we experiment, and maybe tomorrow we discover the theory 
]

#sslide[
    Similar to using neural networks for 40 years without knowing how to train them #pause

    #cimage("figures/lecture_1/timeline.svg", width: 90%) #pause

    Are modern networks are too complex for humans to understand?
]

#sslide[
    Scientific method: #pause
        + Collect observations #pause
        + Form hypothesis #pause
        + Run experiment #pause
        + Publish theory #pause

    However, there is a second part: #pause
        + Find theory #pause
        + Find counterexample #pause
        + Publish counterexample #pause
        + Falsify theory

    Deep learning is new, so much of part 2 has not happened yet!

]

#sslide[
    For many concepts, the *observations* are stronger than the *theory* #pause

    Observe that a concept improves many types of neural networks #pause

    Then, try to create a theory #pause

    Often, these theories are incomplete #pause

    If you do not believe the theory, prove it wrong and be famous! #pause

    Even if we do not agree on *why* a concept works, if we *observe* that it helps, we can still use it #pause

    This is how medicine works (e.g., Anesthetics)!
]

#aslide(ag, 1)
#aslide(ag, 2)

#sslide[
    A 2-layer neural network can represent *any* continuous function to arbitrary precision #pause

    $ | f(bold(x), bold(theta)) - g(bold(x)) | < epsilon $ #pause

    $ lim_(d_h -> oo) epsilon = 0 $ #pause

    However, finding such $bold(theta)$ is a much harder problem
]

#sslide[
    Gradient descent only guarantees convergence to a *local* optima #pause

    #cimage("figures/lecture_6/poor_minima.png", height: 80%)
]

#sslide[
    #cimage("figures/lecture_6/poor_minima.png", height: 75%) #pause

    Harder tasks can have millions of local optima, and many of the local optima are not very good!
]

#sslide[
    Many of the concepts today create a *flat* loss landscape #pause

    #cimage("figures/lecture_6/skip_connection_img.png", height: 70%)

    Gradient descent reaches a better optimum more quickly in these cases 
]

#aslide(ag, 2)
#aslide(ag, 3)

#sslide[
    A two-layer neural network is sufficient to approximate any continuous function to arbitrary precision #pause

    But only with infinite width $d_h -> oo$ #pause

    For certain problems, adding one more layer is equivalent to *exponentially* increasing the width
    #text(size: 18pt)[- Eldan, Ronen, and Ohad Shamir. "The power of depth for feedforward neural networks." Conference on learning theory. PMLR, 2016.]


    $ 2 times 32 times 32 => 2^(2 times 32 times 32) approx 10^616; quad "universe has " 10^80 "atoms"$ #pause

    We need more layers for harder problems 
]

#sslide[
    In fact, we do not just need *deeper* networks, but also *wider* networks #pause

    The number of neurons in a deep neural network affects the quality of local optima #pause

    From Choromanska, Anna, et al. "The loss surfaces of multilayer networks.": #pause

    - "For large-size networks, most local minima are equivalent and yield similar performance on a test set." #pause

    - "The probability of finding a “bad” (high value) local minimum is non-zero for small-size networks and decreases quickly with network size"
]

#sslide[
    To summarize, deeper and wider neural networks tend to produce better results #pause

    Add more layers to your network #pause

    Increase the width of each layer
]

#sslide[
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

#sslide[
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

#sslide[
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


#aslide(ag, 3)
#aslide(ag, 4)

#sslide[
    The sigmoid function was the standard activation function until \~ 2012 #pause

    #cimage("figures/lecture_1/timeline.svg", width: 90%) #pause

    In 2012, people realized that ReLU activation performed much better
]

#sslide[
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

#sslide[
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

#sslide[
    #side-by-side[$ sigma(x) = max(0, x) \ gradient sigma(x) = cases(0 "if" x < 0, 1 "if" x >= 0) $ ][#relu]

    These neurons cannot recover, they are *dead neurons* #pause

    Training for longer results in more dead neurons #pause

    Dead neurons hurt your network!
]

#sslide[
    To fix dying neurons, use *leaky ReLU* #pause

    #side-by-side[$ sigma(x) = max(0.1 x, x) \ gradient sigma(x) = cases(0.1 "if" x < 0, 1 "if" x >= 0) $ #pause][#lrelu #pause]

    Small negative slope allows dead neurons to recover
]


#sslide[
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

#sslide[
    https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

    https://jax.readthedocs.io/en/latest/jax.nn.html#activation-functions
]


#aslide(ag, 4)
#aslide(ag, 5)

#sslide[
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

#sslide[
    Initial $bold(theta)$ is starting position for gradient descent #pause

    #cimage("figures/lecture_6/poor_minima.png", height: 70%) #pause

    Pick $bold(theta)$ that results in good local minima
]

#sslide[
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

#sslide[
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

#sslide[
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

#sslide[
    Here is the magic equation, given the input and output size of the layer is $d_h$

    $ bold(theta) tilde cal(U)[ - sqrt(6) / sqrt(2 d_h), sqrt(6) / sqrt(2 d_h)] $ #pause

    If you have different input or output sizes, such as $d_x, d_y$, then the equation is

    $ bold(theta) tilde cal(U)[ - sqrt(6) / sqrt(d_x + d_y), sqrt(6) / sqrt(d_x + d_y)] $
]

#sslide[
    These equations are designed for ReLU and similar activation functions #pause

    They prevent vanishing or exploding gradients
]


#sslide[
    Usually `torch` and `jax/equinox` will automatically use this initialization when you create `nn.Linear`

    ```python
    layer = nn.Linear(d_x, d_h) # Uses Glorot init
    ```

    You can find many initialization functions at https://pytorch.org/docs/stable/nn.init.html

    For JAX it is https://jax.readthedocs.io/en/latest/jax.nn.initializers.html
]

#sslide[
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

#sslide[
    ```python
    import jax
    d_h = 10

    init = jax.nn.initializers.glorot_uniform()
    theta = init(jax.random.key(0), (d_h + 1, d_h))
    ```
]



#sslide[
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

#sslide[
    Remember, in `equinox` and `torch`, `nn.Linear` will already be initialized correctly! 
]


#aslide(ag, 5)
#aslide(ag, 6)

// Now, let us talk a little bit more about optimization
// We computed the gradient over the entire training dataset
// This works with the datasets we have examined so far
// However, some datasets are much larger
// >700 TB Datasets for Large Language Models: A Comprehensive Survey
// We cannot fit this dataset into the memory on colab, so what do we do?
// We introduce stochastic gradient Descent
// Rather than compute the gradient over the dataset
// We approximate the gradient in a stochastic manner
// algorithm






// Really hard problems can have many local optima
// They can also have very large datasets
// We do not like getting stuck in local minima

#sslide[
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
  }) #pause

  Gradient descent computes $gradient cal(L)$ over all $bold(X)$
]

#sslide[
    This works for our small datasets, where $n = 1000$ #pause

    *Question:* How many GB are the LLM datasets? #pause

    *Answer:* About 774,000 GB according to _Datasets for Large Language Models: A Comprehensive Survey_ #pause

    This is just the dataset size, the gradient is orders of magnitude larger 

    $ gradient_bold(theta) cal(L)(bold(x)_[i], bold(y)_[i], bold(theta)) = mat(
        (partial f_1) / (partial x_1), dots, (partial f_ell) / (partial x_1);
        dots.v, dots.down, dots.v;
        (partial f_n) / (partial x_1), dots, (partial f_ell) / (partial x_1);
    )_[i]
    $
]

#sslide[
    *Question:* We do not have enough memory to compute the gradient. What can we do? #pause

    *Answer:*  We approximate the gradient using a subset of the data 

]

#sslide[

    First, we sample random datapoint indices 
    
    $ i, j, k, dots tilde cal(U)[1, n] $ #pause

    Then construct a *batch* of training data

    $ vec(bold(x)_[i], bold(x)_[j], bold(x)_[k], dots.v); quad vec(bold(y)_[i], bold(y)_[j], bold(y)_[k], dots.v) $ #pause

    We call this *stochastic gradient descent*
]

#sslide[
    #algorithm({
    import algorithmic: *

    Function("Stochastic Gradient Descent", args: ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$), {

      Assign[$bold(theta)$][$"Glorot"()$] 

      For(cond: $i in 1 dots t$, {
        Assign[$bold(X), bold(Y)$][$"Shuffle"(bold(X)), "Shuffle"(bold(Y))$]
        For(cond: $j in 0 dots n / B - 1$, {
        Assign[$bold(X)_j$][$mat(bold(x)_[ j B], bold(x)_[ j B + 1], dots, bold(x)_[ (j + 1) B])$]
        Assign[$bold(Y)_j$][$mat(bold(y)_[ j B], bold(y)_[ j B + 1], dots, bold(y)_[ (j + 1) B])$]
        Assign[$bold(J)$][$gradient_bold(theta) cal(L)(bold(X)_j, bold(Y)_j, bold(theta))$]
        Assign[$bold(theta)$][$bold(theta) - alpha bold(J)$]
        })
      })

    Return[$bold(theta)$]
    })
  })
]


#sslide[
    Stochastic gradient descent (SGD) is useful for saving memory #pause

    But it can also improve performance #pause

    Since the "dataset" changes every update, so does the loss manifold #pause

    This makes it less likely we get stuck in bad optima

    #cimage("figures/lecture_5/saddle.png")
]

#sslide[
    There is `torch.utils.data.DataLoader` to help #pause

    ```python
    import torch
    dataloader = torch.utils.data.DataLoader(
        training_data,
        batch_size=32, # How many datapoints to sample
        shuffle=True, # Randomly shuffle each epoch
    )
    for epoch in number_of_epochs:
        for batch in dataloader:
            X_j, Y_j = batch
            loss = L(X_j, Y_j, theta)
            ...
    ```
]



#aslide(ag, 6)
#aslide(ag, 7)


#sslide[
    Gradient descent is a powerful tool, but it has issues #pause
    + It can be slow to converge #pause
    + It can get stuck in poor local optima #pause

    Many researchers work on improving gradient descent to converge more quickly, while also preventing premature convergence #pause

    It is hard to teach adaptive optimization through math #pause

    So first, I want to show you a video to prepare you

    https://www.youtube.com/watch?v=MD2fYip6QsQ&t=77s
]

#sslide[
    The video simulations provide an intuitive understanding of adaptive optimizers #pause

    The key behind modern optimizers is two concepts:
    - Momentum #pause
    - Adaptive learning rate #pause

    Let us discuss the algorithms more slowly
]

#sslide[

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


#sslide[
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

#sslide[
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


#sslide[
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

#sslide[
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

#sslide[
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

#aslide(ag, 7)
#aslide(ag, 8)

#sslide[
  https://colab.research.google.com/drive/1qTNSvB_JEMnMJfcAwsLJTuIlfxa_kyTD#scrollTo=YVkCyz78x4Rp
]