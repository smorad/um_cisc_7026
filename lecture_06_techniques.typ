#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.0"
#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "common.typ": *
#import "@preview/algorithmic:1.0.5"
#import algorithmic: style-algorithm, algorithm-figure, algorithm
#import "@preview/mannot:0.3.0": *

#let handout = true


// FUTURE TODO (Fall 2025): Break into two lectures, second lecture just for optimization, more coding
// FUTURE TODO: Swap order, modern techniques -> classification

#set math.vec(delim: "[")
#set math.mat(delim: "[")




// 3:00

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(handout: handout),
  config-info(
    title: [Modern Techniques],
    subtitle: [CISC 7026 - Introduction to Deep Learning],
    author: [Steven Morad],
    institution: [University of Macau],
    logo: image("figures/common/bolt-logo.png", width: 4cm)
  ),
  header-right: none,
  header: self => utils.display-current-heading(level: 1)
)

#title-slide()

== Outline <touying:hidden>

#components.adaptive-columns(
    outline(title: none, indent: 1em, depth: 1)
)

= Admin
==
Thank you for coming! #pause

Silly to have Weds+Thursday holiday but not Friday... #pause

==
Was youtube video ok? #pause

YouTube translations better than realtime translation? #pause

Next year, should I record and upload all lectures?

==
How was exam 1? #pause

Do you feel like experts in: #pause
- Linear regression #pause
- Artificial neurons #pause
- Gradient descent

==
How was assignment 2? #pause

Assignment 3 due next week, did anyone start? #pause
- Should I extend due date? #pause
  - Would need to cancel assignment 4 (Convolutional networks) #pause

Meeting with students on Monday to finish grading #pause
- Exam 1 #pause
- Assignment 1 #pause

The rest of the course is final project #pause
- Focus on your interests

==
How was Yutao's lecture? #pause

Anything you liked/disliked?

= Review
==
  Many problems in ML can be reduced to *regression* or *classification* #pause

  *Regression* asks how many #pause
  - How long will I live? #pause
  - How much rain will there be tomorrow? #pause
  - How far away is this object? #pause

  *Classification* asks which one #pause
  - Is this a dog or muffin? #pause
  - Will it rain tomorrow? Yes or no? #pause
  - What color is this object? #pause

  Last lecture, we defined classification
  
==
  *Task:* Given a picture of clothes, predict the text description #pause

  $X: bb(Z)_(0,255)^(32 times 32) #image("figures/lecture_5/classify_input.svg", width: 80%)$ #pause

  $Y : & {"T-shirt, Trouser, Pullover, Dress, Coat,"\ 
    & "Sandal, Shirt, Sneaker, Bag, Ankle boot"}$ #pause

  *Approach:* Learn $bold(theta)$ that produce *conditional probabilities*  #pause

  $ f(bold(x), bold(theta)) = P(bold(y) | bold(x)) = P(vec("T-Shirt", "Trouser", dots.v) mid(|) #image("figures/lecture_5/shirt.png", height: 20%)) = vec(0.2, 0.01, dots.v) $

==
  If $B$ provides information about $A$, they are *conditionally dependent* #pause

  $ P(A | B) != P(A) $ #pause

  $ P("cloud") = 0.2, P("rain") = 0.1 $ #pause

  $ P("rain" | "cloud") = 0.5 $ #pause

  $ P(A | B) = P(A, B) / P(B) $ #pause

  #side-by-side[Walk outside][
    $P("Rain", "Cloud") = 0.1 \ 
    P("Cloud") = 0.2$
    $P("Rain" | "Cloud") = 0.1 / 0.2 = 0.5$
  ]

==
  How can we represent a probability distribution for a neural network? #pause

  $ bold(v) = { vec(v_1, dots.v, v_(d_y)) mid(|) quad sum_(i=1)^(d_y) v_i = 1; quad v_i in [0, 1] } $ #pause

  There is special notation for this vector, called the *simplex* #pause

  $ Delta^(d_y - 1) $

==
  The simplex $Delta^k$ is an $k - 1$-dimensional triangle in $k$-dimensional space #pause

  #cimage("figures/lecture_5/simplex.svg", height: 70%)

  It has only $k - 1$ free variables, because $x_(k) = 1 - sum_(i=1)^(k - 1) x_i$ 

==
  The softmax function maps real numbers to the simplex (probabilities)

  $ "softmax": bb(R)^k |-> Delta^(k - 1) $ #pause

  $ "softmax"(vec(x_1, dots.v, x_k)) = (e^(bold(x))) / (sum_(i=1)^k e^(x_i)) = vec(
    e^(x_1) / (e^(x_1) + e^(x_2) + dots e^(x_k)),
    e^(x_2) / (e^(x_1) + e^(x_2) + dots e^(x_k)),
    dots.v,
    e^(x_k) / (e^(x_1) + e^(x_2) + dots e^(x_k)),
  ) $ #pause

  If we attach it to our model, we can output probabilities!

  $ f(bold(x), bold(theta)) = "softmax"(bold(theta)^top overline(bold(x))) $

==
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

==
    #cimage("figures/lecture_5/fashion_mnist_probs.png", height: 80%)

==
  We consider the label $bold(y)_[i]$ as a conditional distribution

  $ P(bold(y)_[i] | bold(x)_[i]) = vec(
    P("Shirt" | #image("figures/lecture_5/shirt.png", height: 10%)),
    P("Bag" | #image("figures/lecture_5/shirt.png", height: 10%))
  ) = vec(1, 0) $ #pause

  We approximate this with our model $f(bold(x), bold(theta))$ #pause

  Our loss function measures the difference between two distributions  #pause
  
  $ P(bold(y)_[i] | bold(x)_[i]), quad f(bold(x), bold(theta)) $

==
  The *Kullback-Leibler Divergence (KL)* measures the difference between distributions #pause

  #cimage("figures/lecture_5/forwardkl.png", height: 50%)

==
  From the KL divergence, we derived the *cross-entropy loss* function, which we use for classification #pause

  $ = - sum_(i=1)^(d_y) P(y_i | bold(x)) log f(bold(x), bold(theta))_i $ #pause

  $ cal(L)(bold(x), bold(y), bold(theta)) = [- sum_(j=1)^n sum_(i=1)^(d_y) P(y_([j], i) | bold(x)_[j]) log f(bold(x)_[j], bold(theta))_i ] $ 

// 12:30

==
  Finish coding exercise

  https://colab.research.google.com/drive/1BGMIE2CjlLJOH-D2r9AariPDVgxjWlqG#scrollTo=AnHP-PHVhpW_

// 12:30 + 15:00?

==
  #cimage("figures/lecture_6/poor_minima.png", height: 80%) #pause

  1. Objective 2. NN architecture 3. Param initialization 4. Optimizer

==
  #cimage("figures/lecture_1/timeline.svg") #pause

  Today we will move from 1982 to 2012


= Dirty Secret of Deep Learning
==
    So far, I gave you the impression that deep learning is rigorous #pause

    Biological inspiration, theoretical bounds and mathematical guarantees #pause

    For complex neural networks, deep learning is a *science* not *math* #pause

    There is no widely-accepted theory for why deep neural networks are so effective #pause

    In modern deep learning, we progress using trial and error #pause
    - Early DL used deductive reasoning (proofs from axioms) #pause
    - Modern DL uses inductive reasoning (hypothesis, experiments)

==
    Scientific method: #pause
        + Collect observations #pause
        + Form hypothesis #pause
        + Run experiment #pause
        + Publish theory #pause

    However, there is a second part of the scientific method: #pause
        + Find existing theory #pause
        + Show counterexample through experiment #pause
        + Publish counterexample #pause
        + Update or falsify theory #pause

    Deep learning is new, so much of part 2 has not happened yet!

==
    For many ideas, the *observations* are stronger than the *theory* #pause

    Observe that an idea improves many types of neural networks #pause

    Ideas we discuss today are known to improve neural networks #pause
    - We will discuss theories about why they improve neural networks #pause
    - If you do not believe the theory, prove it wrong and be famous! #pause

    Even if we do not agree on *why* an idea works, if we *observe* that it helps, we can still use it #pause

    This is how medicine works (e.g., Anesthetics)!

==
  Theoretical advancements are still very important! #pause

  40 years of neural network research before we understood how to train neural networks #pause

  #cimage("figures/lecture_1/timeline.svg", width: 77%) #pause

  Modern networks too complex for humans to understand


= Optimization is Hard

// 15:00 + 15:00

==
    A 2-layer neural network can represent *any* continuous function to arbitrary precision #pause

    $ | f(bold(x), bold(theta)) - g(bold(x)) | < epsilon $ #pause

    $ lim_(d_h -> oo) epsilon = 0 $ #pause

    However, finding such $bold(theta)$ is a much harder problem

==
    Gradient descent only guarantees convergence to a *local* optima #pause

    #cimage("figures/lecture_6/poor_minima.png", height: 80%)

==
    #cimage("figures/lecture_6/poor_minima.png", height: 75%) 

    Harder tasks can have millions of local optima, and many of the local optima are not very good!

==
    Many of the ideas today create *smoother* loss landscapes #pause

    #cimage("figures/lecture_6/skip_connection_img.png", height: 70%) #pause

    Gradient descent reaches a better optimum more quickly in these cases 

= Deeper Neural Networks
==
    A two-layer neural network is sufficient to approximate any continuous function to arbitrary precision #pause

    But only with infinite width $d_h -> oo$ #pause

    For certain problems, adding one more layer is equivalent to *exponentially* increasing the width
    #text(size: 18pt)[- Eldan, Ronen, and Ohad Shamir. "The power of depth for feedforward neural networks." Conference on learning theory. PMLR, 2016.] #pause


    $ 256 times d_y => e^128 times d_y approx 10^111 times d_y; quad "universe has " 10^80 "atoms"$ #pause

    Another layer can solve this problem in $256 times 256 + 256 times d_y$ params #pause

    We need more layers for harder problems 
// 24:00 + 15:00

==
  #cimage("figures/lecture_6/filters.png", width: 120%)  

==
    In fact, we do not just need *deeper* networks, but also *wider* networks #pause

    The number of neurons in a deep neural network affects the quality of local optima #pause

    From Choromanska et al. _The loss surfaces of multilayer networks._ (2014): #pause

    - "For large-size networks, most local minima are equivalent and yield similar performance on a test set." #pause

    - "The probability of finding a “bad” (high value) local minimum is non-zero for small-size networks and decreases quickly with network size"

==

    #cimage("figures/lecture_6/skip_connection_img.png", height: 70%) 

==
  This is a difficult finding to conceptualize #pause

  The plots I show you are always 3D (2 parameters) #pause

  For neural networks with 1,000,000+ parameters, our geometric intuition begins to fail #pause

  We understand bigger/deeper networks often perform better, but we are missing deeper theory #pause

  We call such networks *overparameterized* neural networks

==
  *Summary:* Deeper and wider neural networks produce better results #pause

  Add more layers to your network #pause

  Increase the width of each layer

==
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
    l1 = nn.Linear(d_x, d_h)
    l2 = nn.Linear(d_h, d_h) 
    l3 = nn.Linear(d_h, d_h) 
    ...
    l6 = nn.Linear(d_h, d_y) 
    ```
    ]
==

    ```python
    import torch
    d_x, d_y, d_h = 1, 1, 256
    net = torch.nn.Sequential(
        torch.nn.Linear(d_x, d_h),
        torch.nn.Sigmoid(),
        torch.nn.Linear(d_h, d_h),
        torch.nn.Sigmoid(),
        ...
        torch.nn.Linear(d_h, d_y),
    )

    x = torch.ones((d_x,))
    y = net(x)
    ```

// 32:00 + 15:00

==
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

= Activation Functions
==
    The sigmoid function was the standard activation function until \~ 2012 #pause

    #cimage("figures/lecture_1/timeline.svg", width: 90%) #pause

    In 2012, people realized that ReLU activation performed much better

==
    #side-by-side[#sigmoid #pause][
        The sigmoid function can result in a *vanishing gradient* #pause
        
        $ f(bold(x), bold(theta)) = sigma(bold(theta)_3^top sigma(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x))))) $ #pause
    ] 


    #only(4)[
    $ (gradient_bold(theta_1) f)(bold(x), bold(theta)) = 
    (gradient sigma)(bold(theta)_3^top sigma(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x)))))

    dot (gradient sigma)(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x))))

    dot (gradient sigma)(bold(theta)_1^top overline(bold(x)))
    $ 

    ]

    #only((5,6))[
    $ (gradient_bold(theta_1) f)(bold(x), bold(theta)) = 
    underbrace((gradient sigma)(bold(theta)_3^top sigma(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x))))), < 0.25)

    dot underbrace((gradient sigma)(bold(theta)_2^top sigma(bold(theta)_1^top overline(bold(x)))), < 0.25)

    dot underbrace((gradient sigma)(bold(theta)_1^top overline(bold(x))), < 0.25) overline(bold(x))
    $ 
    ]

    #only(6)[
      #side-by-side[
      $ max_(bold(theta), bold(x)) (gradient_bold(theta_1) f)(bold(x), bold(theta)) approx 0.01 overline(bold(x)) $
      ][
      $ lim_(ell -> oo) (gradient_bold(theta_1) f)(bold(x), bold(theta)) = 0 $
      ]
    ]
==
    #only((1,2))[Fix vanishing gradient with *rectified linear unit (ReLU)*]
    #side-by-side[$ sigma(z) = max(0, z) \ gradient sigma(z) = cases(0 "if" z < 0, 1 "if" z >= 0) $ #pause][#relu #pause]

    #only((3,4,5))[Looks nothing like a biological neuron]

    #only((4,5,6))[However, it works much better than sigmoid in practice] 

    #only((5,6,7))[Via chain rule, gradient is $1 dot 1 dot 1 dots$ which does not vanish]
    
    #only((6,7,8))[If layer outputs one positive number, gradient will not vanish] 

    #only((7,8,9))[*Question:* Any problems?]

    #only((8,9,10))[*Answer:* Zero gradient region!]

    #only((9, 10))[Neurons can get "stuck", always output 0]

    #only((10))[These neurons cannot recover, they are *dead neurons*]

==
    #side-by-side[$ sigma(x) = max(0, x) \ gradient sigma(x) = cases(0 "if" x < 0, 1 "if" x >= 0) $ ][#relu]

    These neurons cannot recover, they are *dead neurons* #pause

    Training for longer results in more dead neurons #pause

    Dead neurons hurt your network (brain damage)!

==
    To fix dying neurons, use *leaky ReLU* #pause

    #side-by-side[$ sigma(z) = max(0.1 z, z) \ gradient sigma(z) = cases(0.1 "if" z < 0, 1 "if" z >= 0) $ #pause][#lrelu #pause]

    Small negative slope allows dead neurons to recover #pause

    As long as one neuron is positive in each layer, non-vanishing gradient

==
    #side-by-side(align: left)[
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

==
    https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

    https://jax.readthedocs.io/en/latest/jax.nn.html#activation-functions


= Parameter Initialization
==
  Recall the gradient descent algorithm

  #gd_algo()


==
    Initial $bold(theta)$ is starting position for gradient descent #pause

    #cimage("figures/lecture_6/poor_minima.png", height: 70%) #pause

    Pick $bold(theta)$ that results in good local minima

==
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

    *Answer:* For ReLU activation, the gradient will always be zero

    $ (gradient_bold(theta)_1 f)(bold(x), bold(theta)) = (gradient sigma) (bold(theta)^top_2 sigma( bold(theta)_1^top overline(bold(x)))) dot (gradient sigma)( bold(theta)_1^top overline(bold(x))) dot overline(bold(x)) $ #pause

    $ (gradient_bold(theta)_1 f)(bold(x), bold(theta)) = underbrace((gradient sigma) (bold(0)^top sigma ( bold(0)^top overline(bold(x)))), 0) dot underbrace((gradient sigma) ( bold(0)^top overline(bold(x))), 0) dot overline(bold(x)) = 0 $

==
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

    $ z_i = sigma(sum_(j=1)^d_x theta_j dot overline(x)_j)  = sigma(sum_(j=1)^d_x overline(x)_j) $ #pause

    All neurons in layer have the same gradient,and will receive the same update. Equivalent to a network of width 1, not good!

==
    We randomly initialize $bold(theta)$ to break symmetry

    $ bold(theta) = mat(
        -0.5, dots, 2; 
        dots.v, dots.down, dots.v;
        0.1, dots, 0.6
    ), mat(
        1.3, dots, 1.2; 
        dots.v, dots.down, dots.v;
        -0.8, dots, -1.1
    ), dots $ #pause

    But what scale? If $bold(theta) approx 0$ the gradient will vanish to zero, if $|bold(theta)| >> 0$ the gradient explode to infinity #pause

    Almost everyone initializes following a single paper from 2010: #pause
        - Xavier Glorot and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." #pause
        - Maybe there are better options?

==
    Magic expression, given the input and output size of the layer is $d_h$

    $ bold(theta) tilde cal(U)[ - sqrt(6) / sqrt(2 d_h), sqrt(6) / sqrt(2 d_h)] $ #pause

    Magic expression with input/output sizes $d_x, d_y$ 

    $ bold(theta) tilde cal(U)[ - sqrt(6) / sqrt(d_x + d_y), sqrt(6) / sqrt(d_x + d_y)] $

==
    These equations are designed for ReLU and similar activation functions #pause

    They prevent vanishing or exploding gradients in deep networks

==
    Usually `torch` and `jax/equinox` will automatically use Glorot initialization when you create `nn.Linear` #pause

    ```python
    layer = nn.Linear(d_x, d_h) # Uses Glorot init
    ``` #pause

    You can find many initialization functions at https://pytorch.org/docs/stable/nn.init.html #pause

    For JAX it is https://jax.readthedocs.io/en/latest/jax.nn.initializers.html

==
    ```python

    import torch
    d_h = 10
    # Manually
    theta = torch.zeros((d_h + 1, d_h))
    torch.nn.init.xavier_uniform_(theta) 
    theta = torch.nn.Parameter(theta)

    # Using nn.Linear
    layer = torch.nn.Linear(d_h, d_h)
    # Use .data, to bypass autograd security
    torch.nn.init.xavier_uniform_(layer.weight.data)
    torch.nn.init.xavier_uniform_(layer.bias.data) # Or zero

    ```

==
    ```python
    import jax
    d_h = 10

    init = jax.nn.initializers.glorot_uniform()
    theta = init(jax.random.key(0), (d_h + 1, d_h))
    ```

==
    ```python
    import jax, equinox
    d_h = 10
    
    layer = equinox.nn.Linear(d_h, d_h, key=jax.random.key(0))
    # Create new bias and weight
    init = jax.nn.initializers.glorot_uniform()
    new_weight = init(jax.random.key(1), (d_h, d_h))
    new_bias = init(jax.random.key(2), (d_h,))
    # Use a lambda function to save space
    # tree_at creates a new layer with the new weight
    layer = equinox.tree_at(lambda l: l.weight, layer, new_weight)
    layer = equinox.tree_at(lambda l: l.bias, layer, new_weight)
    ```

==
    Remember, in `equinox` and `torch`, `nn.Linear` is already initialized using Glorot! #pause

    In some cases, you may want to use different initializations #pause
    - Reinforcement learning #pause
    - Recurrent neural networks



= Stochastic Gradient Descent
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
==
  #gd_algo(init: "Glorot()") #pause

  Gradient descent computes $gradient cal(L)$ over all $bold(X)$

==
  This works for our small datasets, where $n = 1000$ #pause

  *Question:* How many GB are the LLM datasets? #pause *Answer:* \~1M GB

  //*Answer:* About 774,000 GB according to _Datasets for Large Language Models: A Comprehensive Survey_ #pause

  This is just the dataset size, the gradient is orders of magnitude larger #pause

  $ (gradient_bold(theta) cal(L))(bold(x)_[i], bold(y)_[i], bold(theta)) = overbrace(mat(
      (partial f_1) / (partial z_1), dots, (partial f_(d_h)) / (partial z_1);
      dots.v, dots.down, dots.v;
      (partial f_(d_h)) / (partial z_1), dots, (partial f_(d_h)) / (partial z_(d_h));
  ), "One layer")_[i] dots
  $

  Gradient at least $O(ell dot d_h^2 dot n) = underbrace(128 dot 16384^2 dot 10^12, "GPT guess") approx 10^22 approx 10^12 "GB"$

==
    *Question:* We do not have enough memory to compute the gradient. What can we do? #pause

    *Answer:*  We approximate the gradient using a subset of the data #pause

    We call this *stochastic gradient descent* (SGD)

==
  #sgd_algo
   

==
    Stochastic gradient descent (SGD) is useful for saving memory #pause

    But it can also improve performance #pause

    Since the "dataset" changes every update, so does the loss manifold #pause

    This makes it less likely we get stuck in bad optima

    #cimage("figures/lecture_5/saddle.png")

==
    `torch.utils.data.DataLoader` helps with SGD #pause

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


= Modern Optimization
==
    Gradient descent is a powerful tool, but it has issues #pause
    + It can be slow to converge #pause
    + It can get stuck in poor local optima #pause

    Improve gradient descent convergence rate, while also preventing premature convergence #pause

    It is hard to teach adaptive optimization solely through equations #pause

    I want to show you a video to prepare you

    //https://www.youtube.com/watch?v=MD2fYip6QsQ&t=77s
    https://youtu.be/MD2fYip6QsQ?si=0PFkmaKWKdXAg_Sy&t=197

==
    There are two key concepts behind modern optimizers:
    - Momentum #pause
    - Adaptive learning rate #pause

    Let us discuss the algorithms further

==
    Review gradient descent again, because we will be making changes to it #pause

    We usually use SGD (batching), for simplicity I will ignore this

    #gd_algo(init: "Glorot()")

==
    Introduce *momentum* first #pause

    #gd_momentum_algo

==
    Now *adaptive learning rate* #pause
    
    #gd_adaptive_algo

==
    Combine *#greenm[momentum]* and *#bluem[adaptive learning rate]* to create *Adam* #pause

    #adam_algo


==
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
==
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

= Weight Decay
==
  Many modern optimizers also include *weight decay* #pause

  Weight decay penalizes large parameters #pause

  $ cal(L)_("decay")(bold(X), bold(Y), bold(theta)) = cal(L)_("decay")(bold(X), bold(Y), bold(theta)) + lambda sum_(i) theta_i^2 $ #pause

  This results in a smoother, parabolic loss landscape that is easier to optimize #pause

  (Unproven opinion) It reduces the number of saddle points and local minima

==
  $ z = sin^2(x) + sin^2(y) $
  #side-by-side[
    #cimage("figures/lecture_6/wd0.png")
  ][
    #cimage("figures/lecture_6/wd1.png")
  ]

==
  Weight decay is built into the Adam optimizer #pause

  But the implementation is wrong! #pause

  Years later, someone noticed and published a new paper that fixes weight decay implementation #pause

  ```python
  Lambda = 0.01
  opt = torch.optim.AdamW(..., weight_decay=Lambda)
  opt = optax.adamw(..., weight_decay=Lambda)
  ```

= Coding
==

  https://colab.research.google.com/drive/1qTNSvB_JEMnMJfcAwsLJTuIlfxa_kyTD#scrollTo=YVkCyz78x4Rp