#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#import "@preview/pinit:0.2.2": *
#import "@preview/algorithmic:1.0.5"
#import algorithmic: style-algorithm, algorithm-figure, algorithm

#set text(size: 25pt)
#set math.vec(delim: "[")
#set math.mat(delim: "[")
#let argmin = $op("arg min", limits: #true)$
#let argmax = $op("arg max", limits: #true)$
#let scan = $op("scan")$
#let KL = $op("KL")$
#let softmax = $op("softmax", limits: #true)$
#let attention = $op("attention")$

#let pinit-highlight-equation-from(height: 2em, pos: bottom, fill: rgb(0, 180, 255), highlight-pins, point-pin, body) = {
  pinit-highlight(..highlight-pins, dy: -0.9em, fill: rgb(..fill.components().slice(0, -1), 40))
  pinit-point-from(
    fill: fill, pin-dx: 0em, pin-dy: if pos == bottom { 0.5em } else { -0.9em }, body-dx: 0pt, body-dy: if pos == bottom { -1.7em } else { -1.6em }, offset-dx: 0em, offset-dy: if pos == bottom { 0.8em + height } else { -0.6em - height },
    point-pin,
    rect(
      inset: 0.5em,
      stroke: (bottom: 0.12em + fill),
      {
        set text(fill: fill)
        body
      }
    )
  )
}

#let cimage(..args) = { 
  align(center + horizon, image(..args))
}

#let side-by-side(columns: none, gutter: 1em, align: center + horizon, ..bodies) = {
  let bodies = bodies.pos()
  let columns = if columns ==  none { (1fr,) * bodies.len() } else { columns }
  if columns.len() != bodies.len() {
    panic("number of columns must match number of content arguments")
  }

  grid(columns: columns, gutter: gutter, align: align, ..bodies)
}

#let redm(x) = {
  text(fill: color.red, $#x$)
}

#let bluem(x) = {
  text(fill: color.blue, $#x$)
}

#let greenm(x) = {
  text(fill: color.green, $#x$)
}


// Plots of activation functions
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
    x-label: $z$,
    y-label: none,
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
        label: $ sigma(z) $,
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
        label: $ gradient sigma(z)$,
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
    y-ticks: (0.1, 3, 5),
    y-min: -1,
    y-max: 5,
    x-label: $z$,
    y-label: none,
    {
      plot.add(
        domain: (-5, 5), 
        style: (stroke: (thickness: 5pt, paint: red)),
        label: $ sigma(z) $,
        line: (type: "linear"),
        x => calc.max(0.1 * x, x)
      )
      plot.add(
        domain: (-5, 0), 
        style: (stroke: (thickness: 3pt, paint: blue)),
        x => 0.1,
      )
      plot.add(
        domain: (0, 5), 
        style: (stroke: (thickness: 3pt, paint: blue)),
        label: $ gradient sigma(z)$,
        x => 1,
      )
    })
})}

#let draw_filter(x, y, cells, colors: none) = {
  import cetz.draw: *
  grid((x, y), (x + cells.len(), y + cells.at(0).len()))
  for i in range(cells.len()) {
    for j in range(cells.at(i).len()) {
      if (colors != none)  {
        let cell_color = colors.at(cells.at(i).len() - j - 1).at(i)
        if (cell_color != none){
          rect((i, j), (i + 1, j + 1), fill: cell_color)
        }
        content((x + i + 0.4, y + j + 0.6), (i, j), cells.at(cells.at(i).len() - j - 1).at(i))

      } else {
        content((x + i + 0.4, y + j + 0.6), (i, j), str(cells.at(cells.at(i).len() - j - 1).at(i)))
      }

      }
  }
}

#let gd_algo(init: "Init()") = algorithm(
  line-numbers: false,
  {
    import algorithmic: *
    Procedure(
      "Gradient Descent",
      ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$),
      {
        Assign[$bold(theta)$][#init]
        For(
          $i in {1 dots t}$,
          {
            Comment[Compute the gradient of the loss]        
            Assign[$bold(J)$][$(gradient_bold(theta) cal(L))(bold(X), bold(Y), bold(theta))$]
            Comment[Update the parameters using the negative gradient]
            Assign[$bold(theta)$][$bold(theta) - alpha dot bold(J)$]
          },
        )
        Return[$bold(theta)$]
      },
    )
  }
)

#let sgd_algo = algorithm(
      line-numbers: false,
      {
      import algorithmic: *

      Procedure(
        "Stochastic Gradient Descent", 
        ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$, redm[$B$]), 
        {
          Assign[$bold(theta)$][$"Glorot"()$] 
          For($i in 1 dots t$, {
            Assign[#redm[$bold(X), bold(Y)$]][#redm[$"Shuffle"(bold(X)), "Shuffle"(bold(Y))$]]
            For(redm[$j in 0 dots n / B - 1$], {
            Assign[#redm[$bold(X)_j$]][#redm[$mat(bold(x)_[ j B], bold(x)_[ j B + 1], dots, bold(x)_[ (j + 1) B - 1])^top$]]
            LineBreak
            Assign[#redm[$bold(Y)_j$]][#redm[$mat(bold(y)_[ j B], bold(y)_[ j B + 1], dots, bold(y)_[ (j + 1) B - 1])^top$]]
            LineBreak
            Assign[$bold(J)$][$(gradient_bold(theta) cal(L))(bold(X)_#redm[$j$], bold(Y)_#redm[$j$], bold(theta))$]
            Assign[$bold(theta)$][$bold(theta) - alpha dot bold(J)$]
            })
          })
      Return[$bold(theta)$]
      })
    }) 

#let gd_momentum_algo = algorithm(line-numbers: false, {
    import algorithmic: *

    Function(redm[$"Momentum"$] + " Gradient Descent", ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$, redm[$beta$]), {

      Assign[$bold(theta)$][$"Glorot"()$] 
      Assign[#redm[$bold(M)$]][#redm[$bold(0)$] #text(fill:red)[\# Init momentum]]

      For($i in 1 dots t$, {
        Assign[$bold(J)$][$(gradient_bold(theta) cal(L))(bold(X), bold(Y), bold(theta))$ #text(fill: red)[\# Represents acceleration]]
        Assign[#redm[$bold(M)$]][#redm[$beta dot bold(M) + (1 - beta) dot bold(J)$] #text(fill: red)[\# Momentum and acceleration]]
        Assign[$bold(theta)$][$bold(theta) - alpha dot #redm[$bold(M)$]$ #text(fill: red)[\# Multiply by momentum not gradient]]
      })

    Return[$bold(theta)$]
    })
  })


#let gd_adaptive_algo = algorithm(
      line-numbers: false,
      {
      import algorithmic: *

      Procedure(redm[$"RMSProp"$], ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$, redm[$beta$]), {

        Assign[$bold(theta)$][$"Glorot"()$] 
        Assign[#redm[$bold(V)$]][#redm[$bold(0)$] #text(fill: red)[\# Init variance]] 

        For($i in 1 dots t$, {
          Assign[$bold(J)$][$(gradient_bold(theta) cal(L))(bold(X), bold(Y), bold(theta))$ \# Represents acceleration]
          Assign[#redm[$bold(V)$]][#redm[$beta dot bold(V) + (1 - beta) dot bold(J) dot.circle bold(J) $] #text(fill: red)[\# Magnitude]]
          Comment[#text(fill: red)[\# Rescale grad by magnitude of prev updates]]
          Assign[$bold(theta)$][$bold(theta) - alpha dot #redm[$bold(J) div.circle bold(V)^(dot.circle 1/2)$]$]
        })

      Return[$bold(theta)$]
      })
  })


#let adam_algo = algorithm(line-numbers: false, {
    import algorithmic: *

    Procedure("Adaptive Moment Estimation", ($bold(X)$, $bold(Y)$, $cal(L)$, $t$, $alpha$, greenm[$beta_1$], bluem[$beta_2$]), {
      Assign[$bold(theta)$][$"Glorot"()$] 
      Assign[$#greenm[$bold(M)$], #bluem[$bold(V)$]$][$bold(0)$] 

      For($i in 1 dots t$, {
        Assign[$bold(J)$][$(gradient_bold(theta) cal(L))(bold(X), bold(Y), bold(theta))$]
        Assign[#greenm[$bold(M)$]][#greenm[$beta_1 dot bold(M) + (1 - beta_1) dot bold(J)$] #greenm[\# Update momentum]]
        Assign[#bluem[$bold(V)$]][#bluem[$beta_2 dot bold(V) + (1 - beta_2) dot bold(J) dot.circle bold(J)$] #bluem[\# Update magnitude]]
        Comment[\# Rescale #greenm[momentum] by #bluem[magnitude]]
        //Assign[$hat(bold(M))$][$bold(M)  "/" (1 - beta_1)$ \# Bias correction]
        //Assign[$hat(bold(V))$][$bold(V) "/" (1 - beta_2)$ \# Bias correction]

        Assign[$bold(theta)$][$bold(theta) - alpha dot #greenm[$bold(M)$] #bluem[$div.circle bold(V)^(dot.circle 1/2)$]$]
      })

    Return[$bold(theta)$ \# Note, we use biased $bold(M), bold(V)$ for clarity]
    })
  }) 