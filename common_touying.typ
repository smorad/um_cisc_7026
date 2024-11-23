#import "@preview/touying:0.5.3": *
#import themes.university: *
#import "@preview/cetz:0.3.1": canvas, draw
#import "@preview/cetz-plot:0.1.0": plot, chart

#set text(size: 25pt)
#set math.vec(delim: "[")
#set math.mat(delim: "[")
#let argmin = $op("arg min", limits: #true)$
#let softmax= $op("softmax", limits: #true)$

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