#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#import "@preview/touying:0.6.1": *
#import "@preview/fletcher:0.5.8" as fletcher: node, edge

#let cetz-canvas = touying-reducer.with(reduce: canvas, cover: draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

//
// General Functions
//

#let normal = { 
    canvas(length: 1cm, {
  plot.plot(size: (16, 10),
    x-tick-step: 2,
    y-tick-step: 0.5,
    y-min: 0,
    y-max: 1,
    x-label: [$ bold(z) $],
    y-label: [$ P(bold(z)) $],
    {
      plot.add(
        domain: (-4, 4), 
        style: (stroke: (thickness: 5pt, paint: red)),
        x => calc.pow(calc.e, -(0.5 * calc.pow(x, 2)))
      )
    })
})}

#let forgetting = { 
    set text(size: 22pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 10,
    y-tick-step: 0.5,
    y-min: 0,
    y-max: 1,
    x-label: [Time],
    y-label: [$ bold(h) $],
    {
      plot.add(
        domain: (0, 40), 
        label: [Memory Strength],
        style: (stroke: (thickness: 5pt, paint: red)),
        t => calc.pow(0.9, t)
      )
    })
})}

//
// Flow Charts
//

#let varinf = fletcher-diagram(
  node-stroke: .1em,
  spacing: 4em,
  node((0,0), $bold(z)$, radius: 2em),
  edge($P(bold(x) | bold(z); bold(theta))$, "-|>"),
  node((2,0), $bold(x)$, radius: 2em),
  edge((0,0), (2,0), $P(bold(z) | bold(x); bold(theta))$, "<|-", bend: -40deg),
)

//
// Activation Functions
//

#let heaviside = canvas(length: 1cm, {
  plot.plot(size: (8, 4),
    x-tick-step: 1,
    y-tick-step: 2,
    {
      plot.add(
        domain: (-2, 2), 
        x => calc.clamp(calc.floor(x + 1), 0, 1),
        style: (stroke: (paint: red, thickness: 4pt)),
      )
    })
})



#let sigmoid = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 2,
    y-tick-step: none,
    y-ticks: (0, 0.25, 1),
    y-min: 0,
    y-max: 1,
    x-label: $ z $,
    y-label: none,
    {
      plot.add(
        domain: (-5, 5), 
        style: (stroke: (thickness: 5pt, paint: red)),
        label: $ sigma(z) $,
        x => 1 / (1 + calc.pow(2.718, -x)),
      )
      plot.add(
        domain: (-5, 5), 
        style: (stroke: (thickness: 3pt, paint: blue)),
        label: $ gradient sigma(z)$,
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
    y-ticks: (0, 1, 3, 5),
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

//
// Convolution
//

#let draw_filter(x, y, cells, colors: none) = {
  import draw: *

  grid((x, y), (x + cells.len(), y + cells.at(0).len()))
  
  for i in range(cells.len()) {
    for j in range(cells.at(i).len()) {
      
      if (colors != none)  {
        let cell_color = colors.at(cells.at(i).len() - j - 1).at(i)
        if (cell_color != none){
          rect((i, j), (i + 1, j + 1), fill: cell_color)
        }
        content((x + i + 0.4, y + j + 0.6), (x + i + 0.3, y + j + 0.8), cells.at(cells.at(i).len() - j - 1).at(i))

      } else {
        content((x + i + 0.4, y + j + 0.6), (x + i + 0.3, y + j + 0.8), str(cells.at(cells.at(i).len() - j - 1).at(i)))
      }

      }
  }
}

#let stonks = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 2,
    y-tick-step: 20,
    y-min: 0,
    y-max: 100,
    x-label: $ t $,
    y-label: $ x(t) $,
    {
      plot.add(
        domain: (0, 5), 
        label: [Stock Price (MOP)],
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 10 + 100 * (0.3 * calc.sin(0.2 * t) +
                0.1 * calc.sin(1.5 * t) +
                0.05 * calc.sin(3.5 * t) +
                0.02 * calc.sin(7.0 * t))
      )
    })
})}

#let waveform = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 0.5,
    y-tick-step: 500,
    y-min: -1000,
    y-max: 1000,
    x-label: $ t $,
    y-label: $ x(t) $,
    {
      plot.add(
        domain: (0, 1), 
        label: [dBm],
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 500 * (
          calc.sin(calc.pi * 1320 * t)
        )
      )
    })
})}
#let waveform_left = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 0.5,
    y-tick-step: 500,
    x-min: 0,
    x-max: 1,
    y-min: -1000,
    y-max: 1000,
    x-label: "Time (Seconds)",
    y-label: "dBm",
    {
      plot.add(
        domain: (0, 0.33), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 500 * (
          calc.sin(2 * calc.pi * 9.1 * t)
        )
      )
      plot.add(
        domain: (.33, 1.0), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => t
      )
    })
})}

#let hello = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 0.5,
    y-tick-step: 1,
    x-min: 0,
    x-max: 1,
    y-min: 0,
    y-max: 1,
    x-label: "Time (Seconds)",
    y-label: "Hello",
    {
      plot.add(
        domain: (0, 0.33), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 1
      )
      plot.add(
        domain: (.33, 1.0), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 0
      )
      plot.add-vline(
        0.33,
        style: (stroke: (thickness: 5pt, paint: red)),
      )
    })
})}

#let waveform_right = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 0.5,
    y-tick-step: 500,
    x-min: 0,
    x-max: 1,
    y-min: -1000,
    y-max: 1000,
    x-label: "Time (Seconds)",
    y-label: "Frequency (Hz)",
    {
      plot.add(
        domain: (0.66, 1.0), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 500 * (
          calc.sin(2 * calc.pi * 9.1 * t)
        )
      )
      plot.add(
        domain: (0, 0.66), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => t
      )
    })
})}

#let implot = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (7, 7),
    x-tick-step: 64,
    y-tick-step: 64,
    y-min: 0,
    y-max: 256,
    x-min: 0,
    x-max: 256,
    x-label: [$u$ (Pixels)],
    y-label: [$v$ (Pixels)],
    {
      plot.add(
        label: $ x(u, v) $,
        domain: (0, 1), 
        style: (stroke: (thickness: 0pt, paint: red)),
        t => 1000 * (
          calc.sin(calc.pi * 1320 * t)
        )
      )
      plot.annotate({
        import draw: *
        content((128, 128), image("figures/lecture_7/ghost_dog_bw.svg", width: 7cm))
      })
    })
})}

#let implot_color = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (7, 7),
    x-tick-step: 64,
    y-tick-step: 64,
    y-min: 0,
    y-max: 256,
    x-min: 0,
    x-max: 256,
    x-label: [$u$ (Pixels)],
    y-label: [$v$ (Pixels)],
    {
      plot.add(
        label: $ x(u, v) $,
        domain: (0, 1), 
        style: (stroke: (thickness: 0pt, paint: red)),
        t => 1000 * (
          calc.sin(calc.pi * 1320 * t)
        )
      )
      plot.annotate({
        import draw: *
        content((128, 128), image("figures/lecture_1/dog.png", width: 7cm))
      })
    })
})}

#let implot_left = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (7, 7),
    x-tick-step: 64,
    y-tick-step: 64,
    y-min: 0,
    y-max: 256,
    x-min: 0,
    x-max: 256,
    x-label: [$u$ (Pixels)],
    y-label: [$v$ (Pixels)],
    {
      plot.add(
        domain: (0, 1), 
        style: (stroke: (thickness: 0pt, paint: red)),
        t => 1000 * (
          calc.sin(calc.pi * 1320 * t)
        )
      )
      plot.annotate({
        import draw: *
        rect((0, 0), (256, 256), fill: gray)
      })
      plot.annotate({
        import draw: *
        content((64, 64), image("figures/lecture_7/ghost_dog_bw.svg", width: 3cm))
      })
    })
})}

#let implot_right = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (7, 7),
    x-tick-step: 64,
    y-tick-step: 64,
    y-min: 0,
    y-max: 256,
    x-min: 0,
    x-max: 256,
    x-label: [$u$ (Pixels)],
    y-label: [$v$ (Pixels)],
    {
      plot.add(
        domain: (0, 1), 
        style: (stroke: (thickness: 0pt, paint: red)),
        t => 1000 * (
          calc.sin(calc.pi * 1320 * t)
        )
      )
      plot.annotate({
        import draw: *
        rect((0, 0), (256, 256), fill: gray)
      })
      plot.annotate({
        import draw: *
        content((192, 192), image("figures/lecture_7/ghost_dog_bw.svg", width: 3cm))
      })
    })
})}

#let conv_signal_plot = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (20, 2.7),
    y-tick-step: none,
    x-tick-step: none,
    y-min: 0,
    y-max: 1,
    x-min: 0,
    x-max: 1,
    x-label: "t",
    y-label: "",
    {
      plot.add(
        domain: (0, 0.4), 
        label: $ x(t) $,
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 0.05 + 0.05 * (
          calc.sin(calc.pi * 50 * t)
        )
      )
      plot.add(
        domain: (0.4, 0.6), 
        style: (stroke: (thickness: 5pt, paint: red)),
        //t => (t - 0.4) / 0.3 + 0.05 * (
        //  calc.sin(calc.pi * 50 * t)
        //)
        t => 0.05 * calc.sin(calc.pi * 50 * t) + 0.8 * 1 / (1 + calc.exp(-30 * (t - 0.5)))
      )
      plot.add(
        domain: (0.6, 1.0), 
        style: (stroke: (thickness: 5pt, paint: red)),
        t => 0.75 + 0.05 * (
          calc.sin(calc.pi * 50 * t)
        )
      )
    })
})}  

#let conv_filter_plot = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (20, 2.75),
    y-tick-step: none,
    x-tick-step: none,
    y-ticks: (0, 0.5),
    y-min: 0,
    y-max: 0.5,
    x-min: 0,
    x-max: 1,
    x-label: "t",
    y-label: "",
    {
      plot.add(
        domain: (0, 0.2), 
        label: $ g(t) $,
        style: (stroke: (thickness: 5pt, paint: blue)),
        t => 1 / (calc.pow(2  * 3.14, 0.5)) * calc.exp(-calc.pow((t - 0.1), 2) / 0.002)
      )
    })
})}  

#let conv_result_plot = { 
    set text(size: 25pt)
    canvas(length: 1cm, {
  plot.plot(size: (20, 2.75),
    y-tick-step: none,
    x-tick-step: none,
    y-min: 0,
    y-max: 1,
    x-min: 0,
    x-max: 1,
    x-label: "t",
    y-label: "",
    {
      plot.add(
        domain: (0, 0.4), 
        label: $ x(t) * g(t) $,
        style: (stroke: (thickness: 5pt, paint: purple)),
        t => 0.05 
      )
      plot.add(
        domain: (0.4, 0.6), 
        style: (stroke: (thickness: 5pt, paint: purple)),
        t => 0.01 + 0.8 * 1 / (1 + calc.exp(-30 * (t - 0.5)))
      )
      plot.add(
        domain: (0.6, 1.0), 
        style: (stroke: (thickness: 5pt, paint: purple)),
        t => 0.77 
      )
    })
})}  




#let draw_conv = canvas({
  import draw: *


let filter_values = (
  (1, 0, 1),
  (0, 4, 0),
  (0, 0, 2)
)

let image_values = (
  (4, 4, 4, 4),
  (0, 0, 0, 0),
  (4, 0, 0, 4),
  (4, 0, 0, 4),
)
draw_filter(0, 0, image_values)
})