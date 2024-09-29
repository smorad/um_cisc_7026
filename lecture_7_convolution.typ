#import "@preview/cetz:0.2.2"
#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm


#set math.vec(delim: "[")
#set math.mat(delim: "[")


#let draw_filter(x, y, cells, colors: none) = {
  import cetz.draw: *
  grid((x, y), (x + cells.len(), y + cells.at(0).len()))
  for i in range(cells.len()) {
    for j in range(cells.at(i).len()) {
      if (colors != none)  {
        rect((i, j), (i + 1, j + 1), fill: red)
        content((x + i + 0.4, y + j + 0.6), (i, j), str(cells.at(cells.at(i).len() - j - 1).at(i)))

      } else{
        content((x + i + 0.4, y + j + 0.6), (i, j), str(cells.at(cells.at(i).len() - j - 1).at(i)))
      }

      }

  }
}


#let draw_conv = cetz.canvas({
  import cetz.draw: *


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
let colors = (
  red, red, red, red,
  red, red, red, red,
  red, red, red, red,
  red, red, red, red,
)
draw_filter(0, 0, image_values)
//content((2, 2), image("figures/lecture_6/ghost_dog.svg", width: 4cm))
})


// Signals, large continuous time inputs
// Want translation invariance and scalability to long sequences
// Introduce convolution (continuous)


#let agenda(index: none) = {
  let ag = (
    [Review],
    [Signal Processing], 
    [Continuous Convolution],
    [Discrete Convolution],
    [Audio Convolution],
    [Image Convolution],
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
