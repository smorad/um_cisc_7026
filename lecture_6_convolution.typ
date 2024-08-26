#import "@preview/cetz:0.2.2"

#cetz.canvas({
  import cetz.draw: *


let filter_values = (
  (1, 0, 1),
  (0, 4, 0),
  (0, 0, 2)
)

let image_values = (
  (1, 0, 2, 0, 5),
  (2, 0, 4, 1, 1),
  (0, 0, 1, 0, 0),
  (4, 4, 0, 1, 0),
  (3, 0, 4, 0, 1)
)

grid((0, 0), (3, 3))

for i in range(3) {
  for j in range(3) {
    content((i + 0.4, j + 0.6), (i, j), str(filter_values.at(i).at(j)))
  }
}

grid((4, -1), (8, 3))

for i in range(4) {
  for j in range(4) {
    content((4 + i + 0.4, -1 + j + 0.6), (i, j), str(image_values.at(i).at(j)))
  }
}

let items = for i in range(10) { (i * 2,) }

let conv = {
  for i in range(3){
    for j in range(3){
      (filter_values.at(i).at(j) * image_values.at(i).at(j), )
    }
  }
}

grid((9, 0), (12, 3))
for i in range(3) {
  for j in range(3) {
    content((i + 9.4, j + 0.6), (i, j), str(conv.at(i + 3 * j)))
  }
}


  
})
