/** References:
 * https://www.youtube.com/watch?v=d8U7ygZ48Sc
 * https://www.youtube.com/watch?v=zVlMVanp-tA
 * https://www.youtube.com/watch?v=9KfelZhls2Q&t=879s
 */

 var nn;
 var train = true;

function setup() {
  createCanvas(500, 500);
  background(0);

  nn = new RedeNeural(2, 3, 1);
  
  // XOR Problem
  dataset = {
    inputs:
      [[1, 1],
      [1, 0],
      [0, 1],
      [0, 0]],
    outputs:
      [[0],
      [1],
      [1],
      [0]]
  }

}

function draw() {
  if (train) {
    for (let i = 0; i < 10000; i++) {
      const index = floor(random(4));
      nn.train(dataset.inputs[index], dataset.outputs[index]);      
    }

    if (nn.predict([0, 0])[0] < 0.04 && nn.predict([1, 0])[0] > 0.98) {
      train = false;
      console.log("terminou");
    }
  }

}