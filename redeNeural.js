function sigmoid(x) {
  // Reference: https://pt.wikipedia.org/wiki/Fun%C3%A7%C3%A3o_sigm%C3%B3ide
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(x) {
  return x * (1 - x);
}

class RedeNeural {
  // input, hidden, output
  constructor(i_nodes, h_nodes, o_nodes) {
    this.i_nodes = i_nodes;
    this.h_nodes = h_nodes;
    this.o_nodes = o_nodes;
    
    this.bias_ih = new Matrix(this.h_nodes, 1);
    this.bias_ih.randomize();
    this.bias_ho = new Matrix(this.o_nodes, 1);
    this.bias_ho.randomize();

    this.weigths_ih = new Matrix(this.h_nodes, this.i_nodes);
    this.weigths_ih.randomize();

    this.weigths_ho = new Matrix(this.o_nodes, this.h_nodes);
    this.weigths_ho.randomize();

    this.learning_rate = 0.1;
  }

  train(arr, target) {
    // input -> hidden
    let input = Matrix.arraytoMatrix(arr);
        
    let hidden = Matrix.multiply(this.weigths_ih, input);
    hidden = Matrix.add(hidden, this.bias_ih);

    hidden.map(sigmoid);

    // hidden -> output
    let output = Matrix.multiply(this.weigths_ho, hidden);
    output = Matrix.add(output, this.bias_ho);
    output.map(sigmoid);

    // Backpropagation

    // output -> hidden
    let expected = Matrix.arraytoMatrix(target);
    let output_error = Matrix.subtract(expected, output);
    let d_output = Matrix.map(output, dsigmoid)
    let hidden_T = Matrix.transpose(hidden);
    
    let grandient = Matrix.hadamard(d_output, output_error);
    grandient = Matrix.escalar_multiply(grandient, this.learning_rate);

    // Adjust Bias O -> H
    this.bias_ho = Matrix.add(this.bias_ho, grandient);
    
    // Adjust Weights O -> H
    let weigths_ho_deltas = Matrix.multiply(grandient, hidden_T);
    this.weigths_ho = Matrix.add(this.weigths_ho, weigths_ho_deltas);
    
    // hidden -> input
    let weigths_ho_T = Matrix.transpose(this.weigths_ho);
    let hidden_error = Matrix.multiply(weigths_ho_T, output_error);
    let d_hidden = Matrix.map(hidden, dsigmoid);
    let input_T = Matrix.transpose(input);

    let grandient_H = Matrix.hadamard(d_hidden, hidden_error);
    grandient_H = Matrix.escalar_multiply(grandient_H, this.learning_rate);

    // Adjustment Bias I -> H
    this.bias_ih = Matrix.add(this.bias_ih, grandient_H);

    let weigths_ih_deltas = Matrix.multiply(grandient_H, input_T);
    this.weigths_ih = Matrix.add(this.weigths_ih, weigths_ih_deltas);
  }

  predict(arr) {
    // input -> hidden
    let input = Matrix.arraytoMatrix(arr);
        
    let hidden = Matrix.multiply(this.weigths_ih, input);
    hidden = Matrix.add(hidden, this.bias_ih);

    hidden.map(sigmoid);

    // hidden -> output
    let output = Matrix.multiply(this.weigths_ho, hidden);
    output = Matrix.add(output, this.bias_ho);
    output.map(sigmoid);
    output = Matrix.matrixToArray(output);

    return output;
  }
}