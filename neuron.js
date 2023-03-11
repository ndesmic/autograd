import { Value } from "./autograd.js";

export class Neuron {
	#inCount;
	#weights = [];
	#bias;

	constructor(inCount){
		for(let i = 0; i < inCount; i++){
			this.#weights.push(new Value((Math.random() * 2) - 1));
		}
		this.#bias = new Value((Math.random() * 2) - 1);
		this.#inCount = inCount;
	}

	forward(inValues){
		let outValue = new Value(0);
		for(let i = 0; i < this.#inCount; i++){
			outValue = outValue.add(this.#weights[i].mul(inValues[i]));
		}
		outValue.add(this.#bias);

		return outValue.tanh();
	}

	get parameters(){
		return [...this.#weights, this.#bias];
	}
}

export class Layer {
	#neurons = [];

	constructor(inCount, outCount){
		for(let i = 0; i < outCount; i++){
			this.#neurons.push(new Neuron(inCount));
		}
	}

	forward(inValues){
		return this.#neurons.map(n => n.forward(inValues));
	}

	get parameters(){
		return this.#neurons.flatMap(n => n.parameters);
	}
}

export class Network{
	#layers = [];

	//outputs is an array of output counts
	constructor(inCount, outputs){
		const sizes = [inCount, ...outputs];
		for(let i = 0; i < outputs.length; i++){
			this.#layers.push(new Layer(sizes[i], sizes[i+1]));
		}
	}

	forward(inValues){
		let outValues = inValues;
		for(const layer of this.#layers){
			outValues = layer.forward(outValues);
		}
		return outValues;
	}

	get parameters(){
		return this.#layers.flatMap(l => l.parameters);
	}
}