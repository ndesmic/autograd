import { Network } from "./neuron.js";
import { sum, meanSquared } from "./math-utils.js";
import { asValues } from "./autograd.js";

const inputs = [
	[2,3,-1],
	[3,-1,0.5],
	[0.5,1,1],
	[1,1,-1]
];
const expectations = asValues([
	1,
	-1,
	-1,
	1
]);

const network = new Network(3, [4, 4, 1]);
console.log("parameter count:", network.parameters.length);

function predictWithLoss(){
	const predictions = inputs.map(ex => network.forward(ex)[0]);
	const totalLoss = sum(predictions.map((p, i) => meanSquared(expectations[i], p)));

	return totalLoss;
}

function tune(network, learningRate){
	for (const p of network.parameters) {
		p.value += p.grad * -learningRate;
	}
}

const learningRate = 0.1;
const steps = 50;

for(let i = 0; i < steps; i++){
	const loss = predictWithLoss();
	console.log("loss:", loss.toString());
	network.parameters.forEach(p => p.grad = 0);
	loss.backward();

	tune(network, learningRate);
}

console.log(inputs.map(ex => network.forward(ex)[0]).map(v => v.toString()));