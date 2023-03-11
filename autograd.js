import { topologicalSort } from "./topological-sort.js";

const symbolBackward = Symbol("backward");

export class Tensor {
	#value;
	#children;
	#op;
	#label;
	#grad = 0;
	#backward = () => {};

	constructor(val, children = [], { op = "", label = "" } = {}){
		this.#value = val;
		this.#children = children;
		this.#op = op;
		this.#label = label;
	}
	add(other){
		if(!(other instanceof Tensor)) other = new Tensor(other);
		const out = new Tensor(this.value + other.value, [this, other], { op: "+" });
		out[symbolBackward] = () => {
			this.grad += out.grad;
			other.grad += out.grad;
		};
		return out;
	}
	sub(other) {
		if (!(other instanceof Tensor)) other = new Tensor(other);
		const out = new Tensor(this.value - other.value, [this, other], { op: "-" });
		out[symbolBackward] = () => {
			this.grad += 1 * out.grad;
			other.grad += -1 * out.grad;
		};
		return out;
	}
	mul(other){
		if (!(other instanceof Tensor)) other = new Tensor(other);
		const out = new Tensor(this.value * other.value, [this, other], { op: "*" });
		out[symbolBackward] = () => {
			this.grad += other.value * out.grad;
			other.grad += this.value * out.grad;
		}
		return out;
	}
	div(other) {
		if (!(other instanceof Tensor)) other = new Tensor(other);
		const out = new Tensor(this.value / other.value, [this, other], { op: "/" });
		out[symbolBackward]= () => {
			this.grad += (1 / other.value) * out.grad;
			other.grad += (-this.value * (1 / other.value**2)) * out.grad;
		}
		return out;
	}
	neg(){
		const out = new Tensor(-this.value, [this], { op: `neg` });
		out[symbolBackward] = () => {
			this.grad += -1 * out.grad;
		}
		return out;
	}
	pow(val){
		const out = new Tensor(Math.pow(this.value, val), [this], { op: `pow(${val})` });
		out[symbolBackward] = () => {
			this.grad += val * Math.pow(this.value, val - 1) * out.grad;
		}
		return out;
	}
	exp(){
		const out = new Tensor(Math.exp(this.value), [this], { op: "exp" });
		out[symbolBackward] = () => {
			this.grad += Math.exp(this.value) * out.grad;
		}
		return out;
	}
	tanh(){
		const out = new Tensor(Math.tanh(this.value), [this], { op: "tanh" });
		out[symbolBackward] = () => {
			this.grad += (1 - Math.tanh(this.value)**2) * out.grad;
		}
		return out;
	}
	backward(){
		this.#grad = 1;
		const sortedDependencies = topologicalSort(this, x => x.children).reverse();
		for(const node of sortedDependencies){
			node[symbolBackward]();
		}	
	}
	toString() {
		return `<${this.#label ? `${this.#label} : ` : ""}${this.#value.toString()}>`;
	}
	set [symbolBackward](val){
		this.#backward = val;
	}
	get [symbolBackward](){
		return this.#backward;
	}
	set value(val){
		this.#value = val;
	}
	get value() {
		return this.#value;
	}
	get children() {
		return this.#children;
	}
	get op() {
		return this.#op;
	}
	get grad(){
		return this.#grad;
	}
	set grad(val){
		this.#grad = val;
	}
	set label(val){
		this.#label = val;
	}
	get label(){
		return this.#label;
	}
}

export function asTensors(numbers){
	return numbers.map(n => new Tensor(n));
}