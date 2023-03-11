export function meanSquared(expected, actual) {
	return actual.sub(expected).pow(2);
}
export function sum(values){
	return values.reduce((sum, v) => sum.add(v));
}
export function aggregateMeansSquared(expecteds, actuals){
	return sum(actuals.map((a,i) => meanSquared(expecteds[i], a)));
}