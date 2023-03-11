import { assertEquals, assertAlmostEquals } from "https://deno.land/std@0.173.0/testing/asserts.ts";
import { Tensor } from "./autograd.js";

Deno.test("Value add", () => {
	const a = new Tensor(2);
	const b = new Tensor(3);

	assertEquals(a.add(b).value, 5);
});
Deno.test("Value add (wrap)", () => {
	const a = new Tensor(2);

	assertEquals(a.add(4).value, 6);
});
Deno.test("Value add (gradient)", () => {
	const a = new Tensor(2);
	const b = new Tensor(3);

	const c = a.add(b);
	c.backward();
	
	assertEquals(a.grad, 1);
	assertEquals(b.grad, 1);
});

Deno.test("Value multiply", () => {
	const a = new Tensor(2);
	const b = new Tensor(3);

	assertEquals(a.mul(b).value, 6);
});

Deno.test("Value multiply (wrap)", () => {
	const a = new Tensor(2);

	assertEquals(a.mul(3).value, 6);
});

Deno.test("Value subtract", () => {
	const a = new Tensor(2);
	const b = new Tensor(3);

	assertEquals(a.sub(b).value, -1);
});

Deno.test("Value subtract (wrap)", () => {
	const a = new Tensor(2);

	assertEquals(a.sub(3).value, -1);
});
Deno.test("Value subtract (gradient)", () => {
	const a = new Tensor(2);
	const b = new Tensor(3);

	const c = a.sub(b);
	c.backward();

	assertEquals(a.grad, 1);
	assertEquals(b.grad, -1);
});

Deno.test("Value divide", () => {
	const a = new Tensor(6);
	const b = new Tensor(3);

	assertEquals(a.div(b).value, 2);
});

Deno.test("Value divide (wrap)", () => {
	const a = new Tensor(6);

	assertEquals(a.div(3).value, 2);
});
Deno.test("Value divide (gradient)", () => {
	const a = new Tensor(2);
	const b = new Tensor(3);

	const c = a.div(b);
	c.backward();

	assertEquals(a.grad, 0.3333333333333333);
	assertEquals(b.grad, -0.2222222222222222);
});

Deno.test("Value negate", () => {
	const a = new Tensor(2);

	assertEquals(a.neg().value, -2);
});

Deno.test("Value pow", () => {
	const a = new Tensor(2);

	assertEquals(a.pow(3).value, 8);
});

Deno.test("Value exp", () => {
	const a = new Tensor(2);

	assertEquals(a.exp().value, Math.exp(2));
});

Deno.test("Value tahn", () => {
	const a = new Tensor(2);

	assertEquals(a.tanh().value, Math.tanh(2));
});

Deno.test("Should backprop", () => {
	const x1 = new Tensor(2, [], { label: "x1" });
	const x2 = new Tensor(0, [], { label: "x2" });
	const w1 = new Tensor(-3, [], { label: "w1" });
	const w2 = new Tensor(1, [], { label: "w2" });
	const b = new Tensor(6.8813735870195432, [], { label: "b" });
	const x1w1 = x1.mul(w1); x1w1.label = "x1w1";
	const x2w2 = x2.mul(w2); x2w2.label = "x2w2";
	const x1w1x2w2 = x1w1.add(x2w2); x1w1x2w2.label = "x1w1x2w2";
	const n = x1w1x2w2.add(b); n.label = "n";
	const o = n.tanh(); o.label = "o";

	o.backward();

	assertAlmostEquals(x1.grad, -1.5);
	assertAlmostEquals(x2.grad, 0.5);
	assertAlmostEquals(w1.grad, 1);
	assertAlmostEquals(w2.grad, 0);
	assertAlmostEquals(x1w1.grad, 0.5);
	assertAlmostEquals(x2w2.grad, 0.5);
	assertAlmostEquals(x1w1x2w2.grad, 0.5);
	assertAlmostEquals(b.grad, 0.5);
	assertAlmostEquals(n.grad, 0.5);
	assertAlmostEquals(o.grad, 1);
});

Deno.test("Should backprop (more ops)", () => {
	const x1 = new Tensor(2, [], { label: "x1" });
	const x2 = new Tensor(0, [], { label: "x2" });
	const w1 = new Tensor(-3, [], { label: "w1" });
	const w2 = new Tensor(1, [], { label: "w2" });
	const b = new Tensor(6.8813735870195432, [], { label: "b" });
	const x1w1 = x1.mul(w1); x1w1.label = "x1w1";
	const x2w2 = x2.mul(w2); x2w2.label = "x2w2";
	const x1w1x2w2 = x1w1.add(x2w2); x1w1x2w2.label = "x1w1x2w2";
	const n = x1w1x2w2.add(b); n.label = "n";
	//manual tahn
	const e = n.mul(2).exp(); e.label = "e";
	const o = e.sub(1).div(e.add(1));

	o.backward();

	assertAlmostEquals(x1.grad, -1.5);
	assertAlmostEquals(x2.grad, 0.5);
	assertAlmostEquals(w1.grad, 1);
	assertAlmostEquals(w2.grad, 0);
	assertAlmostEquals(x1w1.grad, 0.5);
	assertAlmostEquals(x2w2.grad, 0.5);
	assertAlmostEquals(x1w1x2w2.grad, 0.5);
	assertAlmostEquals(b.grad, 0.5);
	assertAlmostEquals(n.grad, 0.5);
	assertAlmostEquals(o.grad, 1);
});

Deno.test("Should backprop (duplicate nodes simple)", () => {
	const a = new Tensor(3, [], { label: "a"});
	const b = a.add(a); b.label = "b";

	b.backward();

	assertAlmostEquals(b.grad, 1);
	assertAlmostEquals(a.grad, 2);
})

Deno.test("Should backprop (duplicate nodes)", () => {
	const a = new Tensor(-2, [], { label: "a" });
	const b = new Tensor(3, [], { label: "b" });
	const c = a.mul(b); c.label = "c"; //-6
	const d = a.add(b); d.label = "d"; //1
	const e = c.mul(d); e.label = "e"; //-6

	e.backward();

	assertAlmostEquals(a.grad, -3);
	assertAlmostEquals(b.grad, -8);
	assertAlmostEquals(c.grad, 1);
	assertAlmostEquals(d.grad, -6);
	assertAlmostEquals(e.grad, 1);
});