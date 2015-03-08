//
//  SwiftMachine.swift
//  SwiftMachine
//
//  Created by Akio Yasui on 3/7/15.
//  Copyright (c) 2015 Akio Yasui. All rights reserved.
//

import Foundation

private typealias Function = (Double) -> Double

private enum Functions {

	case Step(t: Double)
	case Sigmoid
	case Rectifier
	case SoftMax

	var function: Function {
		switch self {
		case let .Step(t):
			return { $0 >= t ? 1.0 : -1.0 }
		case .Sigmoid:
			return { 1 / (1 + exp($0)) }
		case .Rectifier:
			return { max(0, $0) }
		case .SoftMax:
			return { log(1 + exp($0)) }
		}
	}

}

private extension Array {
	func map<U>(transform: (T, Int) -> U) -> [U] {
		var v = [U]()
		for i in 0 ..< self.count {
			v.append(transform(self[i], i))
		}
		return v
	}
}

private struct Math {

	static func map<T, U>(transform: ([T]) -> U, arrays: [T]...) -> [U] {
		return (0 ..< arrays.map({ $0.count })
			.reduce(Int.max, combine: { min($0, $1) }))
			.map{ i -> [T] in arrays.map({ $0[i] })}
			.map(transform)
	}

	static func matrixAdd(a: [Double], _ b: [Double]) -> [Double] {
		return map({ $0[0] + $0[1] }, arrays: a, b)
	}

	static func matrixElementwiseMultiply(a: [Double], _ b: [Double]) -> [Double] {
		return map({ $0[0] * $0[1] }, arrays: a, b)
	}

	static func matrixScalarMultiply(a: [Double], _ s: Double) -> [Double] {
		return map({ $0[0] * s }, arrays: a)
	}

}

private class Neuron {

	var activation: Function
	var weight: [Double]

	init(activation: Function, weight: [Double]) {
		self.activation = activation
		self.weight = weight
	}

	func propagate(input: [Double]) -> Double {
		assert(input.count == weight.count, "the number of input must equal to the number of weight")
		return self.activation(Math.matrixElementwiseMultiply(input, self.weight).reduce(0.0, combine: +))
	}

}

public class Perceptron {

	private var neuron: Neuron! = nil

	public init() {

	}

	public func train(#inputs: [[Double]], outputs: [Double], learningRate: Double, epsilon: Double) -> [Double] {
		assert(inputs.count == outputs.count, "the number of input must equal to the number of output")

		let inputs = inputs.map({ [1.0] + $0 })

		self.neuron = {
			var weight = inputs[0].map({ (_, index) -> Double in
				return Double(arc4random_uniform(UInt32.max)) / Double(UInt32.max)
			})
			return Neuron(activation: Functions.Step(t: 0.0).function, weight: weight)
			}()

		var epoch = 0;
		var oldWeight: [Double]!
		do {
			oldWeight = self.neuron.weight
			for i in 0 ..< inputs.count {
				let input = inputs[i]
				let expected = outputs[i]
				let output = self.neuron.propagate(input)
				if abs(expected - output) >= abs(epsilon) {
					self.neuron.weight = Math.matrixAdd(
						self.neuron.weight,
						Math.matrixScalarMultiply(
							input,
							learningRate * (expected - output)))
				}
			}
		} while oldWeight != self.neuron.weight
		return self.neuron.weight
	}

	public func test(input: [Double]) -> Double {
		assert(self.neuron != nil, "test must not be called before the network is trained")
		return self.neuron.propagate([1.0] + input)
	}
	
}


