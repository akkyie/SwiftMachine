//
//  SwiftMachine.swift
//  SwiftMachine
//
//  Created by Akio Yasui on 3/7/15.
//  Copyright (c) 2015 Akio Yasui. All rights reserved.
//

import Foundation

import Accelerate

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
	var weight: la_object_t

	init(activation: Function, weight: la_object_t) {
		self.activation = activation
		self.weight = weight
	}

	func propagate(input: [Double]) -> Double {
		assert(la_count_t(input.count) == la_matrix_rows(self.weight), "the number of input must equal to the number of weight")
		let input = la_matrix_from_double_buffer(input, la_count_t(input.count), 1, 1, la_hint_t(LA_NO_HINT), la_attribute_t(LA_DEFAULT_ATTRIBUTES))
		let product = la_matrix_product(input, self.weight)
		let identity = la_matrix_from_double_buffer([Double](count: Int(la_matrix_rows(product)), repeatedValue: 1.0), 1, la_matrix_rows(product), la_matrix_rows(product), la_hint_t(LA_NO_HINT), la_attribute_t(LA_DEFAULT_ATTRIBUTES))
		let sumMatrix = la_matrix_product(product, identity)
		var elements = [Double](count: Int(la_matrix_rows(sumMatrix) * la_matrix_cols(sumMatrix)), repeatedValue: Double(0.0))
		let status = la_matrix_to_double_buffer(&elements, la_matrix_rows(sumMatrix), sumMatrix)
		assert(status == la_status_t(LA_SUCCESS), "status must be success")
		return self.activation(elements[0])
	}

}

public class Perceptron {

	private var neuron: Neuron! = nil

	public init() {

	}

	public func train(#inputs: [[Double]], outputs: [Double], learningRate: Double, epsilon: Double) {
		assert(inputs.count == outputs.count, "the number of input must equal to the number of output")

		let inputs = inputs.map({ [1.0] + $0 })

		self.neuron = {
			var weight = inputs[0].map({ (_, index) -> Double in
				return Double(arc4random_uniform(UInt32.max)) / Double(UInt32.max)
			})
			return Neuron(activation: Functions.Step(t: 0.0).function, weight: la_matrix_from_double_buffer(weight, la_count_t(weight.count), 1, 1, la_hint_t(LA_NO_HINT), la_attribute_t(LA_DEFAULT_ATTRIBUTES)))
			}()

		var epoch = 0;
		var oldWeight: la_object_t
		do {
			oldWeight = self.neuron.weight
			for i in 0 ..< inputs.count {
				let input = la_matrix_from_double_buffer(inputs[i], la_count_t(inputs[i].count), 1, 1, la_hint_t(LA_NO_HINT), la_attribute_t(LA_DEFAULT_ATTRIBUTES))
				let expected = outputs[i]
				let output = self.neuron.propagate(inputs[i])
				if abs(expected - output) >= abs(epsilon) {
					let error = la_scale_with_double(input, learningRate * (expected - output))
					let newWeight = la_sum(self.neuron.weight, error)
					self.neuron.weight = newWeight
				}
			}
		} while la_norm_as_double(la_difference(oldWeight, self.neuron.weight), la_norm_t(LA_L1_NORM)) > 0
	}

	public func test(input: [Double]) -> Double {
		assert(self.neuron != nil, "test must not be called before the network is trained")
		return self.neuron.propagate([1.0] + input)
	}
	
}


