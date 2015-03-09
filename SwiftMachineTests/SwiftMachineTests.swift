//
//  SwiftMachineTests.swift
//  SwiftMachineTests
//
//  Created by Akio Yasui on 3/7/15.
//  Copyright (c) 2015 Akio Yasui. All rights reserved.
//

import Cocoa
import XCTest

class SwiftMachineTests: XCTestCase {

	func testAND() {
		let perceptron = Perceptron()

		var exampleInput = [[Double]]()
		exampleInput.append([-1.0, -1.0])
		exampleInput.append([-1.0,  1.0])
		exampleInput.append([ 1.0, -1.0])
		exampleInput.append([ 1.0,  1.0])

		var exampleOutput = [Double]()
		exampleOutput.append(-1.0)
		exampleOutput.append(-1.0)
		exampleOutput.append(-1.0)
		exampleOutput.append( 1.0)

		perceptron.train(inputs: exampleInput, outputs: exampleOutput, learningRate: 0.5, epsilon: 0.0001)

		for i in indices(exampleInput) {
			let input = exampleInput[i]
			let expected = exampleOutput[i]
			let output = perceptron.test(input)
			XCTAssertEqual(expected, output, "the output must be equal to what expected")
		}
	}

	func testOR() {
		let perceptron = Perceptron()

		var exampleInput = [[Double]]()
		exampleInput.append([-1.0, -1.0])
		exampleInput.append([-1.0,  1.0])
		exampleInput.append([ 1.0, -1.0])
		exampleInput.append([ 1.0,  1.0])

		var exampleOutput = [Double]()
		exampleOutput.append(-1.0)
		exampleOutput.append( 1.0)
		exampleOutput.append( 1.0)
		exampleOutput.append( 1.0)

		perceptron.train(inputs: exampleInput, outputs: exampleOutput, learningRate: 0.5, epsilon: 0.0001)

		for i in indices(exampleInput) {
			let input = exampleInput[i]
			let expected = exampleOutput[i]
			let output = perceptron.test(input)
			XCTAssertEqual(expected, output, "the output must be equal to what expected")
		}
	}

	func testQuadrant() {
		let COUNT = 100

		let q1 = (1 ..< COUNT).map({ _ -> [Double] in
			var coordinates = [Double]()
			coordinates.append(Double(arc4random_uniform(UInt32.max)) / Double(UInt32.max))
			coordinates.append(Double(arc4random_uniform(UInt32.max)) / Double(UInt32.max))
			coordinates.append(Double(arc4random_uniform(UInt32.max)) / Double(UInt32.max))
			return coordinates
		})
		let q2 = (1 ..< COUNT).map({ _ -> [Double] in
			var coordinates = [Double]()
			coordinates.append(Double(arc4random_uniform(UInt32.max)) / Double(UInt32.max) * -1.0)
			coordinates.append(Double(arc4random_uniform(UInt32.max)) / Double(UInt32.max) * -1.0)
			coordinates.append(Double(arc4random_uniform(UInt32.max)) / Double(UInt32.max) * -1.0)
			return coordinates
		})

		let inputs = q1 + q2
		let outputs = [Double](count: q1.count, repeatedValue: 1.0) + [Double](count: q2.count, repeatedValue: -1.0)

		let p = Perceptron()

		self.measureBlock {
			p.train(
				inputs: inputs,
				outputs: outputs,
				learningRate: 0.5,
				epsilon: 1E-10)
			return
		}

		q1.map({ (input) -> Void in
			XCTAssertEqualWithAccuracy(1.0, p.test(input), 1E-10, "inputs must be correctly classified")
		})
		q2.map({ (input) -> Void in
			XCTAssertEqualWithAccuracy(-1.0, p.test(input), 1E-10, "inputs must be correctly classified")
		})
	}
	
}
