abstract class Node {
    var activation: Double = 0.0
    var net: Double = 0.0
    var delta: Double = 0.0

    abstract fun feed(): Unit
    abstract fun backpropagate(learningRate: Double): Unit
    abstract fun weightUpdate(multiplier: Double)
}


class InputNode(val index: Int, var input: InputSource): Node() {
    override fun feed() {
       activation = input.getInput(index)
       net = activation
    }

    override fun backpropagate(learningRate: Double) {}
    override fun weightUpdate(multiplier: Double) {}

    override fun toString(): String {
        return "act: $activation, delta: $delta"
    }
}

class HiddenNode(val prevLayer: Layer, var nextLayer: Layer?, val activationFunction: ActivationFunction): Node() {
    var weights = DoubleArray(prevLayer.size)
    var bias = 0.0

    var deltaW = DoubleArray(prevLayer.size)
    var deltaB = 0.0

    init {
        for (i in weights.indices) {
            weights[i] = Network.random.nextDouble() - 0.5//Math.random() - 0.5
        }
    }

    override fun feed() {
        delta = 0.0
        net = bias
        for (i in weights.indices) {
            net += weights[i] * prevLayer[i].activation
        }
        activation = activationFunction(net)
    }

    override fun backpropagate(learningRate: Double) {
        val c = delta * activationFunction.derivative(net)
        deltaB += c * learningRate
        for (i in weights.indices) {
            deltaW[i] += c * prevLayer[i].activation * learningRate
            prevLayer[i].delta += delta * weights[i]
        }
    }

    override fun weightUpdate(multiplier: Double) {
        delta = 0.0
        bias += deltaB * multiplier
        deltaB = 0.0
        for (i in weights.indices) {
            weights[i] += deltaW[i] * multiplier
            deltaW[i] = 0.0
        }
    }

    override fun toString(): String {
        return "act: $activation; delta: $delta, bias: $bias, weights: ${weights.joinToString(separator = ", ")}"
    }
}