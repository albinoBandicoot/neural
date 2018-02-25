class Layer(val size: Int, nodeBuilder: (Int) -> Node) {
    val nodes = Array<Node>(size, nodeBuilder)
    operator fun get(i: Int): Node = nodes[i]

    fun feed () {
        for (n in nodes) n.feed()
    }

    fun backpropagate(learningRate: Double) {
        for (n in nodes) n.backpropagate(learningRate)
    }

    fun weightUpdate(multiplier: Double) {
        for (n in nodes) n.weightUpdate(multiplier)
    }

    fun setInputSource(source: InputSource) {
        for (n in nodes) (n as InputNode).input = source
    }

    fun setNextLayer(layer: Layer) {
        for (n in nodes.filterIsInstance<HiddenNode>()) {
            n.nextLayer = layer
        }
    }

    override fun toString(): String {
        val sb = StringBuilder("LAYER of $size nodes:\n")
        for (i in nodes.indices) {
            sb.append(" Node $i: ${nodes[i]}\n")
        }
        return sb.toString()
    }

    companion object {
        fun inputLayer (size: Int, input: TrainingSource): Layer {
            return Layer(size, {index -> InputNode(index, input)})
        }

        fun hiddenLayer (size: Int, prevLayer: Layer, func: ActivationFunction = ReLU): Layer {
            return Layer(size, {index -> HiddenNode(prevLayer, null, func)})
        }
    }
}
