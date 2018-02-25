import java.util.*

class Network (val layerSizes: Array<Int>, val trainingSource: TrainingSource, var learningRate: Double, func: ActivationFunction){
    val layers = mutableListOf<Layer>()
    val outputLayer: Layer get() = layers.last()
    val inputLayer: Layer get() = layers.first()

    init {
        layers.add(Layer.inputLayer(layerSizes[0], trainingSource))
        for (i in 1..layerSizes.size-1) {
            layers.add(Layer.hiddenLayer(layerSizes[i], layers.last(), func))
        }
        for (i in 1..layerSizes.size-2) {
            layers[i].setNextLayer(layers[i+1])
        }
    }

    fun getOutput(i: Int = 0) = outputLayer[i].activation
    operator fun get(layer: Int, node: Int) = layers[layer].nodes[node].activation

    private fun feed () {
        for (layer in layers) {
            layer.feed()
        }
    }

    private fun backpropagate() {
        for (layer in layers.reversed()) {
            layer.backpropagate(learningRate)
        }
    }

    private fun weightUpdate(multiplier: Double) {
        for (layer in layers) layer.weightUpdate(multiplier)
    }

    private fun setOutputDeltas(): Double {
        fun Double.squared() = this*this
        var err = 0.0
        for (i in outputLayer.nodes.indices) {
            val diff = trainingSource.getOutput(i) - outputLayer[i].activation
            outputLayer[i].delta = diff //* Sigmoid.derivative(outputLayer[i].net)
            err += diff.squared()
        }
        return err
    }

    private fun classifyBinary() = if (outputLayer[0].activation > 0.5) 1 else 0

    private fun argmaxOutput(): Int {
        var best = 0.0
        var idx = 0
        for (i in outputLayer.nodes.indices) {
            val out = outputLayer[i].activation
            if (out > best) {
                best = out
                idx = i
            }
        }
        return idx
    }


    fun trainEpoch() {
        var err = 0.0
        var ncorrect = 0.0
        fun trainOnePoint(): Boolean {
//            println("Will feed point: ${trainingSource.inputs().joinToString(separator = ", ", prefix="(", postfix=")")}")
            feed()
            err += setOutputDeltas()
            ncorrect += if (classifyBinary() == trainingSource.getOutput(0).toInt()) 1 else 0
            backpropagate()
            weightUpdate(1.0)
//            println(this)
            if (trainingSource.hasNext()) {
                trainingSource.advance()
                return true
            } else return false
        }
        var n = 0
        while(trainOnePoint()) n++
        println("Training error was: $err; rate = ${ncorrect/n}")
    }

    fun train(numEpochs: Int) {
        inputLayer.setInputSource(trainingSource)
        for (i in 0..numEpochs-1) {
            trainingSource.reset()
            trainEpoch()
        }
    }

    fun test(source: TrainingSource, resultCallback: (Int) -> Unit = {}) {
        inputLayer.setInputSource(source)
        var err = 0.0
        var ncorrect = 0.0
        var n = 0
        while (source.hasNext()) {
            feed()
            err += setOutputDeltas()    // don't need to set deltas here, but convenient
            if (classifyBinary() == source.getOutput(0).toInt()) {
                ncorrect++
            } else {
            }
            resultCallback(n)
            n++
            source.advance()
        }
        source.reset()
        println("Test error was: ${err/n}")
        println("Classification rate: ${ncorrect/n}")
    }

    override fun toString() = layers.joinToString(separator = "\n")

    companion object {
        val random = Random(751)
    }
}