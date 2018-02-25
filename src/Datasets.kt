interface Dataset {
    operator fun get(i: Int): Datapoint
    fun size(): Int
}

interface TrainingDataset: Dataset {
    override operator fun get(i: Int): TrainingDatapoint
}

class ArrayDataset(val data: Array<TrainingDatapoint>): TrainingDataset {
    override fun get(i: Int) = data[i]
    override fun size(): Int = data.size
}

interface InputSource {
    fun hasNext(): Boolean
    fun advance(): Unit
    fun reset(): Unit
    fun getInput(i: Int): Double
    fun inputs(): Array<Double>
}

interface TrainingSource: InputSource {
    fun getOutput(i: Int): Double
}

open class DatasetInputSource(val data: Dataset): InputSource {
    var idx: Int = 0
    override fun advance() { idx++ }
    override fun reset() { idx = 0 }
    override fun hasNext(): Boolean = idx < data.size()-1
    override fun getInput(i: Int) = data[idx].getInput(i)
    override fun inputs() = Array<Double>(data[idx].numInputs(), {i -> data[idx].getInput(i)})
}

class DatasetTrainingSource(data: TrainingDataset): DatasetInputSource(data), TrainingSource {
    override fun getOutput(i: Int) = (data[idx] as TrainingDatapoint).getOutput(i)
}

interface Datapoint {
    fun getInput(i: Int): Double
    fun numInputs(): Int
}

interface TrainingDatapoint: Datapoint {
    fun getOutput(i: Int): Double
    fun numOutputs(): Int
}

class ArrayDatapoint(val data: DoubleArray): Datapoint {
    override fun getInput(i: Int) = data[i]
    override fun numInputs(): Int = data.size
}

class BinaryDatapoint(val inputValue: Datapoint, val output: Boolean): TrainingDatapoint, Datapoint by inputValue {
    override fun getOutput(i: Int) = if (output) 1.0 else 0.0
    override fun numOutputs(): Int = 1
}

class LabeledDatapoint(val value: Datapoint, val category: Int, val nCategories: Int): TrainingDatapoint, Datapoint by value {
    override fun getOutput(i: Int) = if (i == category) 1.0 else 0.0
    override fun numOutputs(): Int = nCategories
}

interface DatapointFactory {
    fun getDatapoint(): TrainingDatapoint
    fun getDataset(n: Int): TrainingDataset = ArrayDataset(Array<TrainingDatapoint>(n, { getDatapoint() }))
}

class FunctionDatapointFactory(val func: (DoubleArray) -> Double, val pointGen: PointGenerator, val arity: Int): DatapointFactory {
    var idx = 0
    override fun getDatapoint(): BinaryDatapoint {
        val x = pointGen.generate(idx++, arity)
        return BinaryDatapoint(ArrayDatapoint(x), func(x) > 0)
    }
}

interface PointGenerator {
    fun generate (index: Int, arity: Int): DoubleArray
}

class RandomPointGenerator(val bound: Double): PointGenerator {
    override fun generate(index: Int, arity: Int) = DoubleArray(arity, { (Network.random.nextDouble() - 0.5) * 2 * bound })
}
class GridPointGenerator(val xpix: Int, val ypix: Int, val xsize: Double, val ysize: Double): PointGenerator {
    override fun generate(index: Int, arity: Int) =
            doubleArrayOf(((index % xpix)/xpix.toDouble() - 0.5) * xsize, ((index / xpix)/ypix.toDouble() - 0.5)*ysize)
}
