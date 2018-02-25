import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

fun funcToEstimate (x: DoubleArray): Double {
    fun Double.squared() = this*this
//    return x[0] + x[1] - 1
//    return x[0].squared() + x[1].squared() - 1
//    return Math.pow(x[0],16.0) + Math.pow(x[1], 16.0) - 1
    return Math.sin(3*x[0])*(0.5 - Math.cos(2*x[1])) - 0.1
}

fun main (args: Array<String>) {
    val dpf = FunctionDatapointFactory(::funcToEstimate, RandomPointGenerator(1.5), 2)
    val trainDataset = DatasetTrainingSource(dpf.getDataset(9501))
    val testDataset = DatasetTrainingSource(dpf.getDataset(10000))
    val net = Network(arrayOf(2,20,20,1), trainDataset, 0.001, Sigmoid)
    for (i in 0..7) {
        net.train(100)
        net.learningRate *= 0.95
        println(net)
        net.test(testDataset)
    }
    val xs = 100
    val ys = 100
    val testImageSource = DatasetTrainingSource(FunctionDatapointFactory(::funcToEstimate, GridPointGenerator(xs,ys,4.0,4.0), 2).getDataset(xs*ys))
    val img = BufferedImage(xs, ys, 1)

    fun BufferedImage.setPixelToNetOutput(n: Int, net: Network) {
        val out = (net.outputLayer[0].activation - 0.5) * 0.5
        val x = n % width
        val y = n / width
//        println("($x, $y) out = ${net.getOutput()}")
        if (out < -1) {
            setRGB(x, y, 256 * 256 * 255)
        } else if (out > 1) {
            setRGB(x, y, 256 * 255)
        } else {
            val d = (out+1)/2
            val g = Math.sqrt(d)
            val r = Math.sqrt(1-d)
            setRGB(x, y, (r*255).toInt()*256*256 + (g*255).toInt()*256)
        }
    }
    net.test(testImageSource, { i -> img.setPixelToNetOutput(i, net) })
    ImageIO.write(img, "png", File(args[0]))
}