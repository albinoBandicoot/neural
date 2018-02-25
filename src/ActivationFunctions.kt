interface ActivationFunction {
    operator fun invoke (x: Double): Double
    fun derivative(x: Double): Double
}

object Sigmoid: ActivationFunction {
    override fun invoke(x: Double): Double = 1/(1+Math.exp(-x))
    override fun derivative(x: Double): Double = invoke(x)*(1.0 - invoke(x))
}

object ReLU: ActivationFunction {
    override fun invoke(x: Double): Double = if (x < 0) 0.0 else x
    override fun derivative(x: Double): Double = if (x < 0) 0.0 else 1.0
}