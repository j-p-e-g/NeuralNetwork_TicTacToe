#include "stdafx.h"
#include <algorithm>

#include "ActivationFunctions.h"

namespace Math
{
    const double LEAKY_RELU_MULTIPLIER = 0.01;

    double ActivationFunctions::identity(double val, bool derivative)
    {
        if (derivative)
        {
            return 1;
        }
        return val;
    }

    double ActivationFunctions::sigmoid(double val, bool derivative)
    {
        // A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve. 
        // Often, sigmoid function refers to the special case of the logistic function  defined by the formula
        // S(x) = 1 / (1 + e^(-1x)) = e^x / (1 + e^x)
        // Note that the sigmoid output is centered around 0.5 and its range is in [0, 1].

        if (derivative)
        {
            const double result = sigmoid(val, false);
            return result * (1 - result);
        }

        const double e = std::exp(val);
        if (e == INFINITY)
        {
            return 1;
        }

        return e / (e + 1);
    }

    double ActivationFunctions::hyperbolicTan(double val, bool derivative)
    {
        // Hyperbolic Tangent Activation Function: 
        // The tanh(z) function is a rescaled version of the sigmoid, and its output range is[-1, 1] instead of[0, 1].

        if (derivative)
        {
            const double result = hyperbolicTan(val, false);
            return 1 - result * result;
        }

        return 2 * sigmoid(2 * val) - 1;
    }

    double ActivationFunctions::relu(double val, bool derivative)
    {
        // ReLU = rectified linear unit
        // In the context of artificial neural networks, the rectifier is an activation function defined as the positive part of its argument:
        // f(x) = x^(+) = max(0, x)

        if (derivative)
        {
            if (val < 0)
            {
                return 0;
            }

            return 1;
        }

        return std::max<double>(0, val);
    }

    double ActivationFunctions::leakyRelu(double val, bool derivative)
    {
        // Leaky ReLUs are one attempt to fix the "dying ReLU" problem. 
        // Instead of the function being zero when x < 0, a leaky ReLU will instead have a small slope (of 0.01, or so) to the left of the y axis.

        if (derivative)
        {
            if (val < 0)
            {
                return LEAKY_RELU_MULTIPLIER;
            }

            return 1;
        }

        if (val < 0)
        {
            return val * LEAKY_RELU_MULTIPLIER;
        }

        return val;
    }
}
