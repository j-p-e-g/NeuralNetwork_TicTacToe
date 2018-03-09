#pragma once

namespace Math
{
    class ActivationFunctions
    {
    public:
        static double identity(double val, bool derivative = false);
        static double sigmoid(double val, bool derivative = false);
        static double hyperbolicTan(double val, bool derivative = false);
        static double relu(double val, bool derivative = false);
        static double leakyRelu(double val, bool derivative = false);
    };
}