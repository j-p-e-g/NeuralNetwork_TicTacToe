#include "stdafx.h"
#include "CppUnitTest.h"
#include "Math/ActivationFunctions.h"

namespace ActivationFunctionsTest
{
    using namespace Microsoft::VisualStudio::CppUnitTestFramework;
    using namespace Math;

    TEST_CLASS(ActivationFunctions_Test)
    {
    public:
        //------------------------------------
        // activation functions
        //------------------------------------
        TEST_METHOD(ActivationFunctions_identity)
        {
            // all values remain unchanged
            Assert::AreEqual(-186.2696, ActivationFunctions::identity(-186.2696), 0.0001);
            Assert::AreEqual(-0.00083, ActivationFunctions::identity(-0.00083), 0.0001);
            Assert::AreEqual(0, ActivationFunctions::identity(0), 0.0001);
            Assert::AreEqual(0.00251, ActivationFunctions::identity(0.00251), 0.0001);
            Assert::AreEqual(96785.573265, ActivationFunctions::identity(96785.573265), 0.0001);
        }

        TEST_METHOD(ActivationFunctions_relu)
        {
            // negative values are capped to zero
            Assert::AreEqual(0, ActivationFunctions::relu(-21907.584), 0.0001);
            Assert::AreEqual(0, ActivationFunctions::relu(-0.01586), 0.0001);

            // non-negative values remain unchanged
            Assert::AreEqual(0, ActivationFunctions::relu(0), 0.0001);
            Assert::AreEqual(0.000009, ActivationFunctions::relu(0.000009), 0.0001);
            Assert::AreEqual(7823.52, ActivationFunctions::relu(7823.52), 0.0001);
        }

        TEST_METHOD(ActivationFunctions_leakyRelu)
        {
            // non-negative values remain unchanged
            Assert::AreEqual(0, ActivationFunctions::leakyRelu(0), 0.0001);
            Assert::AreEqual(0.00723, ActivationFunctions::leakyRelu(0.00723), 0.0001);
            Assert::AreEqual(999999.199, ActivationFunctions::leakyRelu(999999.199), 0.0001);

            // negative values are squashed to a much smaller value
            const double relu1 = ActivationFunctions::leakyRelu(-194.5);
            const double relu2 = ActivationFunctions::leakyRelu(-8986.225);
            const double relu3 = ActivationFunctions::leakyRelu(-0.0000623);
            Assert::AreEqual(true, -194.5 < relu1);
            Assert::AreEqual(true, -8986.225 < relu2);
            Assert::AreEqual(true, -0.0000623 < relu3);

            // ... but the order is still the same
            Assert::AreEqual(true, relu2 < relu1);
            Assert::AreEqual(true, relu1 < relu3);
            Assert::AreEqual(true, relu3 < 0);
        }

        TEST_METHOD(ActivationFunctions_sigmoid)
        {
            // values are squashed to a range between [0, 1]
            const double sigm1 = ActivationFunctions::sigmoid(-255298.2);
            const double sigm2 = ActivationFunctions::sigmoid(-86.76);
            const double sigm3 = ActivationFunctions::sigmoid(-0.0826);
            const double sigm4 = ActivationFunctions::sigmoid(0);
            const double sigm5 = ActivationFunctions::sigmoid(0.000211);
            const double sigm6 = ActivationFunctions::sigmoid(86.76);
            const double sigm7 = ActivationFunctions::sigmoid(1000000.0);

            // ... but the order is still the same
            Assert::AreEqual(true, 0 <= sigm1);
            Assert::AreEqual(true, sigm1 <= sigm2);
            Assert::AreEqual(true, sigm2 <= sigm3);
            Assert::AreEqual(true, sigm3 <= sigm4);
            Assert::AreEqual(true, sigm4 <= sigm5);
            Assert::AreEqual(true, sigm5 <= sigm6);
            Assert::AreEqual(true, sigm6 <= sigm7);
            Assert::AreEqual(true, sigm7 <= 1);

            // while close values may result in identical results, these should be truly smaller
            Assert::AreEqual(true, sigm1 < sigm4);
            Assert::AreEqual(true, sigm4 < sigm7);
        }

        TEST_METHOD(ActivationFunctions_hyperbolicTan)
        {
            // values are squashed to a range between [-1, 1]
            const double tanh1 = ActivationFunctions::hyperbolicTan(-4265.003);
            const double tanh2 = ActivationFunctions::hyperbolicTan(-1.0);
            const double tanh3 = ActivationFunctions::hyperbolicTan(-0.0000001);
            const double tanh4 = ActivationFunctions::hyperbolicTan(0);
            const double tanh5 = ActivationFunctions::hyperbolicTan(0.000211);
            const double tanh6 = ActivationFunctions::hyperbolicTan(27.82);
            const double tanh7 = ActivationFunctions::hyperbolicTan(53962.99);

            // ... but the order is still the same
            Assert::AreEqual(true, -1 <= tanh1);
            Assert::AreEqual(true, tanh1 <= tanh2);
            Assert::AreEqual(true, tanh2 <= tanh3);
            Assert::AreEqual(true, tanh3 <= tanh4);
            Assert::AreEqual(true, tanh4 <= tanh5);
            Assert::AreEqual(true, tanh5 <= tanh6);
            Assert::AreEqual(true, tanh6 <= tanh7);
            Assert::AreEqual(true, tanh7 <= 1);

            // while close values may result in identical results, these should be truly smaller
            Assert::AreEqual(true, tanh1 < tanh4);
            Assert::AreEqual(true, tanh4 < tanh7);
        }

        //------------------------------------
        // derived activation functions
        //------------------------------------
        TEST_METHOD(ActivationFunctions_derivedIdentity)
        {
            // all values map to 1
            Assert::AreEqual(1, ActivationFunctions::identity(-9962.6536, true), 0.0001);
            Assert::AreEqual(1, ActivationFunctions::identity(-0.0714, true), 0.0001);
            Assert::AreEqual(1, ActivationFunctions::identity(0, true), 0.0001);
            Assert::AreEqual(1, ActivationFunctions::identity(0.00005, true), 0.0001);
            Assert::AreEqual(1, ActivationFunctions::identity(7632.24, true), 0.0001);
        }

        TEST_METHOD(ActivationFunctions_derivedRelu)
        {
            // negative values map to 0
            Assert::AreEqual(0, ActivationFunctions::relu(-1468.82, true), 0.0001);
            Assert::AreEqual(0, ActivationFunctions::relu(-0.000014, true), 0.0001);

            // non-negative values map to 1
            Assert::AreEqual(1, ActivationFunctions::relu(0, true), 0.0001);
            Assert::AreEqual(1, ActivationFunctions::relu(0.00252, true), 0.0001);
            Assert::AreEqual(1, ActivationFunctions::relu(16002.5, true), 0.0001);
        }

        TEST_METHOD(ActivationFunctions_derivedLeakyRelu)
        {
            // non-negative values map to 1
            Assert::AreEqual(1, ActivationFunctions::leakyRelu(0, true), 0.0001);
            Assert::AreEqual(1, ActivationFunctions::leakyRelu(0.02006, true), 0.0001);
            Assert::AreEqual(1, ActivationFunctions::leakyRelu(34615.12, true), 0.0001);

            // negative values map to the leaky modifier (0 < m << 1)
            const double derivedRelu1 = ActivationFunctions::leakyRelu(-28968.3, true);
            const double derivedRelu2 = ActivationFunctions::leakyRelu(-0.007, true);
            Assert::AreEqual(true, derivedRelu1 == derivedRelu2);
            Assert::AreEqual(true, 0 < derivedRelu1 && derivedRelu1 < 1);
        }

        TEST_METHOD(ActivationFunctions_derivedSigmoid)
        {
            // the derivative of the sigmoid function looks Gaussian, centered around 0
            // values are at least 0, and max. some value < 1
            const double derivedSigmoidLeft = ActivationFunctions::sigmoid(-255298.2, true);
            const double derivedSigmoidCenterLeft = ActivationFunctions::sigmoid(-0.0826, true);
            const double derivedSigmoidCenter = ActivationFunctions::sigmoid(0, true);
            const double derivedSigmoidCenterRight = ActivationFunctions::sigmoid(0.000211, true);
            const double derivedSigmoidRight = ActivationFunctions::sigmoid(1000000.0, true);

            Assert::AreEqual(0, derivedSigmoidLeft, 0.0001);
            Assert::AreEqual(0, derivedSigmoidRight, 0.0001);
            Assert::AreEqual(true, derivedSigmoidCenter > derivedSigmoidCenterLeft);
            Assert::AreEqual(true, derivedSigmoidCenter > derivedSigmoidCenterRight);
            Assert::AreEqual(true, derivedSigmoidCenterLeft >= derivedSigmoidLeft);
            Assert::AreEqual(true, derivedSigmoidCenterRight >= derivedSigmoidRight);
        }

        TEST_METHOD(ActivationFunctions_derivedHyperbolicTan)
        {
            // the derivative of tanh is also centered around 0 but the slope is much steeper
            // values are in [0, 1]
            const double derivedTanhLeft = ActivationFunctions::hyperbolicTan(-863200.54, true);
            const double derivedTanhCenterLeft = ActivationFunctions::hyperbolicTan(-0.00035, true);
            const double derivedTanhCenter = ActivationFunctions::hyperbolicTan(0, true);
            const double derivedTanhCenterRight = ActivationFunctions::hyperbolicTan(0.009999, true);
            const double derivedTanhRight = ActivationFunctions::hyperbolicTan(11111.8, true);

            Assert::AreEqual(0, derivedTanhLeft, 0.0001);
            Assert::AreEqual(0, derivedTanhRight, 0.0001);
            Assert::AreEqual(1, derivedTanhCenter, 0.001);
            Assert::AreEqual(true, derivedTanhCenter > derivedTanhCenterLeft);
            Assert::AreEqual(true, derivedTanhCenter > derivedTanhCenterRight);
            Assert::AreEqual(true, derivedTanhCenterLeft >= derivedTanhLeft);
            Assert::AreEqual(true, derivedTanhCenterRight >= derivedTanhRight);
        }
    };
}
