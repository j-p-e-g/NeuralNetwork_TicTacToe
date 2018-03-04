#include "stdafx.h"
#include "CppUnitTest.h"
#include "NeuralNetwork/NodeNetwork.h"

namespace NodeNetworkTest
{
    using namespace Microsoft::VisualStudio::CppUnitTestFramework;
    using namespace NeuralNetwork;

    TEST_CLASS(NodeNetwork_Test)
    {
    public:
        // -----------------------------------------
        // NodeNetwork
        // -----------------------------------------
        TEST_METHOD(NodeNetwork_default)
        {
            NodeNetwork nnet;
            Assert::AreEqual(0, nnet.getNumParameters());
        }

        TEST_METHOD(NodeNetwork_createNetwork_tooFewInputNodes)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 0;
            sizeData.numOutputNodes = 3;
            sizeData.numHiddenNodes = std::vector<int>();

            NodeNetwork nnet;
            Assert::AreEqual(false, nnet.createNetwork(sizeData));
        }

        TEST_METHOD(NodeNetwork_createNetwork_tooFewOutputNodes)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 7;
            sizeData.numOutputNodes = 0;
            sizeData.numHiddenNodes = std::vector<int>();

            NodeNetwork nnet;
            Assert::AreEqual(false, nnet.createNetwork(sizeData));
        }

        TEST_METHOD(NodeNetwork_createNetwork_invalidHiddenLayer)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 3;
            sizeData.numOutputNodes = 2;
            sizeData.numHiddenNodes = std::vector<int>({ 2, -1, 1 });

            NodeNetwork nnet;
            Assert::AreEqual(false, nnet.createNetwork(sizeData));
        }

        TEST_METHOD(NodeNetwork_createNetwork_valid)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 2;
            sizeData.numOutputNodes = 1;
            sizeData.numHiddenNodes = std::vector<int>({ 2, 2 });

            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(sizeData));

            // 5 nodes with bias value + 10 edges with weight value
            Assert::AreEqual(15, nnet.getNumParameters());

            std::vector<double> outputValues;
            nnet.getOutputValues(outputValues);

            Assert::AreEqual(1, static_cast<int>(outputValues.size()));
            Assert::AreEqual(0.0, outputValues[0]);
        }

        TEST_METHOD(NodeNetwork_assignInputValues_tooFew)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 3;
            sizeData.numOutputNodes = 1;
            sizeData.numHiddenNodes = std::vector<int>({ 2, 3 });

            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(sizeData));

            // too few values
            Assert::AreEqual(false, nnet.assignInputValues({-1.0, 2.1}));
        }

        TEST_METHOD(NodeNetwork_assignInputValues_tooMany)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 2;
            sizeData.numOutputNodes = 2;
            sizeData.numHiddenNodes = std::vector<int>({ 4 });

            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(sizeData));

            // too many values
            Assert::AreEqual(false, nnet.assignInputValues({ 0.74, 0.2, -0.3, 1.7 }));
        }

        TEST_METHOD(NodeNetwork_assignInputValues)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 3;
            sizeData.numOutputNodes = 2;
            sizeData.numHiddenNodes = std::vector<int>();

            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(sizeData));
            Assert::AreEqual(true, nnet.assignInputValues({ 5.2, -1.34, 0.25 }));
        }

        TEST_METHOD(NodeNetwork_assignParameters_tooFewParameters)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 2;
            sizeData.numOutputNodes = 1;
            sizeData.numHiddenNodes = std::vector<int>({ 2 });

            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(sizeData));

            // 3 nodes with bias value + 6 edges with weight value
            Assert::AreEqual(9, nnet.getNumParameters());

            // too few parameters
            std::queue<double> params = std::queue<double>({1.2, 3.8, 5.32 });
            Assert::AreEqual(false, nnet.assignParameters(params));
        }

        TEST_METHOD(NodeNetwork_assignParameters_tooManyParameters)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 2;
            sizeData.numOutputNodes = 2;
            sizeData.numHiddenNodes = std::vector<int>();

            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(sizeData));

            // 2 nodes with bias value + 4 edges with weights
            Assert::AreEqual(6, nnet.getNumParameters());

            // too few parameters
            std::queue<double> params = std::queue<double>({ -7.3, 5.33, 8.1, 0.52, 0.0, 9.0, -4.2});
            Assert::AreEqual(false, nnet.assignParameters(params));
        }

        TEST_METHOD(NodeNetwork_assignParameters)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 2;
            sizeData.numOutputNodes = 2;
            sizeData.numHiddenNodes = std::vector<int>();

            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(sizeData));

            // 2 nodes with bias value + 4 edges with weights
            Assert::AreEqual(6, nnet.getNumParameters());

            // should succeed
            std::queue<double> params = std::queue<double>({ 0.52, -12.6, -0.001, -5.38, 9.2, 0.0 });
            Assert::AreEqual(true, nnet.assignParameters(params));
        }

        TEST_METHOD(NodeNetwork_compute_noParams_noInputValues)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 2;
            sizeData.numOutputNodes = 2;
            sizeData.numHiddenNodes = std::vector<int>({ 2 });

            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(sizeData));

            std::vector<double> outputValues;
            nnet.computeValues();
            const int bestIndex = nnet.getOutputValues(outputValues);

            Assert::AreEqual(2, static_cast<int>(outputValues.size()));
            Assert::AreEqual(0.0, outputValues[0]);
            Assert::AreEqual(0.0, outputValues[1]);
            Assert::AreEqual(0, bestIndex);
        }

        TEST_METHOD(NodeNetwork_compute)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 3;
            sizeData.numOutputNodes = 1;
            sizeData.numHiddenNodes = std::vector<int>();

            std::shared_ptr<NodeNetwork> network = std::make_shared<NodeNetwork>();
            network->createNetwork(sizeData);
            network->assignInputValues(std::vector<double>({ 1, -0.5, -1 }));

            std::queue<double> params = std::queue<double>({ 0.2, 0.7, -0.3, 1.2 });
            network->assignParameters(params);
            network->computeValues();

            std::vector<double> outputValues;
            const int bestIndex = network->getOutputValues(outputValues);

            Assert::AreEqual(1, static_cast<int>(outputValues.size()));
            Assert::AreEqual(1.35, outputValues[0], 0.00001);
            Assert::AreEqual(0, bestIndex);
        }

        TEST_METHOD(NodeNetwork_getOutputValues_softMax_single)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 5;
            sizeData.numOutputNodes = 1;
            sizeData.numHiddenNodes = std::vector<int>();

            std::shared_ptr<NodeNetwork> network = std::make_shared<NodeNetwork>();
            network->createNetwork(sizeData);
            network->assignInputValues(std::vector<double>({ 0.7, 0.32, -0.4, 0.01, -0.8 }));

            std::queue<double> params = std::queue<double>({ 0.99, 0.1, 1.2, 0.5, -0.3 });
            network->assignParameters(params);
            network->computeValues();

            std::vector<double> outputValues;
            const int bestIndex = network->getOutputValues(outputValues, true);

            // for a single output value, the soft-max output value is exactly 1
            Assert::AreEqual(1, static_cast<int>(outputValues.size()));
            Assert::AreEqual(1, outputValues[0], 0.00001);
            Assert::AreEqual(0, bestIndex);
        }

        TEST_METHOD(NodeNetwork_getOutputValues_softMax_same)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 2;
            sizeData.numOutputNodes = 2;
            sizeData.numHiddenNodes = std::vector<int>();

            // setup a network in which both output nodes share the same parameters and input values
            std::shared_ptr<NodeNetwork> network = std::make_shared<NodeNetwork>();
            network->createNetwork(sizeData);
            network->assignInputValues(std::vector<double>({ 0.83, 0.83 }));

            std::queue<double> params = std::queue<double>({ -0.6, 0.023, 0.39, -0.6, 0.023, 0.39 });
            network->assignParameters(params);
            network->computeValues();

            std::vector<double> outputValues;
            const int bestIndex = network->getOutputValues(outputValues, true);

            // if there are two output nodes with the exact same input values and edge weights and biases,
            // each gets 0.5 probability
            Assert::AreEqual(2, static_cast<int>(outputValues.size()));
            Assert::AreEqual(0.5, outputValues[0], 0.00001);
            Assert::AreEqual(0.5, outputValues[1], 0.00001);
            Assert::AreEqual(0, bestIndex);
        }

        TEST_METHOD(NodeNetwork_getOutputValues_softMax_different)
        {
            NetworkSizeData sizeData;
            sizeData.numInputNodes = 2;
            sizeData.numOutputNodes = 3;
            sizeData.numHiddenNodes = std::vector<int>();

            std::shared_ptr<NodeNetwork> network = std::make_shared<NodeNetwork>();
            network->createNetwork(sizeData);
            network->assignInputValues(std::vector<double>({ 0.75, 0.123 }));

            std::queue<double> params = std::queue<double>({ 0.2, 0.999, 0, 0.34, -0.8, 1.03, 0.1, -0.154, 0.78 });
            network->assignParameters(params);
            network->computeValues();

            std::vector<double> outputValues;
            const int bestIndex = network->getOutputValues(outputValues, true);

            // the soft-max output values always add up to 1
            Assert::AreEqual(3, static_cast<int>(outputValues.size()));
            Assert::AreEqual(1, outputValues[0] + outputValues[1] + outputValues[2], 0.00001);
        }

        //------------------------------------
        // activation functions
        //------------------------------------
        TEST_METHOD(NodeNetwork_identityActivationFunction)
        {
            // all values remain unchanged
            Assert::AreEqual(-186.2696, NodeNetwork::identityActivationFunction(-186.2696), 0.0001);
            Assert::AreEqual(-0.00083, NodeNetwork::identityActivationFunction(-0.00083), 0.0001);
            Assert::AreEqual(0, NodeNetwork::identityActivationFunction(0), 0.0001);
            Assert::AreEqual(0.00251, NodeNetwork::identityActivationFunction(0.00251), 0.0001);
            Assert::AreEqual(96785.573265, NodeNetwork::identityActivationFunction(96785.573265), 0.0001);
        }

        TEST_METHOD(NodeNetwork_reluActivationFunction)
        {
            // negative values are capped to zero
            Assert::AreEqual(0, NodeNetwork::reluActivationFunction(-21907.584), 0.0001);
            Assert::AreEqual(0, NodeNetwork::reluActivationFunction(-0.01586), 0.0001);

            // non-negative values remain unchanged
            Assert::AreEqual(0, NodeNetwork::reluActivationFunction(0), 0.0001);
            Assert::AreEqual(0.000009, NodeNetwork::reluActivationFunction(0.000009), 0.0001);
            Assert::AreEqual(7823.52, NodeNetwork::reluActivationFunction(7823.52), 0.0001);
        }

        TEST_METHOD(NodeNetwork_leakyReluActivationFunction)
        {
            // non-negative values remain unchanged
            Assert::AreEqual(0, NodeNetwork::leakyReluActivationFunction(0), 0.0001);
            Assert::AreEqual(0.00723, NodeNetwork::leakyReluActivationFunction(0.00723), 0.0001);
            Assert::AreEqual(999999.199, NodeNetwork::leakyReluActivationFunction(999999.199), 0.0001);

            // negative values are squashed to a much smaller value
            const double relu1 = NodeNetwork::leakyReluActivationFunction(-194.5);
            const double relu2 = NodeNetwork::leakyReluActivationFunction(-8986.225);
            const double relu3 = NodeNetwork::leakyReluActivationFunction(-0.0000623);
            Assert::AreEqual(true, -194.5 < relu1);
            Assert::AreEqual(true, -8986.225 < relu2);
            Assert::AreEqual(true, -0.0000623 < relu3);

            // ... but the order is still the same
            Assert::AreEqual(true, relu2 < relu1);
            Assert::AreEqual(true, relu1 < relu3);
            Assert::AreEqual(true, relu3 < 0);
        }

        TEST_METHOD(NodeNetwork_sigmoidActivationFunction)
        {
            // values are squashed to a range between [0, 1]
            const double sigm1 = NodeNetwork::sigmoidActivationFunction(-255298.2);
            const double sigm2 = NodeNetwork::sigmoidActivationFunction(-86.76);
            const double sigm3 = NodeNetwork::sigmoidActivationFunction(-0.0826);
            const double sigm4 = NodeNetwork::sigmoidActivationFunction(0);
            const double sigm5 = NodeNetwork::sigmoidActivationFunction(0.000211);
            const double sigm6 = NodeNetwork::sigmoidActivationFunction(86.76);
            const double sigm7 = NodeNetwork::sigmoidActivationFunction(1000000.0);

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

        TEST_METHOD(NodeNetwork_hyperbolicTanActivationFunction)
        {
            // values are squashed to a range between [-1, 1]
            const double tanh1 = NodeNetwork::hyperbolicTanActivationFunction(-4265.003);
            const double tanh2 = NodeNetwork::hyperbolicTanActivationFunction(-1.0);
            const double tanh3 = NodeNetwork::hyperbolicTanActivationFunction(-0.0000001);
            const double tanh4 = NodeNetwork::hyperbolicTanActivationFunction(0);
            const double tanh5 = NodeNetwork::hyperbolicTanActivationFunction(0.000211);
            const double tanh6 = NodeNetwork::hyperbolicTanActivationFunction(27.82);
            const double tanh7 = NodeNetwork::hyperbolicTanActivationFunction(53962.99);

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
        TEST_METHOD(NodeNetwork_derivedIdentityActivationFunction)
        {
            // all values map to 1
            Assert::AreEqual(1, NodeNetwork::identityActivationFunction(-9962.6536, true), 0.0001);
            Assert::AreEqual(1, NodeNetwork::identityActivationFunction(-0.0714, true), 0.0001);
            Assert::AreEqual(1, NodeNetwork::identityActivationFunction(0, true), 0.0001);
            Assert::AreEqual(1, NodeNetwork::identityActivationFunction(0.00005, true), 0.0001);
            Assert::AreEqual(1, NodeNetwork::identityActivationFunction(7632.24, true), 0.0001);
        }

        TEST_METHOD(NodeNetwork_derivedReluActivationFunction)
        {
            // negative values map to 0
            Assert::AreEqual(0, NodeNetwork::reluActivationFunction(-1468.82, true), 0.0001);
            Assert::AreEqual(0, NodeNetwork::reluActivationFunction(-0.000014, true), 0.0001);

            // non-negative values map to 1
            Assert::AreEqual(1, NodeNetwork::reluActivationFunction(0, true), 0.0001);
            Assert::AreEqual(1, NodeNetwork::reluActivationFunction(0.00252, true), 0.0001);
            Assert::AreEqual(1, NodeNetwork::reluActivationFunction(16002.5, true), 0.0001);
        }

        TEST_METHOD(NodeNetwork_derivedLeakyReluActivationFunction)
        {
            // non-negative values map to 1
            Assert::AreEqual(1, NodeNetwork::leakyReluActivationFunction(0, true), 0.0001);
            Assert::AreEqual(1, NodeNetwork::leakyReluActivationFunction(0.02006, true), 0.0001);
            Assert::AreEqual(1, NodeNetwork::leakyReluActivationFunction(34615.12, true), 0.0001);

            // negative values map to the leaky modifier (0 < m << 1)
            const double derivedRelu1 = NodeNetwork::leakyReluActivationFunction(-28968.3, true);
            const double derivedRelu2 = NodeNetwork::leakyReluActivationFunction(-0.007, true);
            Assert::AreEqual(true, derivedRelu1 == derivedRelu2);
            Assert::AreEqual(true, 0 < derivedRelu1 && derivedRelu1 < 1);
        }

        TEST_METHOD(NodeNetwork_derivedSigmoidActivationFunction)
        {
            // the derivative of the sigmoid function looks Gaussian, centered around 0
            // values are at least 0, and max. some value < 1
            const double derivedSigmoidLeft = NodeNetwork::sigmoidActivationFunction(-255298.2, true);
            const double derivedSigmoidCenterLeft = NodeNetwork::sigmoidActivationFunction(-0.0826, true);
            const double derivedSigmoidCenter = NodeNetwork::sigmoidActivationFunction(0, true);
            const double derivedSigmoidCenterRight = NodeNetwork::sigmoidActivationFunction(0.000211, true);
            const double derivedSigmoidRight = NodeNetwork::sigmoidActivationFunction(1000000.0, true);

            Assert::AreEqual(0, derivedSigmoidLeft, 0.0001);
            Assert::AreEqual(0, derivedSigmoidRight, 0.0001);
            Assert::AreEqual(true, derivedSigmoidCenter > derivedSigmoidCenterLeft);
            Assert::AreEqual(true, derivedSigmoidCenter > derivedSigmoidCenterRight);
            Assert::AreEqual(true, derivedSigmoidCenterLeft >= derivedSigmoidLeft);
            Assert::AreEqual(true, derivedSigmoidCenterRight >= derivedSigmoidRight);
        }

        TEST_METHOD(NodeNetwork_derivedHyperbolicTanActivationFunction)
        {
            // the derivative of tanh is also centered around 0 but the slope is much steeper
            // values are in [0, 1]
            const double derivedTanhLeft = NodeNetwork::hyperbolicTanActivationFunction(-863200.54, true);
            const double derivedTanhCenterLeft = NodeNetwork::hyperbolicTanActivationFunction(-0.00035, true);
            const double derivedTanhCenter = NodeNetwork::hyperbolicTanActivationFunction(0, true);
            const double derivedTanhCenterRight = NodeNetwork::hyperbolicTanActivationFunction(0.009999, true);
            const double derivedTanhRight = NodeNetwork::hyperbolicTanActivationFunction(11111.8, true);

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