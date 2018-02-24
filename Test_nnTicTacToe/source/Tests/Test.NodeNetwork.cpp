#include "stdafx.h"
#include "CppUnitTest.h"
#include "NeuralNetwork/NodeNetwork.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NodeNetworkTest
{
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
            NodeNetwork nnet;
            Assert::AreEqual(false, nnet.createNetwork(0, 3, std::vector<int>()));
        }

        TEST_METHOD(NodeNetwork_createNetwork_tooFewOutputNodes)
        {
            NodeNetwork nnet;
            Assert::AreEqual(false, nnet.createNetwork(7, 0, std::vector<int>()));
        }

        TEST_METHOD(NodeNetwork_createNetwork_invalidHiddenLayer)
        {
            NodeNetwork nnet;
            Assert::AreEqual(false, nnet.createNetwork(3, 2, std::vector<int>({ 2, -1, 1 })));
        }

        TEST_METHOD(NodeNetwork_createNetwork_valid)
        {
            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(2, 1, std::vector<int>({2, 2})));

            // 5 nodes with bias value + 10 edges with weight value
            Assert::AreEqual(15, nnet.getNumParameters());

            std::vector<double> outputValues;
            nnet.getOutputValues(outputValues);

            Assert::AreEqual(1, static_cast<int>(outputValues.size()));
            Assert::AreEqual(0.0, outputValues[0]);
        }

        TEST_METHOD(NodeNetwork_assignInputValues_tooFew)
        {
            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(3, 1, std::vector<int>({ 2, 3 })));

            // too few values
            Assert::AreEqual(false, nnet.assignInputValues({-1.0, 2.1}));
        }

        TEST_METHOD(NodeNetwork_assignInputValues_tooMany)
        {
            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(2, 2, std::vector<int>({ 4 })));

            // too many values
            Assert::AreEqual(false, nnet.assignInputValues({ 0.74, 0.2, -0.3, 1.7 }));
        }

        TEST_METHOD(NodeNetwork_assignInputValues)
        {
            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(3, 2, std::vector<int>({})));

            Assert::AreEqual(true, nnet.assignInputValues({ 5.2, -1.34, 0.25 }));
        }

        TEST_METHOD(NodeNetwork_assignParameters_tooFewParameters)
        {
            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(2, 1, std::vector<int>({ 2 })));

            // 3 nodes with bias value + 6 edges with weight value
            Assert::AreEqual(9, nnet.getNumParameters());

            // too few parameters
            std::queue<double> params = std::queue<double>({1.2, 3.8, 5.32 });
            Assert::AreEqual(false, nnet.assignParameters(params));
        }

        TEST_METHOD(NodeNetwork_assignParameters_tooManyParameters)
        {
            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(2, 2, std::vector<int>()));

            // 2 nodes with bias value + 4 edges with weights
            Assert::AreEqual(6, nnet.getNumParameters());

            // too few parameters
            std::queue<double> params = std::queue<double>({ -7.3, 5.33, 8.1, 0.52, 0.0, 9.0, -4.2});
            Assert::AreEqual(false, nnet.assignParameters(params));
        }

        TEST_METHOD(NodeNetwork_assignParameters)
        {
            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(2, 2, std::vector<int>()));

            // 2 nodes with bias value + 4 edges with weights
            Assert::AreEqual(6, nnet.getNumParameters());

            // should succeed
            std::queue<double> params = std::queue<double>({ 0.52, -12.6, -0.001, -5.38, 9.2, 0.0 });
            Assert::AreEqual(true, nnet.assignParameters(params));
        }

        TEST_METHOD(NodeNetwork_compute_noParams_noInputValues)
        {
            NodeNetwork nnet;
            Assert::AreEqual(true, nnet.createNetwork(2, 2, std::vector<int>({ 2 })));

            std::vector<double> outputValues;
            nnet.computeValues();
            nnet.getOutputValues(outputValues);

            Assert::AreEqual(2, static_cast<int>(outputValues.size()));
            Assert::AreEqual(0.0, outputValues[0]);
            Assert::AreEqual(0.0, outputValues[1]);
        }

        TEST_METHOD(NodeNetwork_compute)
        {
            std::shared_ptr<NodeNetwork> network = std::make_shared<NodeNetwork>();
            network->createNetwork(3, 1, std::vector<int>());
            network->assignInputValues(std::vector<double>({ 1, -0.5, -1 }));

            std::queue<double> params = std::queue<double>({ 0.2, 0.7, -0.3, 1.2 });
            network->assignParameters(params);
            network->computeValues();

            std::vector<double> outputValues;
            network->getOutputValues(outputValues);

            Assert::AreEqual(1, static_cast<int>(outputValues.size()));
            Assert::AreEqual(1.35, outputValues[0], 0.00001);
        }
    };
}