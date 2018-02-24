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
    };
}