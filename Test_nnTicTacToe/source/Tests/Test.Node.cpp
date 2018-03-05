#include "stdafx.h"
#include "CppUnitTest.h"
#include "NeuralNetwork/Node.h"
#include "NeuralNetwork/NodeNetwork.h"

namespace NodeTest
{
    using namespace Microsoft::VisualStudio::CppUnitTestFramework;
    using namespace NeuralNetwork;

    TEST_CLASS(NodeEdge_Test)
    {
    public:
        // -----------------------------------------
        // Node
        // -----------------------------------------
        TEST_METHOD(Node_default)
        {
            Node node;
            Assert::AreEqual(0.0, node.getValue());
            Assert::AreEqual(0, node.getNumParameters());

            // check parameters
            std::vector<double> currentParams;
            node.getParameters(currentParams);
            Assert::AreEqual(true, currentParams.empty());
        }

        TEST_METHOD(Node_setValue_getValue)
        {
            Node node;
            node.setValue(5.7);
            Assert::AreEqual(5.7, node.getValue());
        }

        // -----------------------------------------
        // Edge
        // -----------------------------------------
        TEST_METHOD(Edge_default)
        {
            std::shared_ptr<Node> node = std::make_shared<Node>();
            Edge edge = Edge(node);

            Assert::AreEqual(1, edge.getNumParameters());
            Assert::AreEqual(0.0, edge.getValue());

            // check parameters
            std::vector<double> currentParams;
            edge.getParameters(currentParams);
            Assert::AreEqual(1, static_cast<int>(currentParams.size()));
            Assert::AreEqual(1, currentParams[0], 0.0001);
        }

        TEST_METHOD(Edge_default_withNodeValue)
        {
            std::shared_ptr<Node> node = std::make_shared<Node>();
            node->setValue(3.528);

            Edge edge = Edge(node);
            Assert::AreEqual(3.528, edge.getValue());
        }

        TEST_METHOD(Edge_setEdgeWeight_withoutParams)
        {
            std::shared_ptr<Node> node = std::make_shared<Node>();
            node->setValue(1.2);

            Edge edge = Edge(node);

            edge.setEdgeWeight(-3.9081);
            Assert::AreEqual(-3.9081, edge.getEdgeWeight());

            // product of both values
            Assert::AreEqual(-4.68972, edge.getValue());

            // check parameters
            std::vector<double> currentParams;
            edge.getParameters(currentParams);
            Assert::AreEqual(1, static_cast<int>(currentParams.size()));
            Assert::AreEqual(-3.9081, currentParams[0], 0.0001);
        }

        TEST_METHOD(Edge_assignParameters_tooFewParams)
        {
            std::shared_ptr<Node> node = std::make_shared<Node>();
            Edge edge = Edge(node);

            // method should fail if there are too few parameters
            std::queue<double> params;
            Assert::AreEqual(false, edge.assignParameters(params));
        }

        TEST_METHOD(Edge_assignParameters)
        {
            std::shared_ptr<Node> node = std::make_shared<Node>();
            Edge edge = Edge(node);

            // should succeed
            std::queue<double> params({7.15, 9.001});
            Assert::AreEqual(true, edge.assignParameters(params));

            // first value should be removed
            Assert::AreEqual(1, static_cast<int>(params.size()));
            Assert::AreEqual(9.001, params.front());

            // value equal to the assigned parameter
            Assert::AreEqual(7.15, edge.getEdgeWeight());

            // check parameters
            std::vector<double> currentParams;
            edge.getParameters(currentParams);
            Assert::AreEqual(1, static_cast<int>(currentParams.size()));
            Assert::AreEqual(7.15, currentParams[0]);
        }

        TEST_METHOD(Edge_getValue_withParams)
        {
            std::shared_ptr<Node> node = std::make_shared<Node>();
            node->setValue(1.111);

            Edge edge = Edge(node);

            std::queue<double> params({ 2.1 });
            Assert::AreEqual(true, edge.assignParameters(params));

            // should be the product of both values
            Assert::AreEqual(2.3331, edge.getValue(), 0.00001);
        }

        // -----------------------------------------
        // InnerNode, without input edges
        // -----------------------------------------
        TEST_METHOD(InnerNode_noEdges_default)
        {
            std::vector<std::shared_ptr<Edge>> inputEdges; // empty
            InnerNode node = InnerNode(inputEdges);

            Assert::AreEqual(0.0, node.getValue());
            Assert::AreEqual(1, node.getNumParameters());
        }

        TEST_METHOD(InnerNode_noEdges_assignParameters_tooFewParams)
        {
            std::vector<std::shared_ptr<Edge>> inputEdges; // empty
            std::shared_ptr<Node> node = std::make_shared<InnerNode>(InnerNode(inputEdges));

            // method should fail if there are too few parameters
            std::queue<double> params;
            Assert::AreEqual(false, node->assignParameters(params));
        }

        TEST_METHOD(InnerNode_noEdges_assignParameters_update)
        {
            std::vector<std::shared_ptr<Edge>> inputEdges; // empty
            std::shared_ptr<Node> node = std::make_shared<InnerNode>(InnerNode(inputEdges));
            node->setValue(-7.259);

            // should succeed
            std::queue<double> params = std::queue<double>({ 25.963 });
            Assert::AreEqual(true, node->assignParameters(params));

            // value should be removed
            Assert::AreEqual(true, params.empty());

            // should be equal to the new value (overwrites old value)
            node->updateValue(NodeNetwork::identityActivationFunction);
            Assert::AreEqual(25.963, node->getValue());

            // check parameters
            std::vector<double> currentParams;
            node->getParameters(currentParams);
            Assert::AreEqual(1, static_cast<int>(currentParams.size()));
            Assert::AreEqual(25.963, currentParams[0]);
        }

        // -----------------------------------------
        // InnerNode, with input edges
        // -----------------------------------------
        TEST_METHOD(InnerNode_withNodeEdges_default)
        {
            // base node predecessor
            std::shared_ptr<Node> nodeA = std::make_shared<Node>();
            std::shared_ptr<Edge> edgeA = std::make_shared<Edge>(nodeA);

            // inner node predecessor
            std::vector<std::shared_ptr<Edge>> emptyEdgeVector;
            std::shared_ptr<Node> nodeB = std::make_shared<InnerNode>(emptyEdgeVector);
            std::shared_ptr<Edge> edgeB = std::make_shared<Edge>(nodeB);

            // setup edges
            std::vector<std::shared_ptr<Edge>> inputEdges;
            inputEdges.push_back(edgeA);
            inputEdges.push_back(edgeB);

            std::shared_ptr<Node> node = std::make_shared<InnerNode>(InnerNode(inputEdges));

            // 2 edge weights + own bias
            Assert::AreEqual(3, node->getNumParameters());

            // check parameters
            // the two edges have a default weight of 1, the bias is by default 0
            std::vector<double> currentParams;
            node->getParameters(currentParams);
            Assert::AreEqual(3, static_cast<int>(currentParams.size()));
            Assert::AreEqual(1, currentParams[0], 0.0001);
            Assert::AreEqual(1, currentParams[1], 0.0001);
            Assert::AreEqual(0, currentParams[2], 0.0001);
        }

        TEST_METHOD(InnerNode_withNodeEdges_getValue)
        {
            // inner node predecessors
            std::vector<std::shared_ptr<Edge>> emptyEdgeVector;
            std::shared_ptr<Node> nodeA = std::make_shared<Node>();
            nodeA->setValue(-5.18);
            std::shared_ptr<Edge> edgeA = std::make_shared<Edge>(nodeA);

            std::shared_ptr<Node> nodeB = std::make_shared<Node>();
            nodeB->setValue(3.8);
            std::shared_ptr<Edge> edgeB = std::make_shared<Edge>(nodeB);

            // setup edges
            std::vector<std::shared_ptr<Edge>> inputEdges;
            inputEdges.push_back(edgeA);
            inputEdges.push_back(edgeB);

            std::shared_ptr<Node> node = std::make_shared<InnerNode>(InnerNode(inputEdges));
            node->updateValue(NodeNetwork::identityActivationFunction);

            // should be the sum of both values
            Assert::AreEqual(-1.38, node->getValue(), 0.00001);
        }

        TEST_METHOD(InnerNode_withInnerNodeEdges_assignParameters_tooFewParams)
        {
            // inner node predecessors
            std::vector<std::shared_ptr<Edge>> emptyEdgeVector;
            std::shared_ptr<InnerNode> nodeA = std::make_shared<InnerNode>(emptyEdgeVector);
            nodeA->setValue(99.25);
            std::shared_ptr<Edge> edgeA = std::make_shared<Edge>(nodeA);

            std::shared_ptr<InnerNode> nodeB = std::make_shared<InnerNode>(emptyEdgeVector);
            nodeB->setValue(4.2);
            std::shared_ptr<Edge> edgeB = std::make_shared<Edge>(nodeB);

            // setup edges
            std::vector<std::shared_ptr<Edge>> inputEdges;
            inputEdges.push_back(edgeA);
            inputEdges.push_back(edgeB);

            std::shared_ptr<Node> node = std::make_shared<InnerNode>(InnerNode(inputEdges));

            // method should fail if there are too few parameters
            std::queue<double> params = std::queue<double>({-2.001, 17.826});
            Assert::AreEqual(false, node->assignParameters(params));
        }

        TEST_METHOD(InnerNode_withInnerNodeEdges_assignParameters)
        {
            // inner node predecessors
            std::vector<std::shared_ptr<Edge>> emptyEdgeVector;
            std::shared_ptr<InnerNode> nodeA = std::make_shared<InnerNode>(emptyEdgeVector);
            nodeA->setValue(-10);
            std::shared_ptr<Edge> edgeA = std::make_shared<Edge>(nodeA);

            std::shared_ptr<InnerNode> nodeB = std::make_shared<InnerNode>(emptyEdgeVector);
            nodeB->setValue(-0.5);
            std::shared_ptr<Edge> edgeB = std::make_shared<Edge>(nodeB);

            // setup edges
            std::vector<std::shared_ptr<Edge>> inputEdges;
            inputEdges.push_back(edgeA);
            inputEdges.push_back(edgeB);

            std::shared_ptr<Node> node = std::make_shared<InnerNode>(InnerNode(inputEdges));

            // should succeed
            std::queue<double> params = std::queue<double>({ 1.2, -4, 21 });
            Assert::AreEqual(true, node->assignParameters(params));
            node->updateValue(NodeNetwork::identityActivationFunction);

            // nodeA * param1 + nodeB * param2 + param3
            // -12 + 2 + 21.6 = 11
            Assert::AreEqual(11.0, node->getValue());

            // check parameters
            std::vector<double> currentParams;
            node->getParameters(currentParams);
            Assert::AreEqual(3, static_cast<int>(currentParams.size()));
            Assert::AreEqual(1.2, currentParams[0], 0.0001);
            Assert::AreEqual(-4, currentParams[1], 0.0001);
            Assert::AreEqual(21, currentParams[2], 0.0001);
        }

        //---------------------------------------
        // Back propagation
        //---------------------------------------
        TEST_METHOD(Node_getError_identical)
        {
            Node node;
            node.setValue(3.72);

            const double error = node.getError(3.72);
            Assert::AreEqual(0, error, 0.0001);
        }

        TEST_METHOD(Node_getError_different_negative)
        {
            Node node;
            node.setValue(0.734);

            const double error = node.getError(-0.734);
            Assert::AreNotEqual(0, error, 0.0001);
        }

        TEST_METHOD(Node_getError_different_positive)
        {
            Node node;
            node.setValue(1.54);

            const double error = node.getError(1.55);
            Assert::AreNotEqual(0, error, 0.0001);
        }

        TEST_METHOD(Node_getDerivedError)
        {
            const double val1 = -2.49;
            const double val2 = -0.9;

            Node nodeA;
            nodeA.setValue(val1);
            const double errorA = nodeA.getError(val2);
            const double derivedErrorA = nodeA.getError(val2, true);

            // swap current and target value
            Node nodeB;
            nodeB.setValue(val2);
            const double errorB = nodeB.getError(val1);
            const double derivedErrorB = nodeB.getError(val1, true);

            // the non-derived error values are both the same positive value
            Assert::AreEqual(true, errorA > 0, 0.0001);
            Assert::AreEqual(errorA, errorB, 0.0001);

            // the derived error values have the opposite sign
            Assert::AreEqual(derivedErrorA, -derivedErrorB, 0.0001);
        }

        TEST_METHOD(InnerNode_handleBackpropagation_sameAsTargetValue)
        {
            // predecessor nodes
            std::shared_ptr<Node> nodeA = std::make_shared<Node>();
            std::shared_ptr<Node> nodeB = std::make_shared<Node>();
            std::shared_ptr<Edge> edgeA = std::make_shared<Edge>(nodeA);
            std::shared_ptr<Edge> edgeB = std::make_shared<Edge>(nodeB);

            // setup edges
            std::vector<std::shared_ptr<Edge>> inputEdges;
            inputEdges.push_back(edgeA);
            inputEdges.push_back(edgeB);

            std::shared_ptr<Node> node = std::make_shared<InnerNode>(InnerNode(inputEdges));

            nodeA->setValue(2.6);
            nodeB->setValue(4.05);

            std::queue<double> params;
            params.push(-0.1);
            params.push(0.2);
            params.push(0.7);
            node->assignParameters(params);

            // if the targetValue is equal to the output value, 
            // all adjustment values should be zero
            node->updateValue(NodeNetwork::identityActivationFunction);
            const double result = node->getValue();

            std::vector<double> inputValueAdjustments;
            std::vector<double> paramAdjustments;

            node->handleBackpropagation(result, NodeNetwork::identityActivationFunction, inputValueAdjustments);
            node->getParameters(paramAdjustments);

            Assert::AreEqual(2, static_cast<int>(inputValueAdjustments.size()));
            Assert::AreEqual(0, inputValueAdjustments[0], 0.0001);
            Assert::AreEqual(0, inputValueAdjustments[1], 0.0001);

            Assert::AreEqual(3, static_cast<int>(paramAdjustments.size()));
            for (auto& val : paramAdjustments)
            {
                Assert::AreEqual(0, val, 0.0001);
            }
        }

        TEST_METHOD(InnerNode_handleBackpropagation_differentFromTargetValue)
        {
            // predecessor nodes
            std::shared_ptr<Node> nodeA = std::make_shared<Node>();
            std::shared_ptr<Node> nodeB = std::make_shared<Node>();
            std::shared_ptr<Edge> edgeA = std::make_shared<Edge>(nodeA);
            std::shared_ptr<Edge> edgeB = std::make_shared<Edge>(nodeB);

            // setup edges
            std::vector<std::shared_ptr<Edge>> inputEdges;
            inputEdges.push_back(edgeA);
            inputEdges.push_back(edgeB);

            std::shared_ptr<Node> node = std::make_shared<InnerNode>(InnerNode(inputEdges));

            nodeA->setValue(0.7999);
            nodeB->setValue(-3.5);

            std::queue<double> params;
            params.push(0.33);
            params.push(0.8);
            params.push(0.0001);
            node->assignParameters(params);

            // if the targetValue is different from the output value, 
            // all adjustment values are highly likely to be non-zero
            node->updateValue(NodeNetwork::leakyReluActivationFunction);
            const double result = node->getValue();

            std::vector<double> inputValueAdjustments;
            std::vector<double> paramAdjustments;

            node->handleBackpropagation(result + 0.08, NodeNetwork::leakyReluActivationFunction, inputValueAdjustments);
            node->getParameters(paramAdjustments);

            Assert::AreEqual(2, static_cast<int>(inputValueAdjustments.size()));
            Assert::AreNotEqual(0, inputValueAdjustments[0], 0.0001);
            Assert::AreNotEqual(0, inputValueAdjustments[1], 0.0001);

            Assert::AreEqual(3, static_cast<int>(paramAdjustments.size()));
            for (auto& val : paramAdjustments)
            {
                Assert::AreNotEqual(0, val, 0.0001);
            }
        }
    };
}