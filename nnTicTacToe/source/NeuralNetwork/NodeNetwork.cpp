#include "stdafx.h"

#include "FileIO/FileManager.h"
#include "NodeNetwork.h"

#include <assert.h> 
#include <iostream>

namespace NeuralNetwork
{
    using namespace FileIO;

    const double LEAKY_RELU_MULTIPLIER = 0.01;

    // --------------------------
    // NodeNetwork
    // --------------------------
    NodeNetwork::~NodeNetwork()
    {
        destroyNetwork();
    }

    void NodeNetwork::destroyNetwork()
    {
        m_inputLayer.clear();
        m_outputLayer.clear();

        for (auto& l : m_hiddenLayers)
        {
            l.clear();
        }

        m_hiddenLayers.clear();
    }

    bool NodeNetwork::createNetwork(const NetworkSizeData& sizeData, const std::string& activationFunctionType)
    {
        if (sizeData.numInputNodes <= 0 || sizeData.numOutputNodes <= 0)
        {
            std::ostringstream buffer;
            buffer << "Network needs at least one input and one output node (passed " << sizeData.numInputNodes
                    << " input node(s) and " << sizeData.numOutputNodes << " output node(s))";
            PRINT_ERROR(buffer);
            return false;
        }

        destroyNetwork();

        assignActivationFunction(activationFunctionType);

        for (int k = 0; k < sizeData.numInputNodes; k++)
        {
            std::shared_ptr<Node> node = std::make_shared<Node>();
            m_inputLayer.push_back(node);
        }

        Layer prevLayer = m_inputLayer;
        for (auto nh : sizeData.numHiddenNodes)
        {
            if (nh <= 0)
            {
                PRINT_ERROR("Each hidden layer needs at least one node");
                return false;
            }

            prevLayer = createInnerLayer(nh, prevLayer);
            m_hiddenLayers.push_back(prevLayer);
        }

        m_outputLayer = createInnerLayer(sizeData.numOutputNodes, prevLayer);

        describeNetwork();
        return true;
    }

    void NodeNetwork::assignActivationFunction(const std::string &activationFunctionType)
    {
        if (activationFunctionType.find("relu") != std::string::npos)
        {
            if (activationFunctionType.find("leak") != std::string::npos)
            {
                m_activationFunction = leakyReluActivationFunction;
                m_activationFunctionType = "leaky ReLU";

            }
            else
            {
                m_activationFunction = reluActivationFunction;
                m_activationFunctionType = "ReLU";
            }
        }
        else if (activationFunctionType.find("tan") != std::string::npos)
        {
            m_activationFunction = hyperbolicTanActivationFunction;
            m_activationFunctionType = "hyperbolic tan (tanh)";
        }
        else if (activationFunctionType.find("sigm") != std::string::npos)
        {
            m_activationFunction = sigmoidActivationFunction;
            m_activationFunctionType = "sigmoid";
        }
        else
        {
            m_activationFunction = identityActivationFunction;
            m_activationFunctionType = "identity (none)";
        }
    }

    Layer NodeNetwork::createInnerLayer(int numNodes, const Layer& previousLayer)
    {
        assert(numNodes > 0);

        Layer newLayer; // initially empty

        for (int k = 0; k < numNodes; k++)
        {
            std::vector<std::shared_ptr<Edge>> inputEdges; // initially empty

            for (const auto& prevNode : previousLayer)
            {
                std::shared_ptr<Edge> edge = std::make_shared<Edge>(prevNode);
                inputEdges.push_back(edge);
            }

            std::shared_ptr<Node> node = std::make_shared<InnerNode>(InnerNode(inputEdges));
            newLayer.push_back(node);
        }

        return newLayer;
    }

    int NodeNetwork::getNumParameters() const
    {
        int paramCount = 0;

        for (auto& layer : m_hiddenLayers)
        {
            for (auto& node : layer)
            {
                paramCount += node->getNumParameters();
            }
        }

        for (auto& node : m_outputLayer)
        {
            paramCount += node->getNumParameters();
        }

        return paramCount;
    }

    bool NodeNetwork::assignInputValues(const std::vector<double>& inputValues)
    {
        assert(!m_inputLayer.empty());

        if (m_inputLayer.size() != inputValues.size())
        {
            std::ostringstream buffer;
            buffer << "Mismatch between number of input values (" << inputValues.size() << ") and input nodes (" << m_inputLayer.size() << ")!";
            PRINT_ERROR(buffer);
            return false;
        }

        for (int k = 0; k < m_inputLayer.size(); k++)
        {
            m_inputLayer[k]->setValue(inputValues[k]);
        }

        return true;
    }

    bool NodeNetwork::assignParameters(const std::vector<double>& params)
    {
        std::queue<double> qParams;
        for (const auto& val : params)
        {
            qParams.push(val);
        }

        return assignParameters(qParams);
    }

    bool NodeNetwork::assignParameters(std::queue<double> params)
    {
        const size_t expectedParameters = getNumParameters();
        if (expectedParameters != params.size())
        {
            std::ostringstream buffer;
            buffer << "Unexpected number of parameters passed into network: " << params.size() << " (expected " << expectedParameters << ")";
            PRINT_ERROR(buffer);
            return false;
        }

        for (auto& layer : m_hiddenLayers)
        {
            for (auto& node : layer)
            {
                if (!(*node).assignParameters(params))
                {
                    return false;
                }
            }
        }

        for (auto& node : m_outputLayer)
        {
            if (!(*node).assignParameters(params))
            {
                return false;
            }
        }

        return true;
    }

    bool NodeNetwork::computeValues()
    {
        // starting at the input layer and moving towards the output layer, update each node
        for (auto& layer : m_hiddenLayers)
        {
            for (auto& node : layer)
            {
                (*node).updateValue(m_activationFunction);
            }
        }

        // don't apply activation function to output layer
        for (auto& node : m_outputLayer)
        {
            (*node).updateValue(identityActivationFunction);
        }

        return true;
    }

    int NodeNetwork::getOutputValues(std::vector<double>& outputValues, bool applySoftMax) const
    {
        assert(!m_inputLayer.empty());
        assert(!m_outputLayer.empty());

        outputValues.clear();

        int bestIndex = 0;
        double softMaxSum = 0;
        for (int k = 0; k < m_outputLayer.size(); k++)
        {
            double value = m_outputLayer[k]->getValue();
            if (value > m_outputLayer[bestIndex]->getValue())
            {
                bestIndex = k;
            }

            if (applySoftMax)
            {
                value = std::exp(value);
                softMaxSum += value;
            }

            outputValues.push_back(value);
        }

        if (applySoftMax)
        {
            for (auto& val : outputValues)
            {
                val /= softMaxSum;
            }
        }

        return bestIndex;
    }

    void NodeNetwork::describeNetwork() const
    {
        assert(!m_inputLayer.empty());
        assert(!m_outputLayer.empty());

        std::ostringstream buffer;
        buffer << "NodeNetwork: ";
        buffer << std::endl << "  #layers: " << (2 + m_hiddenLayers.size());
        buffer << std::endl << "  #input nodes: " << m_inputLayer.size();
        buffer << std::endl << "  #output nodes: " << m_outputLayer.size();

        buffer << std::endl << "  #hidden layer nodes: ";
        bool first = true;
        for (const auto& layer : m_hiddenLayers)
        {
            if (first)
            {
                first = false;
            }
            else
            {
                buffer << ", ";
            }

            buffer << layer.size();
        }

        buffer << std::endl << "  activation function type: " << m_activationFunctionType;
        buffer << std::endl;

        PRINT_LOG(buffer);
    }

    double NodeNetwork::identityActivationFunction(double val, bool derivative)
    {
        if (derivative)
        {
            return 1;
        }
        return val;
    }

    double NodeNetwork::sigmoidActivationFunction(double val, bool derivative)
    {
        // A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve. 
        // Often, sigmoid function refers to the special case of the logistic function  defined by the formula
        // S(x) = 1 / (1 + e^(-1x)) = e^x / (1 + e^x)
        // Note that the sigmoid output is centered around 0.5 and its range is in [0, 1].

        if (derivative)
        {
            const double result = sigmoidActivationFunction(val, false);
            return result * (1 - result);
        }

        const double e = std::exp(val);
        if (e == INFINITY)
        {
            return 1;
        }

        return e / (e + 1);
    }

    double NodeNetwork::hyperbolicTanActivationFunction(double val, bool derivative)
    {
        // Hyperbolic Tangent Activation Function: 
        // The tanh(z) function is a rescaled version of the sigmoid, and its output range is[-1, 1] instead of[0, 1].

        if (derivative)
        {
            const double result = hyperbolicTanActivationFunction(val, false);
            return 1 - result * result;
        }

        return 2 * sigmoidActivationFunction(2 * val) - 1;
    }

    double NodeNetwork::reluActivationFunction(double val, bool derivative)
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

    double NodeNetwork::leakyReluActivationFunction(double val, bool derivative)
    {
        // Leaky ReLUs are one attempt to fix the "dying ReLU" problem. 
        // Instead of the function being zero when x < 0, a leaky ReLU will instead have a small negative slope (of 0.01, or so).

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
