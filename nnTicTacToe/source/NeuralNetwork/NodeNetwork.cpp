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
        for (auto& l : m_layers)
        {
            l.clear();
        }

        m_layers.clear();
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

        Layer prevLayer; // initially empty
        for (int k = 0; k < sizeData.numInputNodes; k++)
        {
            std::shared_ptr<Node> node = std::make_shared<Node>();
            prevLayer.push_back(node);
        }

        m_layers.push_back(prevLayer);

        for (auto nh : sizeData.numHiddenNodes)
        {
            if (nh <= 0)
            {
                PRINT_ERROR("Each hidden layer needs at least one node");
                return false;
            }

            prevLayer = addInnerLayer(nh, prevLayer);
        }

        addInnerLayer(sizeData.numOutputNodes, prevLayer);
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

    Layer NodeNetwork::addInnerLayer(int numNodes, const Layer& previousLayer)
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

        m_layers.push_back(newLayer);

        return newLayer;
    }

    int NodeNetwork::getNumParameters() const
    {
        int paramCount = 0;

        for (auto& layer : m_layers)
        {
            for (auto& node : layer)
            {
                paramCount += node->getNumParameters();
            }
        }

        return paramCount;
    }

    bool NodeNetwork::assignInputValues(const std::vector<double>& inputValues)
    {
        assert(m_layers.size() >= 2);

        Layer& inputLayer = m_layers[0];
        if (inputLayer.size() != inputValues.size())
        {
            std::ostringstream buffer;
            buffer << "Mismatch between number of input values (" << inputValues.size() << ") and input nodes (" << inputLayer.size() << ")!";
            PRINT_ERROR(buffer);
            return false;
        }

        for (int k = 0; k < inputLayer.size(); k++)
        {
            inputLayer[k]->setValue(inputValues[k]);
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

        for (auto& layer : m_layers)
        {
            for (auto& node : layer)
            {
                if (!(*node).assignParameters(params))
                {
                    return false;
                }
            }
        }

        return true;
    }

    bool NodeNetwork::computeValues()
    {
        // starting at the input layer and moving towards the output layer, update each node
        for (unsigned int k = 1; k < m_layers.size(); k++)
        {
            auto& layer = m_layers[k];
            for (auto& node : layer)
            {
                if (k + 1 == m_layers.size())
                {
                    // don't apply activation function to output layer
                    (*node).updateValue(identityActivationFunction);
                }
                else
                {
                    (*node).updateValue(m_activationFunction);
                }
            }
        }

        return true;
    }

    int NodeNetwork::getOutputValues(std::vector<double>& outputValues, bool applySoftMax) const
    {
        assert(m_layers.size() >= 2);

        outputValues.clear();

        int bestIndex = 0;
        const Layer& outputLayer = m_layers[m_layers.size() - 1];

        double softMaxSum = 0;
        for (int k = 0; k < outputLayer.size(); k++)
        {
            double value = outputLayer[k]->getValue();
            if (value > outputLayer[bestIndex]->getValue())
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
        assert(m_layers.size() >= 2);

        std::ostringstream buffer;
        buffer << "NodeNetwork: ";
        buffer << std::endl << "  #layers: " << m_layers.size();
        
        buffer << std::endl << "  #input nodes: " << m_layers[0].size();
        buffer << std::endl << "  #output nodes: " << m_layers[m_layers.size()-1].size();

        buffer << std::endl << "  #hidden layer nodes: ";
        for (unsigned int k = 1; k < m_layers.size() - 1; k++)
        {
            if (k > 1)
            {
                buffer << ", ";
            }

            buffer << m_layers[k].size();
        }

        buffer << std::endl << "  activation function type: " << m_activationFunctionType;
        buffer << std::endl;

        PRINT_LOG(buffer);
    }

    double NodeNetwork::identityActivationFunction(double val)
    {
        return val;
    }

    double NodeNetwork::sigmoidActivationFunction(double val)
    {
        // A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve. 
        // Often, sigmoid function refers to the special case of the logistic function  defined by the formula
        // S(x) = 1 / (1 + e^(-1x)) = e^x / (1 + e^x)
        // Note that the sigmoid output is centered around 0.5 and its range is in [0, 1].

        const double e = std::exp(val);
        if (e == INFINITY)
        {
            return 1;
        }

        return e / (e + 1);
    }

    double NodeNetwork::hyperbolicTanActivationFunction(double val)
    {
        // Hyperbolic Tangent Activation Function: 
        // The tanh(z) function is a rescaled version of the sigmoid, and its output range is[-1, 1] instead of[0, 1].

        return 2 * sigmoidActivationFunction(2 * val) - 1;
    }

    double NodeNetwork::reluActivationFunction(double val)
    {
        // ReLU = rectified linear unit
        // In the context of artificial neural networks, the rectifier is an activation function defined as the positive part of its argument:
        // f(x) = x^(+) = max(0, x)

        return std::max<double>(0, val);
    }

    double NodeNetwork::leakyReluActivationFunction(double val)
    {
        // Leaky ReLUs are one attempt to fix the "dying ReLU" problem. 
        // Instead of the function being zero when x < 0, a leaky ReLU will instead have a small negative slope (of 0.01, or so).

        if (val < 0)
        {
            return val * LEAKY_RELU_MULTIPLIER;
        }

        return val;
    }
}
