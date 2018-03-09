#include "stdafx.h"

#include "FileIO/FileManager.h"
#include "Math/ActivationFunctions.h"
#include "NodeNetwork.h"

#include <assert.h> 
#include <iostream>

namespace NeuralNetwork
{
    using namespace FileIO;
    using namespace Math;

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
                m_activationFunction = ActivationFunctions::leakyRelu;
                m_activationFunctionType = "leaky ReLU";

            }
            else
            {
                m_activationFunction = ActivationFunctions::relu;
                m_activationFunctionType = "ReLU";
            }
        }
        else if (activationFunctionType.find("tan") != std::string::npos)
        {
            m_activationFunction = ActivationFunctions::hyperbolicTan;
            m_activationFunctionType = "hyperbolic tan (tanh)";
        }
        else if (activationFunctionType.find("sigm") != std::string::npos)
        {
            m_activationFunction = ActivationFunctions::sigmoid;
            m_activationFunctionType = "sigmoid";
        }
        else
        {
            m_activationFunction = ActivationFunctions::identity;
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
        const int expectedParameters = getNumParameters();
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

    void NodeNetwork::getParameters(std::vector<double>& params) const
    {
        params.clear();

        for (auto& layer : m_hiddenLayers)
        {
            for (auto& node : layer)
            {
                (*node).getParameters(params);
            }
        }

        for (auto& node : m_outputLayer)
        {
            (*node).getParameters(params);
        }
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
            (*node).updateValue(ActivationFunctions::identity);
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

    double NodeNetwork::getTotalError(const std::vector<double>& targetValues) const
    {
        assert(m_outputLayer.size() == targetValues.size());

        double error = 0;
        for (unsigned int k = 0; k < m_outputLayer.size(); k++)
        {
            error += m_outputLayer[k]->getError(targetValues[k]);
        }

        return error;
    }

    void NodeNetwork::handleBackpropagation(const std::vector<double>& targetValues, std::vector<double>& parameterAdjustments)
    {
        // during the backpropagation, all parameters get overwritten with their adjustment values, so
        // we need to backup the real parameters
        std::vector<double> initialParameters;
        getParameters(initialParameters);

        // do backpropagation for output layer
        std::vector<double> inputAdjustments;
        handleBackpropagation(m_outputLayer, targetValues, ActivationFunctions::identity, inputAdjustments);

        // compute average input adjustment
        for (auto& val : inputAdjustments)
        {
            val /= m_outputLayer.size();
        }

        // do backpropagation for hidden layers
        for (int k = static_cast<int>(m_hiddenLayers.size()) - 1; k >= 0; k--)
        {
            std::vector<double> innerNodeTargetValues;
            for (unsigned int n = 0; n < m_hiddenLayers[k].size(); n++)
            {
                const auto& node = m_hiddenLayers[k][n];
                innerNodeTargetValues.push_back(node->getValue() + inputAdjustments[n]);
            }

            handleBackpropagation(m_hiddenLayers[k], innerNodeTargetValues, m_activationFunction, inputAdjustments);

            // compute average input adjustment
            for (auto& val : inputAdjustments)
            {
                val /= m_hiddenLayers[k].size();
            }
        }

        getParameters(parameterAdjustments);

        // write back the real parameters
        assignParameters(initialParameters);
    }

    void NodeNetwork::handleBackpropagation(Layer& layer, const std::vector<double>& targetValues, std::function<double(double, bool)> activationFunction, std::vector<double>& inputAdjustments)
    {
        assert(targetValues.size() == layer.size());

        inputAdjustments.clear();
        for (unsigned int k = 0; k < layer.size(); k++)
        {
            layer[k]->handleBackpropagation(targetValues[k], activationFunction, inputAdjustments);
        }
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

}
