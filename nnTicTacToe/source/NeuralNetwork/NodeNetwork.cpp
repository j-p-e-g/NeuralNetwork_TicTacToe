#include "stdafx.h"

#include "FileIO/FileManager.h"
#include "NodeNetwork.h"

#include <assert.h> 
#include <iostream>

namespace NeuralNetwork
{
    using namespace FileIO;

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

    bool NodeNetwork::createNetwork(const NetworkSizeData& sizeData)
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
        for (auto& layer : m_layers)
        {
            for (auto& node : layer)
            {
                (*node).updateValue();
            }
        }

        return true;
    }

    int NodeNetwork::getOutputValues(std::vector<double>& outputValues) const
    {
        assert(m_layers.size() >= 2);

        outputValues.clear();

        int bestIndex = 0;
        const Layer& outputLayer = m_layers[m_layers.size() - 1];

        for (int k = 0; k < outputLayer.size(); k++)
        {
            const double value = outputLayer[k]->getValue();
            if (value > outputLayer[bestIndex]->getValue())
            {
                bestIndex = k;
            }

            outputValues.push_back(value);
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

        buffer << std::endl << "  #parameters: " << getNumParameters() << std::endl;

        PRINT_LOG(buffer);
    }
}
