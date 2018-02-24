#pragma once

#include <vector>
#include <memory>
#include <queue>

#include "Node.h"

namespace NeuralNetwork
{
    class Node;

    typedef std::vector<std::shared_ptr<Node>> Layer;

    class NodeNetworkInterface
    {
    public:
        virtual bool createNetwork(int numInputNodes, int numOutputNodes, std::vector<int> numHiddenNodes) = 0;
        virtual void destroyNetwork() = 0;

        virtual bool assignInputValues(const std::vector<double>& inputValues) = 0;

        /// input is a copy instead of a reference, so we don't affect the initial queue
        virtual bool assignParameters(std::queue<double> params) = 0;

        virtual bool computeValues() = 0;
        virtual void getOutputValues(std::vector<double>& outputValues) const = 0;
    };

    class NodeNetwork
        : public NodeNetworkInterface
    {
    public:
        NodeNetwork() = default;
        ~NodeNetwork();

    public:
        bool createNetwork(int numInputNodes, int numOutputNodes, std::vector<int> numHiddenNodes) override;
        void destroyNetwork() override;

        bool assignInputValues(const std::vector<double>& inputValues) override;
        bool assignParameters(std::queue<double> params) override;
        bool computeValues() override;
        void getOutputValues(std::vector<double>& outputValues) const override;

    public:

        int getNumParameters() const;

    private:
        Layer addInnerLayer(int numNodes, const Layer& previousLayer);

    private:
        std::vector<Layer> m_layers;
    };
}
