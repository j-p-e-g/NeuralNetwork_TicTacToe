#pragma once

#include <memory>
#include <queue>
#include <vector>

#include "General/Globals.h"
#include "Node.h"

namespace NeuralNetwork
{
    class Node;

    typedef std::vector<std::shared_ptr<Node>> Layer;

    class NodeNetworkInterface
    {
    public:
        virtual bool createNetwork(const NetworkSizeData& sizeData, const std::string& acceptanceFunctionType = "none") = 0;
        virtual void destroyNetwork() = 0;

        virtual bool assignInputValues(const std::vector<double>& inputValues) = 0;

        virtual bool assignParameters(const std::vector<double>& params) = 0;

        /// input is a copy instead of a reference, so we don't affect the initial queue
        virtual bool assignParameters(std::queue<double> params) = 0;

        virtual bool computeValues() = 0;
        virtual int getOutputValues(std::vector<double>& outputValues) const = 0;
        virtual void describeNetwork() const = 0;
    };

    class NodeNetwork
        : public NodeNetworkInterface
    {
    public:
        NodeNetwork() = default;
        ~NodeNetwork();

    public:
        bool createNetwork(const NetworkSizeData& sizeData, const std::string& acceptanceFunctionType = "none") override;
        void destroyNetwork() override;

        bool assignInputValues(const std::vector<double>& inputValues) override;
        bool assignParameters(const std::vector<double>& params) override;
        bool assignParameters(std::queue<double> params) override;
        bool computeValues() override;
        int getOutputValues(std::vector<double>& outputValues) const override;

    public:
        int getNumParameters() const;
        void describeNetwork() const override;

    public:
        static double noAcceptanceFunction(double val);
        static double sigmoidAcceptanceFunction(double val);
        static double reluAcceptanceFunction(double val);

    private:
        Layer addInnerLayer(int numNodes, const Layer& previousLayer);

    private:
        std::string m_acceptanceFunctionType;
        std::function<double(double)> m_acceptanceFunction;
        std::vector<Layer> m_layers;
    };
}
