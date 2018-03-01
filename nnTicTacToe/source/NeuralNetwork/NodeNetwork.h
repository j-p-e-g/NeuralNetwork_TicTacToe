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

        /// If applySoftMax is true, all values are proportionally rescaled such that their range is in [0, 1] and their sum is 1.
        virtual int getOutputValues(std::vector<double>& outputValues, bool applySoftMax = false) const = 0;
        virtual void describeNetwork() const = 0;
    };

    class NodeNetwork
        : public NodeNetworkInterface
    {
    public:
        NodeNetwork() = default;
        ~NodeNetwork();

    public:
        bool createNetwork(const NetworkSizeData& sizeData, const std::string& activationFunctionType = "none") override;
        void assignActivationFunction(const std::string &activationFunctionType);

        void destroyNetwork() override;

        bool assignInputValues(const std::vector<double>& inputValues) override;
        bool assignParameters(const std::vector<double>& params) override;
        bool assignParameters(std::queue<double> params) override;
        bool computeValues() override;
        int getOutputValues(std::vector<double>& outputValues, bool applySoftMax = false) const override;

    public:
        int getNumParameters() const;
        void describeNetwork() const override;

    public:
        static double identityActivationFunction(double val);
        static double sigmoidActivationFunction(double val);
        static double hyperbolicTanActivationFunction(double val);
        static double reluActivationFunction(double val);
        static double leakyReluActivationFunction(double val);

    private:
        Layer createInnerLayer(int numNodes, const Layer& previousLayer);

    private:
        std::string m_activationFunctionType;
        std::function<double(double)> m_activationFunction;
        Layer m_inputLayer;
        std::vector<Layer> m_hiddenLayers;
        Layer m_outputLayer;
    };
}
