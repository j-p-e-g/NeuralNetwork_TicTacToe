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
        virtual bool createNetwork(const NetworkSizeData& sizeData, const std::string& activationFunctionType = "none") = 0;
        virtual void destroyNetwork() = 0;

        virtual bool assignInputValues(const std::vector<double>& inputValues) = 0;
        virtual bool assignParameters(const std::vector<double>& params) = 0;
        virtual void getParameters(std::vector<double>& params) const = 0;

        /// input is a copy instead of a reference, so we don't affect the initial queue
        virtual bool assignParameters(std::queue<double> params) = 0;

        virtual bool computeValues() = 0;

        /// If applySoftMax is true, all values are proportionally rescaled such that their range is in [0, 1] and their sum is 1.
        virtual int getOutputValues(std::vector<double>& outputValues, bool applySoftMax = false) const = 0;
        virtual double getTotalError(const std::vector<double>& targetValues) const = 0;
        virtual void handleBackpropagation(const std::vector<double>& targetValues, std::vector<double>& parameterAdjustments) = 0;
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
        void getParameters(std::vector<double>& params) const override;
        bool computeValues() override;
        int getOutputValues(std::vector<double>& outputValues, bool applySoftMax = false) const override;
        double getTotalError(const std::vector<double>& targetValues) const override;
        void handleBackpropagation(const std::vector<double>& targetValues, std::vector<double>& parameterAdjustments) override;
        void handleBackpropagation(Layer& layer, const std::vector<double>& targetValues, std::function<double(double, bool)> activationFunction, std::vector<double>& inputAdjustments);

    public:
        int getNumParameters() const;
        void describeNetwork() const override;

    public:
        static double identityActivationFunction(double val, bool derivative = false);
        static double sigmoidActivationFunction(double val, bool derivative = false);
        static double hyperbolicTanActivationFunction(double val, bool derivative = false);
        static double reluActivationFunction(double val, bool derivative = false);
        static double leakyReluActivationFunction(double val, bool derivative = false);

    private:
        Layer createInnerLayer(int numNodes, const Layer& previousLayer);

    private:
        std::string m_activationFunctionType;
        std::function<double(double, bool)> m_activationFunction;
        Layer m_inputLayer;
        std::vector<Layer> m_hiddenLayers;
        Layer m_outputLayer;
    };
}
