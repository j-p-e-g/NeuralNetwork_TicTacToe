#pragma once

#include <functional>
#include <memory>
#include <queue>
#include <vector>

namespace NeuralNetwork
{
    class Node;

    class Edge
    {
    public:
        Edge() = delete;
        Edge(const std::shared_ptr<Node>& pred);
        ~Edge();

        void setEdgeWeight(double val) { m_edgeWeight = val; }
        double getEdgeWeight() const { return m_edgeWeight; }

        virtual bool assignParameters(std::queue<double>& values);
        virtual void getParameters(std::vector<double>& values) const;
        virtual int getNumParameters() const;

        double getValue() const;

        /// updates the edge weight to the weight adjustment value
        /// returns the input node adjustment value
        double handleBackpropagation(double multiplier);

    private:
        double m_edgeWeight;
        std::shared_ptr<Node> m_predecessor;
    };

    class Node
    {
    public:
        Node();
        ~Node() {};

    public:
        virtual bool assignParameters(std::queue<double>& values);
        virtual void getParameters(std::vector<double>& values) const;
        virtual int getNumParameters() const;

    public:
        virtual void updateValue(std::function<double(double, bool)> activationFunction);
        virtual double getError(double targetValue, bool derivative = false) const;

        void setValue(double val) { m_value = val; }
        double getValue() const { return m_value; }

        /// updates the bias and input edge weights to their weight adjustment value
        /// returns the weight adjustment values for all input nodes (all nodes on the previous layer)
        virtual void handleBackpropagation(double targetValue, std::function<double(double, bool)> activationFunction, std::vector<double>& inputValueAdjustments);

    protected:
        double m_value;
    };

    class InnerNode
        : public Node
    {
    public:
        InnerNode() = delete;
        InnerNode(const std::vector<std::shared_ptr<Edge>>& inputEdges);
        ~InnerNode();

    public:
        virtual bool assignParameters(std::queue<double>& params) override;
        virtual void getParameters(std::vector<double>& params) const override;
        virtual int getNumParameters() const override;

    public:
        virtual void updateValue(std::function<double(double, bool)> activationFunction) override;

        // return non-activated computed value
        virtual double calculateBaseValue() const;

        virtual void handleBackpropagation(double targetValue, std::function<double(double, bool)> activationFunction, std::vector<double>& inputValueAdjustments) override;

        void addValue(double val)
        {
            m_value += val;
        }

    private:
        std::vector<std::shared_ptr<Edge>> m_inputEdges;
        double m_bias;
    };
}
