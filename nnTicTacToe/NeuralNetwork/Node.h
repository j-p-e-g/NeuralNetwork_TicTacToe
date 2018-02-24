#pragma once

#include <vector>
#include <memory>
#include <queue>

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
        virtual int getNumParameters() const;

        double getValue() const;

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
        virtual int getNumParameters() const;

    public:
        virtual void updateValue();

        void setValue(double val) { m_value = val; }
        double getValue() const { return m_value; }

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
        virtual bool assignParameters(std::queue<double>& values) override;
        virtual int getNumParameters() const override;

    public:
        virtual void updateValue() override;

        void addValue(double val)
        {
            m_value += val;
        }

    private:
        std::vector<std::shared_ptr<Edge>> m_inputEdges;
        double m_bias;
    };
}
