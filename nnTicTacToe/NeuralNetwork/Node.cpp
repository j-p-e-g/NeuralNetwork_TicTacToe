#include "../stdafx.h"

#include "Node.h"

namespace NeuralNetwork
{
    // --------------------------
    // Edge
    // --------------------------
    Edge::Edge(const std::shared_ptr<Node>& pred)
        : m_predecessor(pred)
        , m_edgeWeight(0.f)
    {
    }

    Edge::~Edge()
    {
        m_predecessor = nullptr;
    }

    double Edge::getValue() const
    {
        return m_predecessor->getValue() * m_edgeWeight;
    }

    // --------------------------
    // Node
    // --------------------------
    Node::Node()
        : m_value(0.f)
    {
    }

    bool Node::assignParameters(std::queue<double>& values)
    {
        // nothing to do
        return true;
    }

    void Node::updateValue()
    {
        // nothing to do
    }

    int Node::getNumParameters() const
    {
        return 0;
    }

    // --------------------------
    // InnerNode
    // --------------------------
    InnerNode::InnerNode(const std::vector<std::shared_ptr<Edge>>& inputEdges)
        : m_inputEdges(inputEdges)
        , m_bias(0.f)
    {
    }

    InnerNode::~InnerNode()
    {
        m_inputEdges.clear();
    }

    bool InnerNode::assignParameters(std::queue<double>& values)
    {
        if (static_cast<int>(values.size()) < getNumParameters())
        {
            return false;
        }

        for (auto& edge : m_inputEdges)
        {
            edge->setEdgeWeight(values.front());
            values.pop();
        }

        m_bias = values.front();
        values.pop();

        return true;
    }

    int InnerNode::getNumParameters() const
    {
        return static_cast<int>(m_inputEdges.size()) + 1;
    }

    void InnerNode::updateValue()
    {
        m_value = m_bias;

        // sum of all incoming edge values
        for (const auto& ie : m_inputEdges)
        {
            m_value += ie->getValue();
        }
    }
}
